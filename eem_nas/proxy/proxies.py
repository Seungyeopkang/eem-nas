"""Zero-cost proxy implementations.

The implementations follow the standard formulations in the zero-cost NAS
literature. Each proxy takes an initialized PyTorch model and (when
data-dependent) one or more minibatches of (image, label) tensors, and
returns a single scalar score.

Higher scores are better in all cases. For SynFlow, ZiCo, SNIP, GradNorm,
Fisher, NWOT, and SWAP this matches the convention adopted by NAS-Bench-Suite-Zero
and the original papers. Jacov is reported on the same higher-is-better
orientation by negating the original Mellor et al. KL-style penalty.

Required input shapes:
* ``x``: ``(B, 3, H, W)`` image tensor on the same device as ``net``.
* ``y``: ``(B,)`` int64 class labels (when applicable).
"""
from __future__ import annotations

from typing import Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The online proxy backend requires PyTorch. "
        "Install it with `pip install torch`."
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_grads(net: nn.Module) -> None:
    for p in net.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _ce_loss(net: nn.Module, x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    return F.cross_entropy(net(x), y)


# ---------------------------------------------------------------------------
# SynFlow (Tanaka et al., 2020) — data-free
# ---------------------------------------------------------------------------


def synflow(net: nn.Module, x: "torch.Tensor" | None = None) -> float:
    """Sum of ``|theta * d/dtheta sum(F_a(1; |theta|))|`` over all parameters.

    The signed-linearization step replaces every parameter with its absolute
    value before the forward pass, then restores the original signs.
    """
    state = {n: p.data.clone() for n, p in net.named_parameters()}
    for p in net.parameters():
        p.data = p.data.abs()
    if x is None:
        # Find the first conv to infer (C, H, W).
        first_conv = next(m for m in net.modules() if isinstance(m, nn.Conv2d))
        c_in = first_conv.in_channels
        x = torch.ones((1, c_in, 32, 32), device=next(net.parameters()).device)
    else:
        x = x.detach()
        x = x.new_ones(x.size())
    _zero_grads(net)
    out = net(x)
    out.sum().backward()
    score = 0.0
    for p in net.parameters():
        if p.grad is not None:
            score += float((p.data * p.grad).abs().sum().item())
    for n, p in net.named_parameters():
        p.data = state[n]
    _zero_grads(net)
    return score


# ---------------------------------------------------------------------------
# SNIP (Lee et al., 2019)
# ---------------------------------------------------------------------------


def snip(net: nn.Module, x: "torch.Tensor", y: "torch.Tensor") -> float:
    _zero_grads(net)
    loss = _ce_loss(net, x, y)
    loss.backward()
    score = 0.0
    for p in net.parameters():
        if p.grad is not None:
            score += float((p.data * p.grad).abs().sum().item())
    return score


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------


def grad_norm(net: nn.Module, x: "torch.Tensor", y: "torch.Tensor") -> float:
    _zero_grads(net)
    loss = _ce_loss(net, x, y)
    loss.backward()
    score = 0.0
    for p in net.parameters():
        if p.grad is not None:
            score += float(p.grad.detach().norm().item())
    return score


# ---------------------------------------------------------------------------
# NWOT / NASWOT (Mellor et al., 2021)
# ---------------------------------------------------------------------------


def nwot(net: nn.Module, x: "torch.Tensor") -> float:
    """``log |det K|`` where ``K`` is the binary-activation kernel.

    For each post-ReLU feature, the binary indicator vector ``c_i`` of
    sample ``i`` and ``K_ij = c_i^T c_j + (1 - c_i)^T (1 - c_j)`` form a
    Gram-style matrix; the score is the log-determinant of the sum across
    activation layers.
    """
    n = x.size(0)
    K = torch.zeros((n, n), device=x.device)
    handles = []

    def make_hook():
        def hook(_module, _inp, out):
            nonlocal K
            c = (out > 0).float().reshape(out.size(0), -1)
            K = K + c @ c.t() + (1.0 - c) @ (1.0 - c).t()
        return hook

    for module in net.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(make_hook()))

    with torch.no_grad():
        net(x)

    for h in handles:
        h.remove()

    # log |det K|; add a small ridge for numerical stability.
    eps = 1e-4
    K = K + eps * torch.eye(n, device=K.device)
    sign, logabs = torch.linalg.slogdet(K)
    return float(logabs.item())


# ---------------------------------------------------------------------------
# Jacov (Mellor / Abdelfattah)
# ---------------------------------------------------------------------------


def jacov(net: nn.Module, x: "torch.Tensor") -> float:
    """Negative log-determinant penalty on the per-sample input Jacobian.

    Returns ``+ sum_r [log(lambda_r + eps) + 1 / (lambda_r + eps)]`` of the
    Jacobian Gram correlation, with the global sign flipped so higher is
    better. ``eps`` is a small ridge for numerical stability.
    """
    x = x.detach().clone().requires_grad_(True)
    out = net(x)
    grads = torch.autograd.grad(out.sum(), x, create_graph=False)[0]
    J = grads.view(grads.size(0), -1)
    Jc = J - J.mean(dim=0, keepdim=True)
    cov = Jc.t() @ Jc / max(J.size(0), 1)
    eps = 1e-4
    cov = cov + eps * torch.eye(cov.size(0), device=cov.device)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=eps)
    penalty = (torch.log(eigvals) + 1.0 / eigvals).sum()
    return float((-penalty).item())


# ---------------------------------------------------------------------------
# Fisher
# ---------------------------------------------------------------------------


def fisher(net: nn.Module, x: "torch.Tensor", y: "torch.Tensor") -> float:
    """Sum of squared activation-times-gradient over all conv/linear outputs.

    A simplified variant of the Fisher proxy used in NAS-Bench-Suite-Zero:
    we register forward hooks on Conv2d and Linear modules, retain their
    output gradients during backprop, and aggregate ``(act * dact)^2``.
    """
    cache: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inp, out):
            cache[name] = out
            out.retain_grad()
        return hook

    handles = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name)))

    _zero_grads(net)
    loss = _ce_loss(net, x, y)
    loss.backward()

    score = 0.0
    for name, act in cache.items():
        if act.grad is not None:
            score += float((act * act.grad).pow(2).sum().item())
    for h in handles:
        h.remove()
    return score


# ---------------------------------------------------------------------------
# ZiCo (Li et al., 2023)
# ---------------------------------------------------------------------------


def zico(net: nn.Module, batches: Iterable[tuple["torch.Tensor", "torch.Tensor"]]) -> float:
    """Per-parameter log(|grad mean| / grad std), summed across parameters.

    Computes the gradient of the cross-entropy loss on each input batch,
    then aggregates the per-parameter mean-to-std ratio across batches.
    Higher is better.
    """
    grads_per_batch: list[torch.Tensor] = []
    for x, y in batches:
        _zero_grads(net)
        loss = _ce_loss(net, x, y)
        loss.backward()
        flat = []
        for p in net.parameters():
            if p.grad is not None:
                flat.append(p.grad.detach().flatten())
        grads_per_batch.append(torch.cat(flat))

    if len(grads_per_batch) < 2:
        # ZiCo needs at least two minibatches to estimate std across batches.
        return float("nan")

    G = torch.stack(grads_per_batch, dim=0)  # (B, num_params)
    mean_abs = G.mean(dim=0).abs()
    std = G.std(dim=0)
    eps = 1e-12
    score = torch.log(mean_abs / (std + eps) + eps).sum()
    return float(score.item())


# ---------------------------------------------------------------------------
# SWAP (Peng et al., ICLR 2024)
# ---------------------------------------------------------------------------


def _gaussian_init_(net: nn.Module, generator: "torch.Generator") -> None:
    """Official SWAP re-initialization: N(0, 1) conv/linear weights, zero
    biases, unit/zero norm affine parameters.

    Weights are drawn on the CPU from ``generator`` and copied to the
    network's device, so the scores do not depend on the device RNG.
    """
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.copy_(torch.randn(m.weight.shape, generator=generator))
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.zero_()


def _pattern_codes(out: "torch.Tensor") -> "torch.Tensor":
    """Encode each unit's binary activation pattern over the batch as integers.

    ``out`` has shape ``(m, ...)``; every unit (channel x spatial position)
    yields an m-bit pattern ``1[a_u(x_i) > 0]``. Patterns are packed into
    ``ceil(m / 32)`` int64 words, giving a ``(units, words)`` tensor in
    which equal rows correspond to equal patterns.
    """
    m = out.size(0)
    z = (out.reshape(m, -1) > 0).t()  # (units, m)
    words = []
    for start in range(0, m, 32):
        chunk = z[:, start:start + 32].to(torch.int64)
        weights = torch.pow(2, torch.arange(chunk.size(1), device=chunk.device,
                                            dtype=torch.int64))
        words.append((chunk * weights).sum(dim=1))  # int64 matmul is CPU-only
    return torch.stack(words, dim=1)


def swap(net: nn.Module, x: "torch.Tensor", *, repeats: int = 5,
         seed: int = 0) -> float:
    """Sample-Wise Activation Pattern count, official plain (unregularized) score.

    For a minibatch ``{x_i}`` of size ``m``, every post-ReLU unit ``u`` of a
    4-D feature map defines the binary vector ``z_u = (1[a_u(x_i) > 0])_i``;
    the score is the number of distinct vectors ``|{z_u}|`` across the whole
    network. Following the official implementation and the EEM-NAS paper,
    the network is re-initialized with Gaussian weights ``repeats`` times
    and the counts are averaged.

    Reference: Peng et al., "SWAP-NAS: Sample-Wise Activation Patterns for
    Ultra-fast NAS", ICLR 2024.
    """
    codes: list[torch.Tensor] = []

    def hook(_module, inp, out):
        if isinstance(inp, tuple) and inp and inp[0].dim() == 4:
            codes.append(_pattern_codes(out.detach()))

    handles = [m.register_forward_hook(hook)
               for m in net.modules() if isinstance(m, nn.ReLU)]
    generator = torch.Generator().manual_seed(int(seed))
    scores = []
    try:
        for _ in range(int(repeats)):
            _gaussian_init_(net, generator)
            codes.clear()
            with torch.no_grad():
                net(x)
            if not codes:
                scores.append(0.0)
                continue
            allc = torch.cat(codes, dim=0)
            n_unique = torch.unique(allc, dim=0).size(0)
            scores.append(float(n_unique))
    finally:
        for h in handles:
            h.remove()
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


PROXY_FNS = {
    "synflow": "synflow",
    "snip": "snip",
    "grad_norm": "grad_norm",
    "nwot": "nwot",
    "jacov": "jacov",
    "fisher": "fisher",
    "zico": "zico",
    "swap": "swap",
}
PROXY_NAMES = tuple(PROXY_FNS.keys())


def is_data_free(name: str) -> bool:
    return name == "synflow"


def needs_multi_batch(name: str) -> bool:
    return name == "zico"


# SWAP is defined on a 64-sample minibatch (official setting, used in the paper).
SWAP_BATCH_SIZE = 64


def default_batch_size(name: str) -> int:
    return SWAP_BATCH_SIZE if name == "swap" else 16
