\
import math
from typing import Iterable, Optional, Dict, Any
import torch
from torch.optim.optimizer import Optimizer

def _is_conv3x3_or_larger(p: torch.Tensor) -> bool:
    return (p.ndim == 4) and (min(p.shape[-2], p.shape[-1]) >= 3)

def _batched_rank1_svd(Gc: torch.Tensor) -> torch.Tensor:
    """
    Batched rank-1 projection for complex matrices.
    Gc: [..., Cout, Cin] complex tensor
    Returns: same shape, rank-1 approximation.
    """
    # torch.linalg.svd supports complex batched svd
    U, S, Vh = torch.linalg.svd(Gc, full_matrices=False)
    # Keep only first singular triplet
    U1 = U[..., :, :1]
    S1 = S[..., :1].unsqueeze(-1)
    Vh1 = Vh[..., :1, :]
    Gr1 = U1 @ (S1 * Vh1)
    return Gr1

def _poweriter_rank1(Gc: torch.Tensor, steps:int=1) -> torch.Tensor:
    # Gc: [..., Cout, Cin] complex. Use a simple power iteration.
    *b, co, ci = Gc.shape
    v = torch.randn(*b, ci, 1, device=Gc.device, dtype=Gc.real.dtype)
    v = (v / (v.norm(dim=-2, keepdim=True) + 1e-8))
    for _ in range(steps):
        u = Gc @ v
        u = u / (u.norm(dim=-2, keepdim=True) + 1e-8)
        v = Gc.conj().transpose(-2, -1) @ u
        v = v / (v.norm(dim=-2, keepdim=True) + 1e-8)
    # scalar = u^H G v  (approx top singular value)
    scalar = torch.sum((Gc * (u @ v.conj().transpose(-2, -1))).real, dim=(-2,-1), keepdim=True)
    Gr1 = u @ (scalar * v.conj().transpose(-2, -1))
    return Gr1

def falcon_filter_grad(
    g: torch.Tensor,
    retain_energy: float = 0.75,
    rank1_backend: str = "svd",
) -> torch.Tensor:
    """
    g: [Cout, Cin, kH, kW] gradient (float)
    Steps: FFT -> energy mask -> rank1 per frequency -> iFFT -> real
    """
    kH, kW = g.shape[-2:]
    # Work in float32 and complex64 for stability
    g32 = g.float()
    G = torch.fft.rfft2(g32, dim=(-2, -1))  # complex64
    # Energy map over frequency bins
    E = (G.abs() ** 2).sum(dim=(0,1))  # [Hf, Wf]
    total = E.sum()
    if total <= 0:
        return g  # nothing to do
    # Keep top bins by cumulative energy
    flat = E.flatten()
    vals, idx = torch.sort(flat, descending=True)
    csum = torch.cumsum(vals, dim=0)
    cutoff_idx = (csum <= retain_energy * total).sum()
    mask_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_flat[:max(1,int(cutoff_idx))] = True
    mask = mask_flat.reshape_as(E)
    # Apply mask
    G_masked = G * mask.unsqueeze(0).unsqueeze(0)
    # Rank-1 per frequency bin on kept bins
    # Reshape to batch of [Hf*Wf, Cout, Cin]
    Hf, Wf = G.shape[-2], G.shape[-1]
    B = Hf * Wf
    Gb = G_masked.permute(2,3,0,1).reshape(B, G.shape[0], G.shape[1])  # [B, Cout, Cin]
    if rank1_backend == "poweriter":
        Gb_rank1 = _poweriter_rank1(Gb, steps=1)
    else:
        Gb_rank1 = _batched_rank1_svd(Gb)
    G_rank1 = Gb_rank1.reshape(Hf, Wf, G.shape[0], G.shape[1]).permute(2,3,0,1)
    # Preserve zeros for masked-out bins
    G_filtered = torch.where(mask.unsqueeze(0).unsqueeze(0), G_rank1, torch.zeros_like(G_rank1))
    g_filtered = torch.fft.irfft2(G_filtered, s=(kH, kW), dim=(-2, -1)).real
    return g_filtered.to(g.dtype)

class FALCON(Optimizer):
    r"""AdamW-style optimizer with spectral mask + rank-1 projection on conv kernels."""
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 5e-4,
        retain_energy_start: float = 0.75,
        retain_energy_end: float = 0.50,
        total_epochs: int = 60,
        rank1_backend: str = "svd",
        min_kernel: int = 3,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.retain_energy_start = retain_energy_start
        self.retain_energy_end = retain_energy_end
        self.total_epochs = max(1, total_epochs)
        self.rank1_backend = rank1_backend
        self.min_kernel = min_kernel
        self._epoch = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Decoupled weight decay
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                # Spectral filter + rank-1 for conv kernels >= min_kernel
                if _is_conv3x3_or_larger(p) and min(p.shape[-2], p.shape[-1]) >= self.min_kernel:
                    t = min(1.0, self._epoch / float(self.total_epochs))
                    retain = (1 - t) * self.retain_energy_start + t * self.retain_energy_end
                    try:
                        grad = falcon_filter_grad(grad, retain_energy=retain, rank1_backend=self.rank1_backend)
                    except RuntimeError:
                        # Fallback gracefully
                        pass

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def set_epoch(self, epoch: int):
        self._epoch = epoch
