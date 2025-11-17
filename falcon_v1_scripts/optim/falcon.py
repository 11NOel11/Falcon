"""
FALCON v3: Frequency-Aware Low-rank Convolutional Optimizer
Spectral filtering + low-rank approximation per frequency bin with efficient caching.
"""
import os
import math
from typing import Iterable, Optional, Dict, Any, List, Tuple, Union, Literal
import torch
from torch.optim.optimizer import Optimizer


def _is_conv_spatial(p: torch.Tensor, min_kernel: int = 3) -> bool:
    """Check if parameter is a conv weight with spatial kernel >= min_kernel."""
    return (p.ndim == 4) and (min(p.shape[-2], p.shape[-1]) >= min_kernel)


def _guess_vgg_stage(shape: torch.Size) -> int:
    """
    Heuristic stage assignment for VGG-like architectures based on out_channels.
    Stage 0: 64 channels
    Stage 1: 128 channels
    Stage 2: 256 channels
    Stage 3: 512 channels (early)
    Stage 4: 512 channels (late)
    """
    if len(shape) != 4:
        return -1
    out_channels = shape[0]
    if out_channels <= 64:
        return 0
    elif out_channels <= 128:
        return 1
    elif out_channels <= 256:
        return 2
    elif out_channels <= 384:
        return 3
    else:
        return 4


def _batched_rankk_svd(Gc: torch.Tensor, rank_k: int = 1) -> torch.Tensor:
    """
    Batched rank-k projection for complex matrices.
    Gc: [..., Cout, Cin] complex tensor
    Returns: same shape, rank-k approximation.
    """
    try:
        U, S, Vh = torch.linalg.svd(Gc, full_matrices=False)
        k = min(rank_k, S.shape[-1])
        Uk = U[..., :, :k]
        Sk = S[..., :k].unsqueeze(-1)
        Vhk = Vh[..., :k, :]
        Grk = Uk @ (Sk * Vhk)
        return Grk
    except Exception:
        return Gc


def _poweriter_rankk(Gc: torch.Tensor, steps: int = 1, rank_k: int = 1) -> torch.Tensor:
    """
    Power iteration for rank-k approximation of complex matrices.
    More efficient than SVD for rank_k << min(m,n).
    """
    try:
        *b, co, ci = Gc.shape
        k = min(rank_k, min(co, ci))
        
        # Initialize k random orthonormal vectors
        V = torch.randn(*b, ci, k, device=Gc.device, dtype=Gc.real.dtype)
        if Gc.is_complex():
            V = V.to(Gc.dtype)
        
        for _ in range(steps):
            # Power iteration with orthogonalization
            U = Gc @ V
            U, _ = torch.linalg.qr(U)
            V = Gc.conj().transpose(-2, -1) @ U
            V, _ = torch.linalg.qr(V)
        
        # Final projection
        U = Gc @ V
        # Grk ≈ U @ (U^H @ Gc @ V) @ V^H
        temp = U.conj().transpose(-2, -1) @ Gc @ V
        Grk = U @ temp @ V.conj().transpose(-2, -1)
        return Grk
    except Exception:
        return Gc


def falcon_filter_grad(
    g: torch.Tensor,
    retain_energy: float = 0.75,
    rank1_backend: str = "poweriter",
    poweriter_steps: int = 1,
    rank_k: int = 1,
    cached_mask: Optional[torch.Tensor] = None,
    fast_mask: bool = False,
    skip_mix: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    FALCON v3 gradient filtering in frequency domain.
    
    Args:
        g: Gradient tensor [Cout, Cin, H, W]
        retain_energy: Fraction of spectral energy to keep (0-1)
        rank1_backend: "poweriter" or "svd"
        poweriter_steps: Number of power iteration steps
        rank_k: Target rank for approximation
        cached_mask: Optional precomputed mask to reuse
        fast_mask: If True, approximate mask by top-k count instead of energy
        skip_mix: Blend factor: 1.0=full filtered, 0.0=raw gradient
        
    Returns:
        If cached_mask is None: (filtered_grad, new_mask)
        Otherwise: filtered_grad
    """
    try:
        device = g.device
        dtype = g.dtype
        
        # 1) FFT to frequency domain
        Gf = torch.fft.rfft2(g, dim=(-2, -1))  # [Cout, Cin, H, W//2+1]
        
        # 2) Build or reuse mask
        return_mask = (cached_mask is None)
        if cached_mask is not None:
            mask = cached_mask
        else:
            # Compute energy per frequency bin
            energy = (Gf.abs() ** 2).sum(dim=(0, 1))  # [H, W//2+1]
            total_energy = energy.sum()
            
            if fast_mask:
                # Approximate: keep top-k bins by count
                numel = energy.numel()
                k = max(1, int(retain_energy * numel))
                threshold = torch.kthvalue(energy.flatten(), numel - k).values
                mask = (energy >= threshold).to(dtype)
            else:
                # Exact: sort by energy and cumsum until retain_energy
                flat_energy = energy.flatten()
                sorted_energy, indices = torch.sort(flat_energy, descending=True)
                cumsum_energy = torch.cumsum(sorted_energy, dim=0)
                cutoff_idx = torch.searchsorted(cumsum_energy, retain_energy * total_energy)
                cutoff_idx = min(cutoff_idx.item() + 1, len(sorted_energy))
                
                mask = torch.zeros_like(flat_energy)
                mask[indices[:cutoff_idx]] = 1.0
                mask = mask.reshape(energy.shape)
        
        # 3) Apply mask and rank-k approximation per frequency bin
        Gf_masked = Gf * mask.unsqueeze(0).unsqueeze(0)
        
        # Rank-k per frequency bin
        if rank_k >= min(Gf_masked.shape[0], Gf_masked.shape[1]):
            # Full rank, no approximation needed
            Gf_filtered = Gf_masked
        else:
            # Apply rank-k approximation
            # Reshape to [..., Cout, Cin] for batched operation
            *spatial, co, ci = Gf_masked.shape
            Gf_reshaped = Gf_masked.permute(2, 3, 0, 1)  # [H, W//2+1, Cout, Cin]
            
            if rank1_backend == "poweriter":
                Gf_lowrank = _poweriter_rankk(Gf_reshaped, steps=poweriter_steps, rank_k=rank_k)
            else:  # svd
                Gf_lowrank = _batched_rankk_svd(Gf_reshaped, rank_k=rank_k)
            
            Gf_filtered = Gf_lowrank.permute(2, 3, 0, 1)  # [Cout, Cin, H, W//2+1]
        
        # 4) Inverse FFT back to spatial domain
        g_filtered = torch.fft.irfft2(Gf_filtered, s=g.shape[-2:], dim=(-2, -1))
        
        # 5) Skip-mix blending
        if skip_mix < 1.0:
            g_filtered = skip_mix * g_filtered + (1.0 - skip_mix) * g
        
        if return_mask:
            return g_filtered, mask
        else:
            return g_filtered
            
    except Exception as e:
        # Fallback to raw gradient on any error
        if os.environ.get("FALCON_DEBUG") == "1":
            print(f"[FALCON] Filtering failed: {e}, using raw gradient")
        if cached_mask is None:
            return g, torch.ones(1, device=g.device)
        return g


class FALCON(Optimizer):
    """
    FALCON v3: Frequency-Aware Low-rank Convolutional Optimizer
    
    Applies spectral filtering + low-rank approximation to convolutional gradients
    in targeted stages, with efficient mask caching and skip-connection mixing.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        total_epochs: int = 60,
        rank1_backend: Literal["poweriter", "svd"] = "poweriter",
        poweriter_steps: int = 1,
        rank_k: int = 1,
        late_rank_k_epoch: int = 40,
        retain_energy_start: float = 0.90,
        retain_energy_end: float = 0.60,
        skip_mix_start: float = 0.0,
        skip_mix_end: float = 0.7,
        mask_interval: int = 5,
        fast_mask: bool = False,
        apply_stages: Optional[List[int]] = None,
        freq_wd_beta: float = 0.0,
        min_kernel: int = 3,
    ):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: AdamW momentum coefficients
            eps: AdamW epsilon
            weight_decay: Weight decay coefficient
            total_epochs: Total training epochs for scheduling
            rank1_backend: "poweriter" or "svd" for low-rank approximation
            poweriter_steps: Number of power iteration steps
            rank_k: Target rank (can be upgraded late)
            late_rank_k_epoch: Epoch after which last stage uses rank_k=2
            retain_energy_start: Initial spectral energy retention
            retain_energy_end: Final spectral energy retention
            skip_mix_start: Initial skip connection weight
            skip_mix_end: Final skip connection weight
            mask_interval: Recompute mask every N steps
            fast_mask: Use approximate top-k mask by count
            apply_stages: Which stages to apply FALCON (None=auto detect last two)
            freq_wd_beta: Extra frequency-domain weight decay (0=disabled)
            min_kernel: Minimum kernel size to apply FALCON
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
            
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        super(FALCON, self).__init__(params, defaults)
        
        self.total_epochs = total_epochs
        self.rank1_backend = rank1_backend
        self.poweriter_steps = poweriter_steps
        self.rank_k = rank_k
        self.late_rank_k_epoch = late_rank_k_epoch
        self.retain_energy_start = retain_energy_start
        self.retain_energy_end = retain_energy_end
        self.skip_mix_start = skip_mix_start
        self.skip_mix_end = skip_mix_end
        self.mask_interval = mask_interval
        self.fast_mask = fast_mask
        self.apply_stages = apply_stages if apply_stages is not None else [3, 4]
        self.freq_wd_beta = freq_wd_beta
        self.min_kernel = min_kernel
        
        self._epoch = 0
        self._step = 0
        self._debug = os.environ.get("FALCON_DEBUG") == "1"
        
        # For debug logging
        self._debug_bins_kept = []
        self._debug_timer_start = None
        self._debug_timer_end = None
    
    def set_epoch(self, epoch: int):
        """Set current epoch for scheduling."""
        self._epoch = epoch
    
    def _get_schedule_value(self, start: float, end: float) -> float:
        """Linear schedule from start to end over total_epochs."""
        if self.total_epochs <= 1:
            return end
        progress = min(1.0, self._epoch / (self.total_epochs - 1))
        return start + (end - start) * progress
    
    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if self._debug and self._debug_timer_start is None:
            self._debug_timer_start = torch.cuda.Event(enable_timing=True)
            self._debug_timer_end = torch.cuda.Event(enable_timing=True)
            self._debug_timer_start.record()
        
        # Current schedule values
        retain_energy = self._get_schedule_value(self.retain_energy_start, self.retain_energy_end)
        skip_mix = self._get_schedule_value(self.skip_mix_start, self.skip_mix_end)
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FALCON does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['falcon_mask'] = None
                    state['falcon_mask_age'] = 0
                
                state['step'] += 1
                
                # Determine if FALCON applies
                apply_falcon = False
                if _is_conv_spatial(p, self.min_kernel):
                    stage = _guess_vgg_stage(p.shape)
                    if stage in self.apply_stages:
                        apply_falcon = True
                        
                        # Determine effective rank_k
                        effective_rank_k = self.rank_k
                        if self._epoch >= self.late_rank_k_epoch and stage == max(self.apply_stages):
                            effective_rank_k = 2
                        
                        # Mask recomputation logic
                        recompute_mask = (state['falcon_mask'] is None or 
                                        state['falcon_mask_age'] >= self.mask_interval)
                        
                        if recompute_mask:
                            # Filter and get new mask
                            filtered_grad, new_mask = falcon_filter_grad(
                                grad,
                                retain_energy=retain_energy,
                                rank1_backend=self.rank1_backend,
                                poweriter_steps=self.poweriter_steps,
                                rank_k=effective_rank_k,
                                cached_mask=None,
                                fast_mask=self.fast_mask,
                                skip_mix=skip_mix,
                            )
                            state['falcon_mask'] = new_mask
                            state['falcon_mask_age'] = 0
                            
                            if self._debug:
                                bins_kept = new_mask.sum().item() / new_mask.numel()
                                self._debug_bins_kept.append(bins_kept)
                        else:
                            # Use cached mask
                            filtered_grad = falcon_filter_grad(
                                grad,
                                retain_energy=retain_energy,
                                rank1_backend=self.rank1_backend,
                                poweriter_steps=self.poweriter_steps,
                                rank_k=effective_rank_k,
                                cached_mask=state['falcon_mask'],
                                fast_mask=self.fast_mask,
                                skip_mix=skip_mix,
                            )
                            state['falcon_mask_age'] += 1
                        
                        grad = filtered_grad
                        
                        # Optional frequency-domain weight decay
                        if self.freq_wd_beta > 0.0 and recompute_mask:
                            try:
                                Wf = torch.fft.rfft2(p.data, dim=(-2, -1))
                                hf_mask = 1.0 - state['falcon_mask']
                                decay_factor = 1.0 - self.freq_wd_beta
                                Wf = Wf * (1.0 - hf_mask.unsqueeze(0).unsqueeze(0) * (1.0 - decay_factor))
                                p.data = torch.fft.irfft2(Wf, s=p.data.shape[-2:], dim=(-2, -1))
                            except Exception:
                                pass
                
                # AdamW update
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Weight decay (decoupled)
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        self._step += 1
        
        # Debug logging once per epoch
        if self._debug and len(self._debug_bins_kept) > 0:
            if self._step % 100 == 0:  # Log periodically
                avg_bins = sum(self._debug_bins_kept) / len(self._debug_bins_kept)
                print(f"[FALCON Epoch {self._epoch}] Avg bins kept: {avg_bins:.2%}, "
                      f"retain_energy: {retain_energy:.2f}, skip_mix: {skip_mix:.2f}")
                self._debug_bins_kept = []
                
                if self._debug_timer_end is not None and self._debug_timer_start is not None:
                    self._debug_timer_end.record()
                    torch.cuda.synchronize()
                    elapsed = self._debug_timer_start.elapsed_time(self._debug_timer_end)
                    print(f"[FALCON] Optimizer overhead: {elapsed:.2f}ms per 100 steps")
                    self._debug_timer_start.record()
        
        return loss
