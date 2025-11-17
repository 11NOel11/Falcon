"""
FALCON v4: Hybrid Frequency+Orthogonal Optimizer
- Convolutions: Spectral filtering + low-rank (FALCON)
- 2D Linear: Orthogonal updates (Muon-style if available, else Ortho2D-lite)
- Speed optimizations: falcon_every, mask caching, FFT mixed precision, channels_last support
"""
import os
import math
import time
from typing import Iterable, Optional, Dict, Any, List, Tuple, Union, Literal
import torch
from torch.optim.optimizer import Optimizer


def _is_conv_spatial(p: torch.Tensor, min_kernel: int = 3) -> bool:
    """Check if parameter is a conv weight with spatial kernel >= min_kernel."""
    return (p.ndim == 4) and (min(p.shape[-2], p.shape[-1]) >= min_kernel)


def _guess_vgg_stage(shape: torch.Size) -> int:
    """Heuristic stage assignment for VGG-like architectures."""
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


def _poweriter_rankk(Gc: torch.Tensor, steps: int = 1, rank_k: int = 1) -> torch.Tensor:
    """Power iteration for rank-k approximation of complex matrices."""
    try:
        *b, co, ci = Gc.shape
        k = min(rank_k, min(co, ci))
        
        V = torch.randn(*b, ci, k, device=Gc.device, dtype=Gc.real.dtype)
        if Gc.is_complex():
            V = V.to(Gc.dtype)
        
        for _ in range(steps):
            U = Gc @ V
            U, _ = torch.linalg.qr(U)
            V = Gc.conj().transpose(-2, -1) @ U
            V, _ = torch.linalg.qr(V)
        
        U = Gc @ V
        temp = U.conj().transpose(-2, -1) @ Gc @ V
        Grk = U @ temp @ V.conj().transpose(-2, -1)
        return Grk
    except Exception:
        return Gc


def _batched_rankk_svd(Gc: torch.Tensor, rank_k: int = 1) -> torch.Tensor:
    """Batched rank-k projection via SVD."""
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


def falcon_filter_grad(
    g: torch.Tensor,
    retain_energy: float = 0.75,
    rank1_backend: str = "poweriter",
    poweriter_steps: int = 1,
    rank_k: int = 1,
    cached_mask: Optional[torch.Tensor] = None,
    fast_mask: bool = False,
    skip_mix: float = 1.0,
    fft_mixed_precision: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    FALCON v4 gradient filtering with optional mixed precision FFT.
    """
    try:
        device = g.device
        orig_dtype = g.dtype
        
        # Optional: work in half precision for FFT
        if fft_mixed_precision and torch.is_autocast_enabled():
            g_work = g.to(torch.float16)
        else:
            g_work = g
        
        # 1) FFT to frequency domain
        Gf = torch.fft.rfft2(g_work, dim=(-2, -1))
        
        # 2) Build or reuse mask
        return_mask = (cached_mask is None)
        if cached_mask is not None:
            mask = cached_mask
        else:
            energy = (Gf.abs() ** 2).sum(dim=(0, 1))
            total_energy = energy.sum()
            
            if fast_mask:
                # Approximate: keep top-k bins by count
                numel = energy.numel()
                k = max(1, int(retain_energy * numel))
                threshold = torch.kthvalue(energy.flatten(), numel - k).values
                mask = (energy >= threshold).to(g_work.dtype)
            else:
                # Exact: sort by energy
                flat_energy = energy.flatten()
                sorted_energy, indices = torch.sort(flat_energy, descending=True)
                cumsum_energy = torch.cumsum(sorted_energy, dim=0)
                cutoff_idx = torch.searchsorted(cumsum_energy, retain_energy * total_energy)
                cutoff_idx = min(cutoff_idx.item() + 1, len(sorted_energy))
                
                mask = torch.zeros_like(flat_energy)
                mask[indices[:cutoff_idx]] = 1.0
                mask = mask.reshape(energy.shape)
        
        # 3) Apply mask and rank-k
        Gf_masked = Gf * mask.unsqueeze(0).unsqueeze(0)
        
        if rank_k >= min(Gf_masked.shape[0], Gf_masked.shape[1]):
            Gf_filtered = Gf_masked
        else:
            Gf_reshaped = Gf_masked.permute(2, 3, 0, 1)
            
            if rank1_backend == "poweriter":
                Gf_lowrank = _poweriter_rankk(Gf_reshaped, steps=poweriter_steps, rank_k=rank_k)
            else:
                Gf_lowrank = _batched_rankk_svd(Gf_reshaped, rank_k=rank_k)
            
            Gf_filtered = Gf_lowrank.permute(2, 3, 0, 1)
        
        # 4) Inverse FFT
        g_filtered = torch.fft.irfft2(Gf_filtered, s=g.shape[-2:], dim=(-2, -1))
        
        # Convert back to original dtype
        if g_filtered.dtype != orig_dtype:
            g_filtered = g_filtered.to(orig_dtype)
        
        # 5) Skip-mix blending
        if skip_mix < 1.0:
            g_filtered = skip_mix * g_filtered + (1.0 - skip_mix) * g
        
        if return_mask:
            return g_filtered, mask
        else:
            return g_filtered
            
    except Exception as e:
        if os.environ.get("FALCON_DEBUG") == "1":
            print(f"[FALCON] Filtering failed: {e}, using raw gradient")
        if cached_mask is None:
            return g, torch.ones(1, device=g.device)
        return g


def _ortho2d_lite(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Lightweight orthogonal projection for 2D params when Muon unavailable.
    Projects gradient orthogonal to weight matrix rows.
    """
    try:
        if w.ndim != 2 or g.ndim != 2:
            return g
        
        # g_orth = g - W @ (W.T @ g) if shapes compatible
        # Use pseudo-inverse for robustness
        try:
            wt_g = w.t() @ g
            g_orth = g - w @ wt_g
        except Exception:
            # Fallback: least squares
            try:
                solution = torch.linalg.lstsq(w, g).solution
                g_orth = g - w @ solution
            except Exception:
                # Ultimate fallback: return raw
                g_orth = g
        
        return g_orth
    except Exception:
        return g


class FALCONv4(Optimizer):
    """
    FALCON v4: Hybrid Frequency+Orthogonal Optimizer
    
    - Convolutions: Spectral filtering + low-rank updates (FALCON)
    - 2D params: Orthogonal updates (Muon if available, else Ortho2D-lite)
    - Speed: falcon_every, mask caching, FFT mixed precision, auto-tuning
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 5e-4,
        # Spectral/low-rank
        rank1_backend: str = "poweriter",
        poweriter_steps: int = 1,
        rank_k: int = 1,
        retain_energy_start: float = 0.90,
        retain_energy_end: float = 0.60,
        skip_mix_start: float = 0.0,
        skip_mix_end: float = 0.7,
        mask_interval: int = 10,
        fast_mask: bool = True,
        apply_stages: Optional[List[int]] = None,
        min_kernel: int = 3,
        freq_wd_beta: float = 0.0,
        # Speed knobs
        falcon_every: int = 1,
        fft_mixed_precision: bool = True,
        target_opt_ms: float = 0.0,
        # 2D orthogonal
        orth_all_2d: bool = True,
        use_external_muon: bool = True,
        muon_lr_mult: float = 1.0,
        # Bookkeeping
        total_epochs: int = 60,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FALCONv4, self).__init__(params, defaults)
        
        # Config
        self.total_epochs = total_epochs
        self.rank1_backend = rank1_backend
        self.poweriter_steps = poweriter_steps
        self.rank_k = rank_k
        self.retain_energy_start = retain_energy_start
        self.retain_energy_end = retain_energy_end
        self.skip_mix_start = skip_mix_start
        self.skip_mix_end = skip_mix_end
        self.mask_interval = mask_interval
        self.fast_mask = fast_mask
        self.apply_stages = apply_stages if apply_stages is not None else [4]  # default: deepest stage
        self.min_kernel = min_kernel
        self.freq_wd_beta = freq_wd_beta
        self.falcon_every = falcon_every
        self.fft_mixed_precision = fft_mixed_precision
        self.target_opt_ms = target_opt_ms
        self.orth_all_2d = orth_all_2d
        self.use_external_muon = use_external_muon
        self.muon_lr_mult = muon_lr_mult
        
        # State
        self._epoch = 0
        self._step = 0
        self._debug = os.environ.get("FALCON_DEBUG") == "1"
        
        # Timing
        self._timing_samples = []
        self._debug_bins_kept = []
        
        # Partition parameters
        self._partition_params()
        
        # Initialize sub-optimizers for 2D params
        self._init_2d_optimizer()
    
    def _partition_params(self):
        """Partition parameters into conv, 2D, and other groups."""
        self.conv_params_4d = set()  # Use set of ids for O(1) lookup
        self.two_d_params = set()
        self.other_params = set()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p_id = id(p)
                    if _is_conv_spatial(p, self.min_kernel):
                        stage = _guess_vgg_stage(p.shape)
                        if stage in self.apply_stages:
                            self.conv_params_4d.add(p_id)
                        else:
                            self.other_params.add(p_id)
                    elif p.ndim == 2:
                        if self.orth_all_2d:
                            self.two_d_params.add(p_id)
                        else:
                            self.other_params.add(p_id)
                    else:
                        self.other_params.add(p_id)
        
        if self._debug:
            print(f"[FALCON v4] Partitioned: {len(self.conv_params_4d)} conv4D, "
                  f"{len(self.two_d_params)} 2D, {len(self.other_params)} other")
    
    def _init_2d_optimizer(self):
        """Initialize optimizer for 2D params (Muon if available)."""
        self.muon_opt = None
        
        if len(self.two_d_params) == 0:
            return
        
        # Collect actual parameter objects for 2D params
        two_d_param_list = []
        for group in self.param_groups:
            for p in group['params']:
                if id(p) in self.two_d_params:
                    two_d_param_list.append(p)
        
        if self.use_external_muon and len(two_d_param_list) > 0:
            try:
                # Try to import and use external Muon
                from muon import Muon
                lr_2d = self.param_groups[0]['lr'] * self.muon_lr_mult
                wd_2d = self.param_groups[0]['weight_decay']
                self.muon_opt = Muon(two_d_param_list, lr=lr_2d, weight_decay=wd_2d)
                if self._debug:
                    print(f"[FALCON v4] Using external Muon for 2D params (lr_mult={self.muon_lr_mult})")
            except Exception as e:
                if self._debug:
                    print(f"[FALCON v4] Muon unavailable ({e}), using Ortho2D-lite")
                self.muon_opt = None
    
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
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Timing
        start_event = None
        end_event = None
        if torch.cuda.is_available() and self._debug:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        # Schedule values
        retain_energy = self._get_schedule_value(self.retain_energy_start, self.retain_energy_end)
        skip_mix = self._get_schedule_value(self.skip_mix_start, self.skip_mix_end)
        
        # Decide if we apply FALCON filtering this step
        apply_falcon_this_step = (self._step % self.falcon_every == 0)
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['falcon_mask'] = None
                    state['falcon_mask_age'] = 0
                
                state['step'] += 1
                
                # Determine if this param gets FALCON treatment
                p_id = id(p)
                is_falcon_conv = (p_id in self.conv_params_4d)
                is_2d_orth = (p_id in self.two_d_params)
                
                # Process gradient
                if is_falcon_conv and apply_falcon_this_step:
                    # Apply FALCON filtering
                    recompute_mask = (state['falcon_mask'] is None or 
                                    state['falcon_mask_age'] >= self.mask_interval)
                    
                    if recompute_mask:
                        filtered_grad, new_mask = falcon_filter_grad(
                            grad,
                            retain_energy=retain_energy,
                            rank1_backend=self.rank1_backend,
                            poweriter_steps=self.poweriter_steps,
                            rank_k=self.rank_k,
                            cached_mask=None,
                            fast_mask=self.fast_mask,
                            skip_mix=skip_mix,
                            fft_mixed_precision=self.fft_mixed_precision,
                        )
                        state['falcon_mask'] = new_mask
                        state['falcon_mask_age'] = 0
                        
                        if self._debug:
                            bins_kept = new_mask.sum().item() / new_mask.numel()
                            self._debug_bins_kept.append(bins_kept)
                        
                        # Optional freq-domain weight decay
                        if self.freq_wd_beta > 0.0:
                            try:
                                Wf = torch.fft.rfft2(p.data, dim=(-2, -1))
                                hf_mask = 1.0 - state['falcon_mask']
                                decay_factor = 1.0 - self.freq_wd_beta
                                Wf = Wf * (1.0 - hf_mask.unsqueeze(0).unsqueeze(0) * (1.0 - decay_factor))
                                p.data = torch.fft.irfft2(Wf, s=p.data.shape[-2:], dim=(-2, -1))
                            except Exception:
                                pass
                    else:
                        filtered_grad = falcon_filter_grad(
                            grad,
                            retain_energy=retain_energy,
                            rank1_backend=self.rank1_backend,
                            poweriter_steps=self.poweriter_steps,
                            rank_k=self.rank_k,
                            cached_mask=state['falcon_mask'],
                            fast_mask=self.fast_mask,
                            skip_mix=skip_mix,
                            fft_mixed_precision=self.fft_mixed_precision,
                        )
                        state['falcon_mask_age'] += 1
                    
                    grad = filtered_grad
                
                elif is_2d_orth and self.muon_opt is None:
                    # Apply Ortho2D-lite if Muon not available
                    grad = _ortho2d_lite(grad, p.data)
                
                # AdamW update for all params (including FALCON-filtered and ortho2d)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = lr / bias_correction1
                
                # Weight decay (decoupled)
                if wd > 0 and p.ndim >= 2:  # Apply wd to 2D+ params
                    p.data.mul_(1 - lr * wd)
                
                # Update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Step Muon optimizer if available
        if self.muon_opt is not None:
            try:
                self.muon_opt.step()
            except Exception as e:
                if self._debug:
                    print(f"[FALCON v4] Muon step failed: {e}")
        
        self._step += 1
        
        # Timing and auto-tuning
        if torch.cuda.is_available() and self._debug and start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            self._timing_samples.append(elapsed_ms)
            
            # Auto-tune mask_interval based on target
            if self.target_opt_ms > 0 and len(self._timing_samples) >= 10:
                avg_ms = sum(self._timing_samples[-10:]) / 10
                if avg_ms > self.target_opt_ms * 1.2:  # 20% over budget
                    self.mask_interval = min(50, self.mask_interval + 2)
                elif avg_ms < self.target_opt_ms * 0.8:  # 20% under budget
                    self.mask_interval = max(5, self.mask_interval - 1)
        
        # Debug logging
        if self._debug and self._step % 100 == 0 and len(self._debug_bins_kept) > 0:
            avg_bins = sum(self._debug_bins_kept) / len(self._debug_bins_kept)
            avg_ms = sum(self._timing_samples[-100:]) / len(self._timing_samples[-100:]) if self._timing_samples else 0
            print(f"[FALCON v4 Epoch {self._epoch}] bins_kept: {avg_bins:.2%}, "
                  f"retain: {retain_energy:.2f}, skip_mix: {skip_mix:.2f}, "
                  f"step_ms: {avg_ms:.2f}, falcon_every: {self.falcon_every}, "
                  f"mask_interval: {self.mask_interval}")
            self._debug_bins_kept = []
        
        return loss
