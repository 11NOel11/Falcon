"""
FALCON v5: Ultimate Hybrid Frequency+Orthogonal Optimizer
- Interleaved filtering schedule (every K steps, K: 4→1 over training)
- Adaptive retain-energy per layer with EMA tracking
- Mask sharing across layers with identical spatial shapes
- Muon for 2D params (or Ortho2D-lite fallback)
- EMA weights (Polyak averaging) for evaluation
- Frequency-weighted decoupled weight decay on HF bins
- Auto-tuning mask_interval to meet target step time budget
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
    """Power iteration for rank-k approximation via deflation."""
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
    return_bins_kept: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    FALCON v5 gradient filtering with adaptive energy tracking.
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
                # SVD fallback
                try:
                    U, S, Vh = torch.linalg.svd(Gf_reshaped, full_matrices=False)
                    k = min(rank_k, S.shape[-1])
                    Uk = U[..., :, :k]
                    Sk = S[..., :k].unsqueeze(-1)
                    Vhk = Vh[..., :k, :]
                    Gf_lowrank = Uk @ (Sk * Vhk)
                except Exception:
                    Gf_lowrank = Gf_reshaped
            
            Gf_filtered = Gf_lowrank.permute(2, 3, 0, 1)
        
        # 4) Inverse FFT
        g_filtered = torch.fft.irfft2(Gf_filtered, s=g.shape[-2:], dim=(-2, -1))
        
        # Convert back to original dtype
        if g_filtered.dtype != orig_dtype:
            g_filtered = g_filtered.to(orig_dtype)
        
        # 5) Skip-mix blending
        if skip_mix < 1.0:
            g_filtered = skip_mix * g_filtered + (1.0 - skip_mix) * g
        
        # Compute bins kept fraction if requested
        bins_kept = 0.0
        if return_bins_kept and cached_mask is None:
            bins_kept = mask.sum().item() / mask.numel()
        
        if return_bins_kept:
            if return_mask:
                return g_filtered, mask, bins_kept
            else:
                return g_filtered, bins_kept
        else:
            if return_mask:
                return g_filtered, mask
            else:
                return g_filtered
            
    except Exception as e:
        if os.environ.get("FALCON_DEBUG") == "1":
            print(f"[FALCON] Filtering failed: {e}, using raw gradient")
        if return_bins_kept:
            if cached_mask is None:
                return g, torch.ones(1, device=g.device), 0.0
            else:
                return g, 0.0
        else:
            if cached_mask is None:
                return g, torch.ones(1, device=g.device)
            return g


def _ortho2d_lite(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Lightweight orthogonal projection for 2D params when Muon unavailable.
    Projects gradient orthogonal to weight matrix columns.
    """
    try:
        if w.ndim != 2 or g.ndim != 2:
            return g
        
        # g_orth = g - W @ (W.T @ g)
        try:
            wt_g = w.t() @ g
            g_orth = g - w @ wt_g
        except Exception:
            # Fallback: least squares
            try:
                solution = torch.linalg.lstsq(w, g).solution
                g_orth = g - w @ solution
            except Exception:
                g_orth = g
        
        return g_orth
    except Exception:
        return g


class FALCONv5(Optimizer):
    """
    FALCON v5: Ultimate Hybrid Frequency+Orthogonal Optimizer
    
    - Convolutions: Interleaved spectral filtering with adaptive retain-energy
    - 2D params: Muon optimizer (or Ortho2D-lite fallback)
    - EMA weights for evaluation (Polyak averaging)
    - Auto-tuning and efficiency optimizations
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
        retain_energy_start: float = 0.95,
        retain_energy_end: float = 0.50,
        skip_mix_start: float = 0.0,
        skip_mix_end: float = 0.85,
        min_kernel: int = 3,
        apply_stages: Optional[List[int]] = None,
        # Efficiency
        falcon_every_start: int = 4,
        falcon_every_end: int = 1,
        mask_interval: int = 20,
        fast_mask: bool = True,
        fft_mixed_precision: bool = True,
        share_masks_by_shape: bool = True,
        target_opt_ms: float = 0.0,
        # 2D orthogonal
        use_external_muon: bool = True,
        muon_lr_mult: float = 1.0,
        orth_all_2d: bool = True,
        # Robustness
        freq_wd_beta: float = 0.05,
        # EMA/SWA
        ema_decay: float = 0.999,
        use_ema: bool = True,
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
        super(FALCONv5, self).__init__(params, defaults)
        
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
        self.apply_stages = apply_stages if apply_stages is not None else [4]
        self.min_kernel = min_kernel
        self.freq_wd_beta = freq_wd_beta
        self.falcon_every_start = falcon_every_start
        self.falcon_every_end = falcon_every_end
        self.fft_mixed_precision = fft_mixed_precision
        self.share_masks_by_shape = share_masks_by_shape
        self.target_opt_ms = target_opt_ms
        self.orth_all_2d = orth_all_2d
        self.use_external_muon = use_external_muon
        self.muon_lr_mult = muon_lr_mult
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # State
        self._epoch = 0
        self._step = 0
        self._debug = os.environ.get("FALCON_DEBUG") == "1"
        
        # Timing
        self._timing_samples = []
        self._debug_bins_kept = []
        
        # Shared masks by shape (spatial dimensions)
        self._shared_masks: Dict[Tuple[int, int], torch.Tensor] = {}
        self._shared_mask_ages: Dict[Tuple[int, int], int] = {}
        
        # Partition parameters
        self._partition_params()
        
        # Initialize sub-optimizers for 2D params
        self._init_2d_optimizer()
        
        # Initialize EMA buffers
        self._ema_params: Dict[int, torch.Tensor] = {}
        if self.use_ema:
            self._init_ema()
    
    def _partition_params(self):
        """Partition parameters into conv, 2D, and other groups."""
        self.conv_params_4d = set()  # Store param ids
        self.two_d_params = set()
        self.other_params = set()
        
        # Also store param shapes for mask sharing
        self.conv_param_shapes: Dict[int, Tuple[int, int]] = {}
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p_id = id(p)
                    if _is_conv_spatial(p, self.min_kernel):
                        stage = _guess_vgg_stage(p.shape)
                        if stage in self.apply_stages:
                            self.conv_params_4d.add(p_id)
                            # Store spatial shape for mask sharing
                            self.conv_param_shapes[p_id] = (p.shape[2], p.shape[3])
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
            print(f"[FALCON v5] Partitioned: {len(self.conv_params_4d)} conv4D, "
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
                from muon import Muon
                lr_2d = self.param_groups[0]['lr'] * self.muon_lr_mult
                wd_2d = self.param_groups[0]['weight_decay']
                self.muon_opt = Muon(two_d_param_list, lr=lr_2d, weight_decay=wd_2d)
                if self._debug:
                    print(f"[FALCON v5] Using external Muon for 2D params (lr_mult={self.muon_lr_mult})")
            except Exception as e:
                if self._debug:
                    print(f"[FALCON v5] Muon unavailable ({e}), using Ortho2D-lite")
                self.muon_opt = None
    
    def _init_ema(self):
        """Initialize EMA buffers for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self._ema_params[id(p)] = p.data.clone().detach()
    
    def _update_ema(self):
        """Update EMA parameters (Polyak averaging)."""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        p_id = id(p)
                        if p_id in self._ema_params:
                            self._ema_params[p_id].mul_(self.ema_decay).add_(
                                p.data, alpha=1.0 - self.ema_decay
                            )
    
    def set_epoch(self, epoch: int):
        """Set current epoch for scheduling."""
        self._epoch = epoch
    
    def _get_schedule_value(self, start: float, end: float) -> float:
        """Linear schedule from start to end over total_epochs."""
        if self.total_epochs <= 1:
            return end
        progress = min(1.0, self._epoch / (self.total_epochs - 1))
        return start + (end - start) * progress
    
    def _get_falcon_every(self) -> int:
        """Get current falcon_every value based on schedule."""
        if self.total_epochs <= 1:
            return self.falcon_every_end
        progress = min(1.0, self._epoch / (self.total_epochs - 1))
        # Linear interpolation, round to int
        val = self.falcon_every_start + (self.falcon_every_end - self.falcon_every_start) * progress
        return max(1, int(round(val)))
    
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
        falcon_every = self._get_falcon_every()
        
        # Decide if we apply FALCON filtering this step
        apply_falcon_this_step = (self._step % falcon_every == 0)
        
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
                    state['adaptive_retain_ema'] = retain_energy  # Track per-layer retain
                
                state['step'] += 1
                
                # Determine if this param gets FALCON treatment
                p_id = id(p)
                is_falcon_conv = (p_id in self.conv_params_4d)
                is_2d_orth = (p_id in self.two_d_params)
                
                # Process gradient
                if is_falcon_conv and apply_falcon_this_step:
                    # Apply FALCON filtering with adaptive retain
                    spatial_shape = self.conv_param_shapes.get(p_id, (0, 0))
                    
                    # Check for shared mask
                    use_shared_mask = (self.share_masks_by_shape and spatial_shape in self._shared_masks)
                    
                    if use_shared_mask:
                        cached_mask = self._shared_masks[spatial_shape]
                        mask_age = self._shared_mask_ages[spatial_shape]
                    else:
                        cached_mask = state['falcon_mask']
                        mask_age = state['falcon_mask_age']
                    
                    recompute_mask = (cached_mask is None or mask_age >= self.mask_interval)
                    
                    # Use adaptive retain-energy for this layer
                    layer_retain = state['adaptive_retain_ema']
                    
                    if recompute_mask:
                        filtered_grad, new_mask, bins_kept = falcon_filter_grad(
                            grad,
                            retain_energy=layer_retain,
                            rank1_backend=self.rank1_backend,
                            poweriter_steps=self.poweriter_steps,
                            rank_k=self.rank_k,
                            cached_mask=None,
                            fast_mask=self.fast_mask,
                            skip_mix=skip_mix,
                            fft_mixed_precision=self.fft_mixed_precision,
                            return_bins_kept=True,
                        )
                        
                        # Update adaptive retain: nudge toward global schedule
                        target_retain = retain_energy
                        state['adaptive_retain_ema'] = 0.9 * state['adaptive_retain_ema'] + 0.1 * target_retain
                        state['adaptive_retain_ema'] = max(0.4, min(0.98, state['adaptive_retain_ema']))
                        
                        # Store mask
                        if self.share_masks_by_shape and spatial_shape != (0, 0):
                            self._shared_masks[spatial_shape] = new_mask
                            self._shared_mask_ages[spatial_shape] = 0
                        else:
                            state['falcon_mask'] = new_mask
                            state['falcon_mask_age'] = 0
                        
                        if self._debug:
                            self._debug_bins_kept.append(bins_kept)
                        
                        # Frequency-weighted weight decay
                        if self.freq_wd_beta > 0.0:
                            try:
                                Wf = torch.fft.rfft2(p.data, dim=(-2, -1))
                                hf_mask = 1.0 - new_mask
                                decay_factor = 1.0 - self.freq_wd_beta
                                Wf = Wf * (1.0 - hf_mask.unsqueeze(0).unsqueeze(0) * (1.0 - decay_factor))
                                p.data = torch.fft.irfft2(Wf, s=p.data.shape[-2:], dim=(-2, -1))
                            except Exception:
                                pass
                    else:
                        # Use cached mask
                        filtered_grad, bins_kept = falcon_filter_grad(
                            grad,
                            retain_energy=layer_retain,
                            rank1_backend=self.rank1_backend,
                            poweriter_steps=self.poweriter_steps,
                            rank_k=self.rank_k,
                            cached_mask=cached_mask,
                            fast_mask=self.fast_mask,
                            skip_mix=skip_mix,
                            fft_mixed_precision=self.fft_mixed_precision,
                            return_bins_kept=True,
                        )
                        
                        # Increment mask age
                        if self.share_masks_by_shape and spatial_shape in self._shared_mask_ages:
                            self._shared_mask_ages[spatial_shape] += 1
                        else:
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
                    print(f"[FALCON v5] Muon step failed: {e}")
        
        # Update EMA
        self._update_ema()
        
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
                    self.mask_interval = min(50, self.mask_interval + 5)
                elif avg_ms < self.target_opt_ms * 0.8:  # 20% under budget
                    self.mask_interval = max(10, self.mask_interval - 5)
        
        # Debug logging
        if self._debug and self._step % 100 == 0 and len(self._debug_bins_kept) > 0:
            avg_bins = sum(self._debug_bins_kept) / len(self._debug_bins_kept)
            avg_ms = sum(self._timing_samples[-100:]) / len(self._timing_samples[-100:]) if self._timing_samples else 0
            print(f"[FALCON v5 Epoch {self._epoch}] bins_kept: {avg_bins:.2%}, "
                  f"retain: {retain_energy:.2f}, skip_mix: {skip_mix:.2f}, "
                  f"step_ms: {avg_ms:.2f}, falcon_every: {falcon_every}, "
                  f"mask_interval: {self.mask_interval}")
            self._debug_bins_kept = []
        
        return loss
    
    def swap_ema(self, model: torch.nn.Module, use_ema: bool = True):
        """
        Swap model weights with EMA weights for evaluation.
        Call with use_ema=True before eval, use_ema=False to restore.
        """
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p_id = id(p)
                    if p_id in self._ema_params:
                        if use_ema:
                            # Swap: param <-> EMA
                            p.data, self._ema_params[p_id] = self._ema_params[p_id], p.data.clone()
                        else:
                            # Restore: param <-> EMA (swap back)
                            p.data, self._ema_params[p_id] = self._ema_params[p_id], p.data.clone()
    
    def state_dict(self):
        """Return state dict including EMA parameters."""
        state = super().state_dict()
        if self.use_ema:
            state['ema_params'] = {k: v.clone() for k, v in self._ema_params.items()}
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict including EMA parameters."""
        # Extract EMA params before calling super
        ema_params = state_dict.pop('ema_params', None)
        super().load_state_dict(state_dict)
        
        if ema_params is not None and self.use_ema:
            self._ema_params = ema_params
