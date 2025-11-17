import os, argparse, time
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


from torchvision import datasets, transforms
from models.cifar_vgg import CIFARVGG
from optim.falcon import FALCON
from optim.falcon_v4 import FALCONv4
from optim.falcon_v5 import FALCONv5
from utils import CSVLogger, accuracy, set_seed

def get_data(batch_size=128, workers=4, dataset_fraction=1.0, test_highfreq_noise=0.0):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()

    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    # Dataset fraction for data-efficiency experiments
    if dataset_fraction < 1.0:
        n = len(full_train)
        indices = torch.randperm(n)[:int(n * dataset_fraction)].tolist()
        train = torch.utils.data.Subset(full_train, indices)
    else:
        train = full_train

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)

    # Store test noise for evaluation
    test_loader.test_highfreq_noise = test_highfreq_noise

    return train_loader, test_loader

def get_model():
    return CIFARVGG(name="VGG11", num_classes=10, bn=True)

class HybridOptimizer(torch.optim.Optimizer):
    """Hybrid optimizer: Muon for 2D params, AdamW for others"""
    def __init__(self, params, lr, weight_decay, muon_opt, adamw_opt):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt

    def zero_grad(self, set_to_none=True):
        if self.muon_opt is not None:
            self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.adamw_opt is not None:
            self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable[[], float]] = None):  # type: ignore[override]
        """
        Forward step to contained optimizers.
        """
        if self.muon_opt is not None:
            try:
                self.muon_opt.step(closure)
            except TypeError:
                # Some optimizers may not accept a closure argument
                self.muon_opt.step()

        if self.adamw_opt is not None:
            try:
                self.adamw_opt.step(closure)
            except TypeError:
                self.adamw_opt.step()

def maybe_muon(params, lr, weight_decay, lr_mult=1.0):
    """
    Create Muon hybrid optimizer with proper DDP shim.
    lr_mult: multiplier for Muon LR (for fair comparison)
    """
    import torch
    import torch.distributed as dist

    params_list = [p for p in params if p.requires_grad]

    # Separate 2D and non-2D params
    params_2d = [p for p in params_list if p.ndim == 2]
    params_other = [p for p in params_list if p.ndim != 2]

    all_2d = len(params_other) == 0

    if all_2d:
        print("[INFO] All parameters are 2D, using pure Muon")
    else:
        print(f"[INFO] Found params with dims: {set(p.ndim for p in params_list)}")
        print(f"[INFO] Using hybrid: Muon for {len(params_2d)} 2D params, AdamW for {len(params_other)} non-2D params")

    muon_opt = None
    adamw_opt = None

    # Try to create Muon optimizer for 2D params
    if len(params_2d) > 0:
        try:
            # Patch distributed before importing muon
            if not (dist.is_available() and dist.is_initialized()):
                # Create stubs for single-GPU usage
                dist.get_rank = (lambda _group=None: 0)
                dist.get_world_size = (lambda _group=None: 1)

                def _all_gather_stub(tensor_list, tensor, _group=None, _async_op=False):
                    if tensor_list is not None and len(tensor_list) > 0:
                        tensor_list[0].copy_(tensor)
                    return None

                dist.all_gather = _all_gather_stub
                def _all_gather_object(obj_list, obj, _group=None):
                    if obj_list is not None:
                        obj_list[:] = [obj]
                dist.all_gather_object = _all_gather_object
                dist.barrier = (lambda *_args, **_kwargs: None)
                dist.broadcast_object_list = (lambda *_args, **_kwargs: None)
                dist.all_reduce = (lambda *_args, **_kwargs: None)

            from muon import Muon as ExtMuon
            muon_opt = ExtMuon(params_2d, lr=lr * lr_mult, weight_decay=weight_decay)
            print(f"[INFO] External Muon loaded for 2D params (lr_mult={lr_mult})")
        except Exception as e_ext:
            print(f"[INFO] External Muon failed: {e_ext}")
            try:
                from torch.optim._muon import Muon as TorchMuon
                muon_opt = TorchMuon(params_2d, lr=lr, weight_decay=weight_decay)
                print(f"[INFO] Torch Muon loaded for 2D params")
            except Exception as e_torch:
                print(f"[INFO] Torch Muon failed: {e_torch}")

    # Create AdamW for non-2D params with wd=0 for 1D (BN/bias)
    if len(params_other) > 0 or muon_opt is None:
        adamw_params = params_other if muon_opt is not None else params_list
        # Separate 1D (BN/bias) from others for weight decay
        params_wd = [p for p in adamw_params if p.ndim >= 2]
        params_no_wd = [p for p in adamw_params if p.ndim < 2]
        param_groups = []
        if params_wd:
            param_groups.append({'params': params_wd, 'weight_decay': weight_decay})
        if params_no_wd:
            param_groups.append({'params': params_no_wd, 'weight_decay': 0.0})

        if param_groups:
            adamw_opt = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
        else:
            adamw_opt = None

        if muon_opt is None:
            print("[WARN] Muon unavailable, using pure AdamW")
            return adamw_opt

    # Return hybrid or pure optimizer
    if len(params_other) > 0:
        return HybridOptimizer(params_list, lr, weight_decay, muon_opt, adamw_opt)
    else:
        return muon_opt

def _maybe_denorm(x):
    # Try to detect normalization; if values look outside [0,1], assume CIFAR-10 norm and denorm.
    # CIFAR-10 stats:
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device, dtype=x.dtype).view(1,-1,1,1)
    STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device, dtype=x.dtype).view(1,-1,1,1)
    vmin, vmax = float(x.min()), float(x.max())
    if vmin < -0.1 or vmax > 1.1:   # crude check: likely normalized
        return x * STD + MEAN, (MEAN, STD), True
    return x, (None, None), False

def _maybe_norm(x, stats):
    MEAN, STD = stats
    if MEAN is None: return x
    return (x - MEAN) / STD

def add_highfreq_noise(x: torch.Tensor, sigma_img: float) -> torch.Tensor:
    """
    sigma_img: std of Gaussian noise in pixel space [0,1]. Recommended 0.02–0.06.
    """
    if sigma_img <= 0:
        return x
    # 1) Work in pixel space
    x_img, stats, was_normed = _maybe_denorm(x)
    # 2) Add Gaussian noise in [0,1]
    x_noisy = x_img + torch.randn_like(x_img) * sigma_img
    # 3) Gentle high-pass (depthwise Laplacian)
    lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    lap = lap.repeat(x_noisy.size(1), 1, 1, 1)
    hp = F.conv2d(x_noisy, lap, padding=1, groups=x_noisy.size(1))
    x_noisy = x_noisy + 0.08 * hp  # small blend; keep subtle
    # 4) Clamp and re-normalize if needed
    x_noisy = x_noisy.clamp(0.0, 1.0)
    return _maybe_norm(x_noisy, stats)

def validate(net, loader, device, noise_sigma: float = 0.0):
    net.eval()
    top1 = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if noise_sigma > 0.0:
                x = add_highfreq_noise(x, noise_sigma)
            out = net(x)
            top1 += accuracy(out, y) * x.size(0)
            n += x.size(0)
    return top1 / max(1, n)

def train_epoch(net, loader, optimizer, scheduler, scaler, device, epoch, log, max_norm=None, channels_last=False):
    net.train()
    running = 0.0
    steps = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        with autocast('cuda'):
            out = net(x)
            loss = F.cross_entropy(out, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if max_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        # Update scheduler after each step
        if scheduler is not None:
            if isinstance(scheduler, list):
                # Handle multiple schedulers for Hybrid
                for sched in scheduler:
                    sched.step()
            else:
                scheduler.step()

        # For FALCON: pass epoch into optimizer to update schedule
        if isinstance(optimizer, (FALCON, FALCONv4, FALCONv5)):
            optimizer.set_epoch(epoch)
        elif isinstance(optimizer, HybridOptimizer):
            # Update epoch for both sub-optimizers if they're FALCON
            if optimizer.muon_opt is not None and hasattr(optimizer.muon_opt, 'set_epoch'):
                optimizer.muon_opt.set_epoch(epoch)
            if optimizer.adamw_opt is not None and hasattr(optimizer.adamw_opt, 'set_epoch'):
                optimizer.adamw_opt.set_epoch(epoch)

        running += loss.item()
        steps += 1
        if steps % 100 == 0:
            log.log(epoch, steps, train_loss=running/steps)

    return running / max(1, steps)

def main():
    ap = argparse.ArgumentParser()
    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--optimizer", type=str, default="falcon_v5", choices=["adamw","muon","falcon","falcon_v4","falcon_v5","scion","gluon"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", type=str, default="run")

    # Speed optimizations
    ap.add_argument("--channels-last", action="store_true",
                    help="Use channels_last memory format for speed.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(model) for speed if PyTorch>=2.0.")

    # FALCON v3/v4 args
    ap.add_argument("--rank1-backend", type=str, default="poweriter", choices=["svd","poweriter"])
    ap.add_argument("--poweriter-steps", type=int, default=1)
    ap.add_argument("--mask-interval", type=int, default=5)
    ap.add_argument("--fast-mask", action="store_true")
    ap.add_argument("--apply-stages", type=str, default="3,4", help="Comma-separated stage indices")
    ap.add_argument("--skip-mix-start", type=float, default=0.0)
    ap.add_argument("--skip-mix-end", type=float, default=0.7)
    ap.add_argument("--rank-k", type=int, default=1)
    ap.add_argument("--late-rank-k-epoch", type=int, default=40)
    ap.add_argument("--retain-energy-start", type=float, default=0.90)
    ap.add_argument("--retain-energy-end", type=float, default=0.60)
    ap.add_argument("--freq-wd-beta", type=float, default=0.0)
    ap.add_argument("--min-kernel", type=int, default=3)

    # Muon fairness
    ap.add_argument("--muon-lr-mult", type=float, default=1.0)

    # FALCON v4 speed knobs
    ap.add_argument("--falcon-every", type=int, default=2,
                    help="Apply FALCON filtering every K steps (default 2 for speed).")
    ap.add_argument("--target-opt-ms", type=float, default=0.0,
                    help="Auto-tune mask_interval to target this optimizer step time (ms).")
    ap.add_argument("--fft-mixed-precision", action="store_true",
                    help="Use half precision for FFT operations under autocast.")
    ap.add_argument("--orth-all-2d", action="store_true",
                    help="Apply orthogonal updates to all 2D params (not just classifier).")
    ap.add_argument("--use-external-muon", action="store_true",
                    help="Use external Muon package for 2D params if available.")
    
    # FALCON v5 specific
    ap.add_argument("--falcon-every-start", type=int, default=4,
                    help="v5: Initial falcon_every value (scheduled to falcon-every-end).")
    ap.add_argument("--falcon-every-end", type=int, default=1,
                    help="v5: Final falcon_every value.")
    ap.add_argument("--share-masks-by-shape", action="store_true",
                    help="v5: Share frequency masks across layers with same spatial shape.")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="v5: EMA decay rate for Polyak averaging.")
    ap.add_argument("--no-ema", action="store_true",
                    help="v5: Disable EMA weight averaging.")
    ap.add_argument("--eval-ema", action="store_true",
                    help="Use EMA weights for evaluation (--eval-only mode).")

    # Experiments
    ap.add_argument("--dataset-fraction", type=float, default=1.0)
    ap.add_argument("--test-highfreq-noise", type=float, default=0.0,
                    help="Eval-only: add Gaussian + mild high-pass noise in image space [0,1]. Recommended 0.02–0.06.")
    ap.add_argument("--time-budget-min", type=float, default=0)

    # Eval-only mode
    ap.add_argument("--eval-only", action="store_true",
                    help="Evaluate a saved checkpoint and exit (no training).")
    ap.add_argument("--load", type=str, default="",
                    help="Path to a checkpoint .pt to load in --eval-only mode.")

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, test_loader = get_data(
        batch_size=args.batch_size,
        dataset_fraction=args.dataset_fraction,
        test_highfreq_noise=args.test_highfreq_noise
    )

    net = get_model().to(device)

    # Speed optimizations
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
        print("[INFO] Using channels_last memory format")
    
    if args.compile:
        try:
            net = torch.compile(net)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed ({e}), continuing without compilation")

    # Early eval-only path (no optimizer needed)
    if args.eval_only:
        assert args.load, "--load path required in --eval-only mode"
        assert os.path.exists(args.load), f"Checkpoint not found: {args.load}"
        ckpt = torch.load(args.load, map_location=device)
        # support both {"net": state_dict} and raw state_dict
        state = ckpt.get("net", ckpt)
        net.load_state_dict(state)
        
        # If eval_ema requested, create optimizer to load EMA and swap
        if args.eval_ema:
            # Create a temporary optimizer to load EMA
            apply_stages = [int(x) for x in args.apply_stages.split(",")] if args.apply_stages else [4]
            temp_opt = FALCONv5(
                net.parameters(), lr=args.lr, weight_decay=args.wd,
                total_epochs=args.epochs, use_ema=True,
                apply_stages=apply_stages,
            )
            # Load optimizer state if available
            if "optimizer" in ckpt:
                try:
                    temp_opt.load_state_dict(ckpt["optimizer"])
                    temp_opt.swap_ema(net, use_ema=True)
                    print("[INFO] Evaluating with EMA weights")
                except Exception as e:
                    print(f"[WARN] Could not load EMA weights: {e}")
        
        val_acc = validate(net, test_loader, device, noise_sigma=args.test_highfreq_noise)
        ema_tag = " (EMA)" if args.eval_ema else ""
        print(f"Eval-only{ema_tag} | noise_sigma={args.test_highfreq_noise} | val@1 {val_acc:.2f}")
        return

    # Create optimizer
    if args.optimizer == "adamw":
        # AdamW with proper weight decay policy
        params_wd = [p for p in net.parameters() if p.ndim >= 2 and p.requires_grad]
        params_no_wd = [p for p in net.parameters() if p.ndim < 2 and p.requires_grad]
        param_groups = [
            {'params': params_wd, 'weight_decay': args.wd},
            {'params': params_no_wd, 'weight_decay': 0.0}
        ]
        opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == "muon":
        opt = maybe_muon(net.parameters(), lr=args.lr, weight_decay=args.wd, lr_mult=args.muon_lr_mult)
    elif args.optimizer == "scion":
        try:
            from scion import Scion
            print("[INFO] Using Scion optimizer")
            opt = Scion(net.parameters(), lr=args.lr, weight_decay=args.wd)
        except Exception as e:
            print(f"[WARN] Scion unavailable ({e}), falling back to AdamW")
            params_wd = [p for p in net.parameters() if p.ndim >= 2 and p.requires_grad]
            params_no_wd = [p for p in net.parameters() if p.ndim < 2 and p.requires_grad]
            param_groups = [
                {'params': params_wd, 'weight_decay': args.wd},
                {'params': params_no_wd, 'weight_decay': 0.0}
            ]
            opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == "gluon":
        try:
            from gluon_optimizer import Gluon
            print("[INFO] Using Gluon optimizer")
            opt = Gluon(net.parameters(), lr=args.lr, weight_decay=args.wd)
        except Exception as e:
            print(f"[WARN] Gluon unavailable ({e}), falling back to AdamW")
            params_wd = [p for p in net.parameters() if p.ndim >= 2 and p.requires_grad]
            params_no_wd = [p for p in net.parameters() if p.ndim < 2 and p.requires_grad]
            param_groups = [
                {'params': params_wd, 'weight_decay': args.wd},
                {'params': params_no_wd, 'weight_decay': 0.0}
            ]
            opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == "falcon":
        apply_stages = [int(x) for x in args.apply_stages.split(",")] if args.apply_stages else [3, 4]
        opt = FALCON(
            net.parameters(), lr=args.lr, weight_decay=args.wd,
            retain_energy_start=args.retain_energy_start,
            retain_energy_end=args.retain_energy_end,
            total_epochs=args.epochs,
            rank1_backend=args.rank1_backend,
            poweriter_steps=args.poweriter_steps,
            mask_interval=args.mask_interval,
            fast_mask=args.fast_mask,
            apply_stages=apply_stages,
            skip_mix_start=args.skip_mix_start,
            skip_mix_end=args.skip_mix_end,
            rank_k=args.rank_k,
            late_rank_k_epoch=args.late_rank_k_epoch,
            freq_wd_beta=args.freq_wd_beta,
            min_kernel=args.min_kernel,
        )
    elif args.optimizer == "falcon_v4":
        apply_stages = [int(x) for x in args.apply_stages.split(",")] if args.apply_stages else [4]
        opt = FALCONv4(
            net.parameters(), lr=args.lr, weight_decay=args.wd,
            retain_energy_start=args.retain_energy_start,
            retain_energy_end=args.retain_energy_end,
            total_epochs=args.epochs,
            rank1_backend=args.rank1_backend,
            poweriter_steps=args.poweriter_steps,
            mask_interval=args.mask_interval,
            fast_mask=args.fast_mask,
            apply_stages=apply_stages,
            skip_mix_start=args.skip_mix_start,
            skip_mix_end=args.skip_mix_end,
            rank_k=args.rank_k,
            freq_wd_beta=args.freq_wd_beta,
            min_kernel=args.min_kernel,
            falcon_every=args.falcon_every,
            fft_mixed_precision=args.fft_mixed_precision,
            target_opt_ms=args.target_opt_ms,
            orth_all_2d=args.orth_all_2d,
            use_external_muon=args.use_external_muon,
            muon_lr_mult=args.muon_lr_mult,
        )
    else:  # falcon_v5
        apply_stages = [int(x) for x in args.apply_stages.split(",")] if args.apply_stages else [4]
        opt = FALCONv5(
            net.parameters(), lr=args.lr, weight_decay=args.wd,
            retain_energy_start=args.retain_energy_start,
            retain_energy_end=args.retain_energy_end,
            total_epochs=args.epochs,
            rank1_backend=args.rank1_backend,
            poweriter_steps=args.poweriter_steps,
            mask_interval=args.mask_interval,
            fast_mask=args.fast_mask,
            apply_stages=apply_stages,
            skip_mix_start=args.skip_mix_start,
            skip_mix_end=args.skip_mix_end,
            rank_k=args.rank_k,
            freq_wd_beta=args.freq_wd_beta,
            min_kernel=args.min_kernel,
            falcon_every_start=args.falcon_every_start,
            falcon_every_end=args.falcon_every_end,
            fft_mixed_precision=args.fft_mixed_precision,
            share_masks_by_shape=args.share_masks_by_shape,
            target_opt_ms=args.target_opt_ms,
            orth_all_2d=args.orth_all_2d,
            use_external_muon=args.use_external_muon,
            muon_lr_mult=args.muon_lr_mult,
            ema_decay=args.ema_decay,
            use_ema=(not args.no_ema),
        )

    scaler = GradScaler('cuda')

    # Cosine scheduler - handle Hybrid case
    total_steps = len(train_loader) * args.epochs
    if isinstance(opt, HybridOptimizer):
        # For Hybrid, create schedulers for sub-optimizers
        schedulers = []
        if opt.muon_opt is not None:
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt.muon_opt, T_max=total_steps))
        if opt.adamw_opt is not None:
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt.adamw_opt, T_max=total_steps))
        scheduler = schedulers  # Store list for step() later
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    run_dir = os.path.join("runs", args.exp)
    os.makedirs(run_dir, exist_ok=True)
    log = CSVLogger(os.path.join(run_dir, "metrics.csv"))

    best = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_epoch(net, train_loader, opt, scheduler, scaler, device, epoch, log, channels_last=args.channels_last)
        val_acc = validate(net, test_loader, device, noise_sigma=0.0)
        dt = time.time() - t0
        wall_min = (time.time() - start_time) / 60.0
        
        # Compute images/sec
        imgs_per_sec = len(train_loader.dataset) / dt if dt > 0 else 0
        
        log.log(epoch, -1, train_loss=tr_loss, val_acc=val_acc, epoch_time=dt, wall_min=wall_min, imgs_per_sec=imgs_per_sec)

        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | val@1 {val_acc:.2f} | epoch_time {dt:.1f}s | wall_min {wall_min:.2f} | imgs/s {imgs_per_sec:.0f}")

        if val_acc > best:
            best = val_acc
            save_dict = {
                "net": net.state_dict(),
                "epoch": epoch,
                "best": best,
                "optimizer": opt.state_dict() if hasattr(opt, 'state_dict') else None,
            }
            torch.save(save_dict, os.path.join(run_dir, "best.pt"))

        # Time budget check
        if args.time_budget_min > 0 and wall_min >= args.time_budget_min:
            print(f"Time budget {args.time_budget_min}min reached at epoch {epoch}")
            break

    print(f"Best val@1: {best:.2f}")

if __name__ == "__main__":
    main()
