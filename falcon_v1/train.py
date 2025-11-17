\
import os, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from models.cifar_vgg import CIFARVGG
from optim.falcon import FALCON
from utils import CSVLogger, accuracy, set_seed

def get_data(batch_size=128, workers=4):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader

def get_model():
    return CIFARVGG(name="VGG11", num_classes=10, bn=True)

def maybe_muon(params, lr, weight_decay):
    try:
        # If torch has experimental muon
        from torch.optim._muon import Muon as TorchMuon
        return TorchMuon(params, lr=lr, weight_decay=weight_decay)
    except Exception:
        try:
            from muon import Muon
            return Muon(params, lr=lr, weight_decay=weight_decay)
        except Exception as e:
            print("[WARN] Muon not available; falling back to AdamW. Install with: pip install git+https://github.com/KellerJordan/Muon")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9,0.999))

def validate(net, loader, device):
    net.eval()
    top1 = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            top1 += accuracy(out, y) * x.size(0)
            n += x.size(0)
    return top1 / max(1, n)

def train_epoch(net, loader, optimizer, scaler, device, epoch, log, max_norm=None):
    net.train()
    running = 0.0
    steps = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            out = net(x)
            loss = F.cross_entropy(out, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if max_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        # for FALCON: pass epoch into optimizer to update schedule
        if isinstance(optimizer, FALCON):
            optimizer.set_epoch(epoch)

        running += loss.item()
        steps += 1
        if steps % 100 == 0:
            log.log(epoch, steps, train_loss=running/steps)

    return running / max(1, steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--optimizer", type=str, default="falcon", choices=["adamw","muon","falcon"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retain-energy-start", type=float, default=0.75)
    ap.add_argument("--retain-energy-end", type=float, default=0.50)
    ap.add_argument("--rank1-backend", type=str, default="svd", choices=["svd","poweriter"])
    ap.add_argument("--min-kernel", type=int, default=3)
    ap.add_argument("--exp", type=str, default="run")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, test_loader = get_data(batch_size=args.batch_size)

    net = get_model().to(device)

    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    elif args.optimizer == "muon":
        opt = maybe_muon(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = FALCON(
            net.parameters(), lr=args.lr, weight_decay=args.wd,
            retain_energy_start=args.retain_energy_start,
            retain_energy_end=args.retain_energy_end,
            total_epochs=args.epochs,
            rank1_backend=args.rank1_backend,
            min_kernel=args.min_kernel,
        )

    scaler = GradScaler()

    run_dir = os.path.join("runs", args.exp)
    os.makedirs(run_dir, exist_ok=True)
    log = CSVLogger(os.path.join(run_dir, "metrics.csv"))

    best = 0.0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_epoch(net, train_loader, opt, scaler, device, epoch, log)
        val_acc = validate(net, test_loader, device)
        log.log(epoch, -1, train_loss=tr_loss, val_acc=val_acc)
        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | val@1 {val_acc:.2f} | epoch_time {dt:.1f}s")
        if val_acc > best:
            best = val_acc
            torch.save({"net": net.state_dict(), "epoch": epoch, "best": best}, os.path.join(run_dir, "best.pt"))

    print("Best val@1:", best)

if __name__ == "__main__":
    main()
