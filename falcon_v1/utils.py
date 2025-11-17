\
import os, math, time, csv, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["epoch","step","wall_min","train_loss","val_acc"])
        self.f.flush()
        self.t0 = time.time()

    def log(self, epoch, step, train_loss=None, val_acc=None):
        wall_min = (time.time() - self.t0)/60.0
        self.w.writerow([epoch, step, wall_min, 
                         "" if train_loss is None else f"{train_loss:.6f}", 
                         "" if val_acc is None else f"{val_acc:.4f}"])
        self.f.flush()

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / batch_size)).item())
        return res[0]
