"""Train a CIFAR model with PCD, as in the paper's experiments. Defaults are the paper's settings.

K = 2, structured pruning (Sec. 5.1): cross-entropy, subject to a group lasso on the conv filters.
    python experiments/train.py --task pruning --arch resnet34 --dataset cifar100 --tau 0.02

K = 3, sparse and low rank (Sec. 5.2): cross-entropy, subject to l1 and nuclear-norm penalties.
    python experiments/train.py --task sparse-lowrank --arch resnet34 --dataset cifar10 --tau 0.02

Each epoch appends one JSON row of metrics to <out>/metrics.jsonl. A checkpoint (including PCD's
normalization state) is written every --save-every epochs, and rerunning the same command resumes it.
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models import ARCHITECTURES, build_model
from objectives import PROX, SECONDARIES, effective_rank, group_sparsity, param_sparsity, rank_reduction

try:
    from pcd import PCD
except ImportError:                                   # running from a clone without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcd import PCD

TASKS = {"pruning": ["group_lasso"], "sparse-lowrank": ["l1", "nuclear"]}
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--arch", choices=ARCHITECTURES, default="resnet34")
    p.add_argument("--dataset", choices=STATS, default="cifar10")
    p.add_argument("--tau", type=float, default=0.02, help="PCD tolerance in [0, 1]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.999, help="EMA decay of PCD's gradient normalization")
    p.add_argument("--eps", type=float, default=1e-8, help="stabilizer of PCD's gradient normalization")
    p.add_argument("--prox", type=float, default=0.0, metavar="LAMBDA",
                   help="apply each secondary's proximal step with threshold LAMBDA*lr after every step; "
                        "not part of PCD, off by default (see experiments/README.md)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default=None, help="run directory (default: runs/<task>/<arch>/<dataset>/tau<tau>_seed<seed>)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=10, help="checkpoint interval in epochs")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="use at most this many train and test batches per epoch (smoke tests)")
    return p.parse_args()


def loaders(dataset, data_dir, batch_size, workers):
    mean, std = STATS[dataset]
    to_tensor = [T.ToTensor(), T.Normalize(mean, std)]
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    train = cls(data_dir, train=True, download=True,
                transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), *to_tensor]))
    test = cls(data_dir, train=False, download=True, transform=T.Compose(to_tensor))
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return (DataLoader(train, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(test, 256, shuffle=False, **kw))


def train_epoch(model, loader, opt, pcd, secondaries, prox, device, limit):
    model.train()
    weights = model.penalized_weights()
    lr = opt.param_groups[0]["lr"]
    log = defaultdict(list)
    for step, (x, y) in enumerate(loader):
        if limit and step >= limit:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ce = F.cross_entropy(model(x), y)
        penalties = [SECONDARIES[name](weights) for name in secondaries]
        info = pcd.backward([ce, *penalties])        # primary first; writes the PCD direction into .grad
        opt.step()
        if prox:
            for name in secondaries:
                PROX[name](weights, prox * lr)
        log["train_ce"].append(ce.item())
        for j, name in enumerate(secondaries, start=1):
            log[f"{name}_loss"].append(penalties[j - 1].item())
            log[f"mu_{name}"].append(info.mu[j - 1])
            log[f"active_{name}"].append(float(j in info.active))
            log[f"cos_{name}"].append(info.cosine[j - 1])
        log["infeasible"].append(float(not info.feasible))
    return {k: float(np.mean(v)) for k, v in log.items()}


@torch.no_grad()
def evaluate(model, loader, device, limit=0):
    model.eval()
    correct = total = 0
    ce = 0.0
    for step, (x, y) in enumerate(loader):
        if limit and step >= limit:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        ce += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return {"test_acc": 100.0 * correct / total, "test_ce": ce / total}


@torch.no_grad()
def structure_metrics(model, task):
    weights = model.penalized_weights()
    metrics = {"param_sparsity": param_sparsity(p for p in model.parameters() if p.requires_grad)}
    if task == "pruning":
        metrics["group_sparsity"] = group_sparsity(weights)
    else:
        metrics.update(penalized_param_sparsity=param_sparsity(weights), effective_rank=effective_rank(weights),
                       rank_reduction=rank_reduction(weights))
    return metrics


def main():
    args = parse_args()
    out = Path(args.out or f"runs/{args.task}/{args.arch}/{args.dataset}/tau{args.tau:g}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    for seed_fn in (random.seed, np.random.seed, torch.manual_seed):
        seed_fn(args.seed)

    num_classes = 10 if args.dataset == "cifar10" else 100
    model = build_model(args.arch, num_classes).to(device)
    train_loader, test_loader = loaders(args.dataset, args.data_dir, args.batch_size, args.workers)
    secondaries = TASKS[args.task]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pcd = PCD(model.parameters(), tau=args.tau, beta=args.beta, eps=args.eps)

    ckpt_path, start = out / "checkpoint.pt", 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        pcd.load_state_dict(ckpt["pcd"])
        start = ckpt["epoch"]
        print(f"resuming {out} from epoch {start}")
    else:
        (out / "metrics.jsonl").unlink(missing_ok=True)
        (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    for epoch in range(start, args.epochs):
        t0, lr = time.time(), opt.param_groups[0]["lr"]
        row = {"epoch": epoch + 1, "lr": lr,
               **train_epoch(model, train_loader, opt, pcd, secondaries, args.prox, device, args.limit_batches)}
        sched.step()
        row.update(evaluate(model, test_loader, device, args.limit_batches))
        row.update(structure_metrics(model, args.task))
        row["seconds"] = time.time() - t0
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        shown = "  ".join(f"{k} {row[k]:.3g}" for k in ("test_acc", "group_sparsity", "param_sparsity",
                                                       "effective_rank") if k in row)
        mu = "  ".join(f"mu_{s} {row[f'mu_{s}']:.3g}" for s in secondaries)
        print(f"epoch {epoch + 1}/{args.epochs}  lr {lr:.2e}  ce {row['train_ce']:.3f}  {shown}  {mu}  "
              f"({row['seconds']:.0f}s)", flush=True)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "pcd": pcd.state_dict(), "config": vars(args)}, ckpt_path)

    if start < args.epochs:
        (out / "final.json").write_text(json.dumps({"config": vars(args), "final": row}, indent=2))


if __name__ == "__main__":
    main()
