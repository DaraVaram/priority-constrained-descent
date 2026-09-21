"""Quickstart: feature selection with PCD, in about half a minute on a CPU.

Only 5 of 40 input features carry signal. The primary objective is the classification loss. The
secondary is a group lasso on the first layer's input columns, which pushes whole features out of the
network. PCD keeps each update as close to the cross-entropy gradient as it can while guaranteeing the
penalty a tau-fraction of the progress its own normalized gradient would make. So tau alone decides how
hard the penalty pushes, with no penalty weight to tune.

    python examples/quickstart.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcd import PCD

FEATURES, INFORMATIVE = 40, 5


def make_data(n, seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, FEATURES, generator=gen)
    w = torch.randn(INFORMATIVE, generator=torch.Generator().manual_seed(123))
    z = x[:, :INFORMATIVE] @ w + 0.8 * torch.sin(2 * x[:, 0]) * x[:, 1]
    return x, (z > 0).long()


def train(tau, epochs=60, batch_size=128, lr=3e-3):
    torch.manual_seed(0)
    x_train, y_train = make_data(4000, seed=1)
    model = nn.Sequential(nn.Linear(FEATURES, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pcd = PCD(model.parameters(), tau=tau)

    for _ in range(epochs):
        for idx in torch.randperm(len(x_train)).split(batch_size):
            task = F.cross_entropy(model(x_train[idx]), y_train[idx])          # primary
            features = model[0].weight.norm(dim=0).sum()                       # secondary: one group per input
            pcd.backward([task, features])                                     # sets .grad to the PCD direction
            opt.step()
        sched.step()
    return model


if __name__ == "__main__":
    x_test, y_test = make_data(2000, seed=2)
    print(" tau   test acc   features kept   informative kept")
    for tau in (0.0, 0.05, 0.1, 0.2, 0.5):
        model = train(tau)
        with torch.no_grad():
            acc = (model(x_test).argmax(1) == y_test).float().mean().item()
            kept = model[0].weight.norm(dim=0) > 1e-2
        print(f"{tau:4.2f}   {100 * acc:6.1f}%   {int(kept.sum()):6d} / {FEATURES}   "
              f"{int(kept[:INFORMATIVE].sum()):8d} / {INFORMATIVE}")
