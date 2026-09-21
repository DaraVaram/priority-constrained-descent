"""Why priority matters: the two-objective toy of the paper's Fig. 1.

The primary L1 is a double well with minima at (-2, 0) and (2, 0). The secondary L2 is the l1-distance to
a small box around (2, 0), so only the right-hand minimum is good for both. Started in the left basin, a
weighted sum settles where the two gradients cancel (a conflict equilibrium, not a minimum of L1), while
PCD keeps the secondary making progress, crosses the barrier and stops at (2, 0), where both objectives
are stationary. Both runs use Adam, as in the paper.

    python examples/toy_conflict.py                 # print where each method ends
    python examples/toy_conflict.py --plot toy.png  # also draw the trajectories (needs matplotlib)
"""
import argparse

import torch

from pcd import PCD

STAR = torch.tensor([2.0, 0.0], dtype=torch.float64)
INITS = [(-2.55, 0.40), (-2.90, 0.12), (-1.65, -0.55)]


def primary(theta):
    return (theta[..., 0] ** 2 - 4) ** 2 / 4 + theta[..., 1] ** 2 / 2


def secondary(theta):
    return torch.relu((theta - STAR).abs() - 0.5).sum(-1)


def run(method, init, steps=3000, lr=0.01, tau=0.3):
    theta = torch.nn.Parameter(torch.tensor(init, dtype=torch.float64))
    opt = torch.optim.Adam([theta], lr=lr)
    pcd = PCD([theta], tau=tau)
    path = [theta.detach().clone()]
    for _ in range(steps):
        opt.zero_grad()
        if method == "PCD":
            pcd.backward([primary(theta), secondary(theta)])
        else:
            (0.5 * primary(theta) + 0.5 * secondary(theta)).backward()
        opt.step()
        path.append(theta.detach().clone())
    return torch.stack(path)


def plot(paths, filename):
    import matplotlib.pyplot as plt

    xs, ys = torch.meshgrid(torch.linspace(-3.5, 3.5, 400), torch.linspace(-2, 2, 240), indexing="xy")
    grid = torch.stack([xs, ys], -1).double()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.contour(xs, ys, primary(grid), levels=[0.05, 0.3, 1, 2, 4, 7, 11], colors="#9aa0b4", linewidths=0.8)
    ax.contourf(xs, ys, secondary(grid) == 0, levels=[0.5, 1.5], colors=["#e8dcf4"])
    for (method, i), path in sorted(paths.items(), key=lambda kv: kv[0][0] != "PCD"):   # PCD first, WS on top
        pcd = method == "PCD"
        ax.plot(path[:, 0], path[:, 1], color="#6A1B9A" if pcd else "#EF6C00", lw=2.4 if pcd else 1.6,
                label=None if i else ("PCD" if pcd else "Weighted sum"))
        ax.plot(*path[0], "o", color="#1a1523", ms=5)
        if not pcd:
            ax.plot(*path[-1], "X", color="#EF6C00", ms=11, mec="white",
                    label=None if i else "weighted sum stalls (conflict equilibrium)")
    ax.plot(*STAR, "*", color="#6A1B9A", ms=16, label="both objectives stationary")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"saved {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", metavar="FILE", help="save a figure of the trajectories")
    args = parser.parse_args()

    paths = {}
    for method in ("Weighted sum", "PCD"):
        for i, init in enumerate(INITS):
            path = paths[method, i] = run(method, init)
            end = path[-1]
            print(f"{method:>12} from ({init[0]:+.2f}, {init[1]:+.2f}) -> ({end[0]:+.3f}, {end[1]:+.3f})   "
                  f"L1 = {primary(end):.4f}   L2 = {secondary(end):.4f}")
    if args.plot:
        plot(paths, args.plot)
