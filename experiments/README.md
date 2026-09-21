# Reproducing the paper's experiments

Two CIFAR studies, both trained with `train.py`:

| Study | Paper | Objectives (primary first) | Architectures |
|---|---|---|---|
| `--task pruning` (K = 2) | Sec. 5.1 | cross-entropy, group lasso over conv output filters | ResNet-34, DenseNet-121, Inception, MobileNetV2 |
| `--task sparse-lowrank` (K = 3) | Sec. 5.2 | cross-entropy, ℓ1, nuclear norm | ResNet-34, Inception |

```bash
pip install -e ".[experiments]"          # from the repository root; adds torchvision
python experiments/train.py --task pruning --arch resnet34 --dataset cifar100 --tau 0.02
python experiments/train.py --task sparse-lowrank --arch resnet34 --dataset cifar10 --tau 0.02
```

CIFAR downloads to `./data` on first use (`--data-dir` to change). A run writes `runs/<task>/<arch>/<dataset>/tau<τ>_seed<seed>/`:

- `metrics.jsonl`: one row per epoch with the test accuracy, the structure metrics below, the mean training losses and PCD's per-step diagnostics averaged over the epoch (`mu_*` multipliers, `active_*` fraction of steps a constraint bound, `cos_*` primary/secondary cosine, `infeasible` fraction).
- `checkpoint.pt`: model, optimizer, schedule and PCD state every `--save-every` epochs. Rerunning the same command resumes from it.
- `final.json`: the configuration and the last epoch's row.

`bash experiments/sweeps.sh pruning` and `bash experiments/sweeps.sh sparse-lowrank` print the full grids behind the paper: 800 and 180 runs of 300 epochs. On one NVIDIA A10G, a K = 2 ResNet-34/CIFAR-100 run takes about 3.6 hours. K = 3 runs cost more, mostly because the nuclear norm needs an SVD of every penalized layer at every step (for every method, not just PCD). For a quick check that everything runs, add `--epochs 2 --limit-batches 5`.

## Metrics

| Logged as | Meaning | Used for |
|---|---|---|
| `test_acc` | top-1 test accuracy (%) | everything |
| `group_sparsity` | % of penalized output filters whose L2 norm is below 1e-4 (native group sparsity) | Sec. 5.1.2 (Fig. 7, Table 3) |
| `param_sparsity` | % of all trainable parameters with \|w\| < 1e-8 | Table 5, Fig. 8 |
| `penalized_param_sparsity` | the same, over the penalized conv weights only | Table 15 |
| `effective_rank` | mean over penalized layers of exp(entropy of the normalized singular values) | Table 5, Fig. 8 |
| `rank_reduction` | % drop in numerical rank; singular values above max(1e-5, 1e-3 σ_max) count | |

All values are taken after the last epoch. They depend on where the schedule ends: weights the penalty drives toward zero stay at a small floor set by the step size until the cosine schedule brings the learning rate to zero, so most of the measured sparsity appears in the last few epochs. This holds for every method under this protocol. Compare runs only at the end of the same schedule.

## Reproducibility checklist

| | |
|---|---|
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8), learning rate 1e-3, no weight decay, no warm-up |
| Schedule | 300 epochs, cosine annealing to 0, stepped once per epoch |
| Batches | 128 for training (last partial batch dropped), 256 for testing |
| Augmentation | random 32×32 crop with 4-pixel padding, horizontal flip, per-dataset normalization |
| PCD | EMA β = 0.999 with bias correction, ε = 1e-8; scalar τ |
| τ grid, K = 2 | 0, 0.01, 0.02, …, 0.1, 0.2, …, 1.0 (20 values; τ = 0 is the unpruned reference) |
| τ grid, K = 3 | 0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, …, 0.8, 1.0 (15 values) |
| Fixed τ | 0.02 wherever the paper reports a single PCD operating point |
| Seeds | K = 2: 42, 123, 7, 2024, 3407. K = 3: 7, 123, 2024 |
| Penalized weights | every Conv2d except ResNet-34's three 1×1 projection shortcuts, DenseNet-121's stem and MobileNetV2's depthwise convs; never BatchNorm or Linear layers (`penalized_weights()` in `models.py`) |
| Groups (K = 2) | one per output filter: 7,616 (ResNet-34), 10,176 (DenseNet-121), 1,984 (Inception), 9,920 (MobileNetV2) |
| Matrices (K = 3) | each conv weight viewed as C_out × (C_in·k·k) |
| Selection | no validation split; baselines at their highest mean test accuracy, PCD at τ = 0.02 unless swept |
| Proximal step | the published runs applied one after every optimizer step: group soft-thresholding (K = 2) or ℓ1 soft-thresholding followed by singular-value thresholding (K = 3), with threshold λ·lr at the current epoch's learning rate, λ = 1e-3 (5e-4 for MobileNetV2). It is not part of PCD; `--prox λ` turns it on and `sweeps.sh` includes it |
| Precision | float32 with PyTorch's default TF32 settings; cuDNN autotuning on |

## Relation to the research code

This is a rewrite of the code that produced the paper, checked against it:

- **Solver.** Fed the same random gradient sequences, `pcd.PCD` writes the same directions as the research solvers, for K = 2 and for every feasible K = 3 configuration. They agree to within 1e-11 (relative) with the research code's normalization and within 2e-10 with the paper's, which the release uses. The one difference is when the K = 3 constraints are exactly incompatible (anti-parallel secondaries). There the release follows Algorithm 1 and falls back to the primary gradient, while the research code's absolute determinant test could return an arbitrary direction. That case never occurred in the published runs (App. B.2).
- **Normalization.** The release uses the paper's s = 1/√(v̂ + ε); the research code used 1/(√v̂ + ε). That accounts for the 2e-10 above.
- **Architectures.** Module names, parameter shapes and the penalized sets are identical, and so is the initialization under the same seed. Checkpoints from the research code load into `models.py` directly.
- **Resuming** restores PCD's normalization state; the research code restarted it.

## Not included

- **Physical pruning.** Accuracy at a fixed parameter-reduction target (Fig. 4; Tables 1, 2 and 6) and the measured FLOPs, CPU latency and model size (Tables 1, 2 and 7; Figs. 5, 6, 14 and 15) come from exporting each trained network with its smallest-norm filters physically removed until the target fraction of parameters is gone. That export is not part of this release yet. The training runs it starts from, and the native sparsity that Sec. 5.1.2 reports, are.
- **Baselines.** This repository contains PCD only.
