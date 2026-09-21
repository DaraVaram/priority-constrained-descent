#!/usr/bin/env bash
# Print the training commands behind the paper's PCD results, one run per line. Hand them to your
# scheduler, or run them locally, e.g.
#   bash experiments/sweeps.sh pruning | parallel -j 4
#   bash experiments/sweeps.sh sparse-lowrank > k3_jobs.txt
# Each run is 300 epochs. --prox replicates the published runs exactly; it is not part of PCD and can be
# dropped (see experiments/README.md).
set -euo pipefail

case "${1:-}" in
  pruning)   # Sec. 5.1, K = 2: 4 architectures x 2 datasets x 5 seeds x 20 tolerances = 800 runs
    for arch in resnet34 densenet121 inception mobilenetv2; do
      prox=1e-3; [ "$arch" = mobilenetv2 ] && prox=5e-4
      for dataset in cifar10 cifar100; do
        for seed in 42 123 7 2024 3407; do
          for tau in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
            echo "python experiments/train.py --task pruning --arch $arch --dataset $dataset --seed $seed --tau $tau --prox $prox"
          done
        done
      done
    done ;;
  sparse-lowrank)   # Sec. 5.2, K = 3: 2 architectures x 2 datasets x 3 seeds x 15 tolerances = 180 runs
    for arch in resnet34 inception; do
      for dataset in cifar10 cifar100; do
        for seed in 7 123 2024; do
          for tau in 0 0.01 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1; do
            echo "python experiments/train.py --task sparse-lowrank --arch $arch --dataset $dataset --seed $seed --tau $tau --prox 1e-3"
          done
        done
      done
    done ;;
  *) echo "usage: bash experiments/sweeps.sh pruning|sparse-lowrank" >&2; exit 1 ;;
esac
