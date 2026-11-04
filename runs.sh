#!/bin/sh
# Every run behind the README, the plots and the guide, in order. About 100 minutes on a fanless
# M4, which slows down as it heats up. Each step writes its own log under runs/ for zero.ipynb.
set -e
mkdir -p runs
uv run pytest test_zero.py -q -s > runs/test.log 2>&1
uv run sweep.py --out runs/sweep --steps 5 > runs/sweep.log 2>&1
for s in 0 1 2 3; do
  uv run torchrun --nproc_per_node 32 --master_port 2990$s train.py --stage $s --out runs/s${s}_n32 \
    --max-steps 60 --eval-every 30 > runs/s${s}_n32.log 2>&1
done
uv run plot.py --runs runs --out plots > runs/plot.log 2>&1
