"""Run every stage at every world size and collect one table: memory, bytes sent and time per rank.

    uv run sweep.py --out runs/sweep                                  # 4 stages x N in 1..32, 5 steps each, ~15 min
    uv run sweep.py --out runs/sweep --worlds 2 4 --steps 3           # quick check

Each (stage, N) pair is one torchrun launch of train.py, with its console output saved next to
its logs in <out>/s<stage>_n<N>/stdout.txt. Memory and bytes sent do not change after step 2,
so a few steps are enough; time per step is the mean over steps 2 and later (step 1 includes
gloo warming up its connections). Writes <out>/sweep.json, one record per run.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def run_one(stage, world, steps, out, port, extra):
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(world), "--master_port", str(port),
           "train.py", "--stage", str(stage), "--out", out, "--max-steps", str(steps), "--eval-every", "0", *extra]
    t0 = time.time()
    with open(f"{out}/stdout.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    return time.time() - t0


def collect(out):
    """Read summary.json and metrics.jsonl of one run into one flat record."""
    s = json.load(open(f"{out}/summary.json"))
    steps = [json.loads(line) for line in open(f"{out}/metrics.jsonl")]
    steps = [r for r in steps if r["type"] == "train"][1:]
    mean = lambda k: sum(r[k] for r in steps) / len(steps)
    return {**s, "step_time": mean("step_time"), "comm_time": mean("comm_time"), "opt_time": mean("opt_time")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sweep")
    ap.add_argument("--worlds", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=5)
    args, extra = ap.parse_known_args()  # anything else goes to train.py, e.g. --micro-batch 2

    records = []
    for world in args.worlds:
        for stage in args.stages:
            out = f"{args.out}/s{stage}_n{world}"
            secs = run_one(stage, world, args.steps, out, 29800 + 4 * world + stage, extra)
            rec = collect(out)
            records.append(rec)
            print(f"stage {stage} N={world:<3} peak {rec['peak_total'] / 1e6:7.2f} MB   "
                  f"sent {sum(rec['sent_per_step'].values()) / 1e6:6.2f} MB/step   "
                  f"{rec['step_time'] * 1000:6.0f} ms/step   ({secs:.0f} s wall)", flush=True)
            with open(f"{args.out}/sweep.json", "w") as f:
                json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
