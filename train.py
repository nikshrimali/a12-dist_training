"""Train the small GPT on N CPU processes (the "virtual GPUs") with DDP or ZeRO stage 1, 2 or 3.

    uv run torchrun --nproc_per_node 32 train.py --stage 0 --out runs/s0_n32     # DDP, the baseline
    uv run torchrun --nproc_per_node 32 train.py --stage 3 --out runs/s3_n32     # ZeRO-3
    uv run torchrun --nproc_per_node 4 train.py --stage 2 --out runs/smoke --max-steps 5

Every process is one rank. torchrun starts them and sets RANK and WORLD_SIZE; they talk over the
gloo backend, which runs collectives over local TCP. Each rank reads its own slice of every global
batch, so the global batch is micro-batch x N sequences.

Rank 0 writes:
    metrics.jsonl   one line per step: loss, grad norm, lr, step time, time in collectives,
                    time in the optimizer, bytes sent by kind
    config.json     the flags and the model shape
    summary.json    memory (predicted, measured between steps, peak), activations, bytes sent per
                    step, FLOPs per rank
    timeline.json   every allocation and free during one step, for the memory timeline plot
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist

from model import GPT, GPTConfig, count_params, token_loss
from zero import STAGE_NAMES, Comm, Ledger, ZeRO, predicted_bill


def load_tokens(data_dir, split):
    """Open <data_dir>/<split>.bin as a read-only array of uint16 token ids without loading it into RAM."""
    return np.memmap(f"{data_dir}/{split}.bin", dtype=np.uint16, mode="r")


def get_batch(tokens, seq_ids, seq_len):
    """Cut windows of seq_len + 1 tokens out of the array; input is the first seq_len, target the last seq_len."""
    w = seq_len + 1
    batch = torch.from_numpy(np.stack([tokens[i * w : (i + 1) * w] for i in seq_ids]).astype(np.int64))
    return batch[:, :-1], batch[:, 1:]


def lr_at(step, peak, warmup, total, min_ratio):
    """Linear warmup to `peak`, then a cosine curve down to peak * min_ratio (same as a10)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


class saved_activations:
    """Count the bytes autograd keeps for backward, leaving out the weights themselves.

    Like a10's flops.saved_tensors, but it counts CPU tensors and skips tensors that share storage
    with a parameter, because ZeRO's ledger already counts those.
    """

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.seen = {}
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        self._hooks.__exit__(*exc)

    def _pack(self, t):
        key = t.untyped_storage().data_ptr()
        weights = {p.untyped_storage().data_ptr() for p in self.model.parameters()}
        if key not in weights:
            self.seen[key] = t.untyped_storage().nbytes()
        return t

    @property
    def total_bytes(self):
        return sum(self.seen.values())


@torch.no_grad()
def evaluate(engine, tokens, seq_len, n_seqs, rank, world):
    """Held-out loss: every rank scores its own n_seqs windows, one all-reduce adds them up."""
    x, y = get_batch(tokens, range(rank * n_seqs, (rank + 1) * n_seqs), seq_len)
    t = torch.tensor([token_loss(engine.forward(x), y, reduction="sum").item(), y.numel()], dtype=torch.float64)
    dist.all_reduce(t)
    return (t[0] / t[1]).item()


def mb(n):
    return f"{n / 1e6:8.2f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[0, 1, 2, 3], required=True)
    ap.add_argument("--data", default="../a10-pipeline/data/fwe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=4, help="sequences per rank per step")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--eval-every", type=int, default=100, help="0 turns eval off")
    ap.add_argument("--eval-seqs", type=int, default=8, help="held-out sequences per rank")
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--timeline-step", type=int, default=2, help="record every allocation during this step")
    ap.add_argument("--threads", type=int, default=None, help="torch threads per rank (default: cores / N)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 1) // world))

    def say(msg):
        if rank == 0:
            print(msg, flush=True)

    meta = json.load(open(f"{args.data}/meta.json"))
    cfg = GPTConfig(vocab=meta["vocab_size"], context=args.seq_len, layers=args.layers, width=args.width, heads=args.heads)
    torch.manual_seed(args.seed)  # same seed on every rank, so every rank starts from the same weights
    model = GPT(cfg)
    n_params = count_params(model)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    comm, ledger = Comm(), Ledger()
    engine = ZeRO(model, args.stage, comm, ledger, dtype, args.weight_decay, args.grad_clip)
    ledger.reset_peak()
    bill = predicted_bill(args.stage, engine.units, world, engine.bytes_per_number)
    padded = sum(u.padded for u in engine.units)

    tokens_per_rank = args.micro_batch * args.seq_len
    global_tokens = tokens_per_rank * world
    say(f"{STAGE_NAMES[args.stage]} on {world} ranks: {n_params:,} params ({padded:,} after padding to a multiple of {world}), "
        f"{args.dtype} working weights")
    say(f"units: " + ", ".join(f"{u.name} {u.numel:,}" for u in engine.units))
    say(f"global batch: {args.micro_batch} x {world} sequences x {args.seq_len} tokens = {global_tokens:,} tokens per step")
    say("predicted memory per rank between steps:")
    for k, v in bill.items():
        say(f"  {k:<10}{mb(v)}")
    say(f"  {'total':<10}{mb(sum(bill.values()))}")

    train_tokens = load_tokens(args.data, "train")
    eval_tokens = load_tokens(args.data, "eval")
    n_seq = len(train_tokens) // (args.seq_len + 1)
    order = np.random.default_rng(args.seed).permutation(n_seq)
    total_steps = min(args.max_steps, n_seq // (args.micro_batch * world))

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        log = open(f"{args.out}/metrics.jsonl", "w")
        with open(f"{args.out}/config.json", "w") as f:
            json.dump({**vars(args), "world": world, "stage_name": STAGE_NAMES[args.stage], "model": asdict(cfg),
                       "n_params": n_params, "padded_params": padded, "global_tokens": global_tokens,
                       "total_steps": total_steps, "torch_threads": torch.get_num_threads()}, f, indent=2)

    def write(rec):
        if rank == 0:
            log.write(json.dumps(rec) + "\n")
            log.flush()

    if args.eval_every:
        ev = evaluate(engine, eval_tokens, args.seq_len, args.eval_seqs, rank, world)
        write({"type": "eval", "step": 0, "loss": ev})
        say(f"eval step 0: held-out loss {ev:.4f} (random guessing would be ln({cfg.vocab}) = {math.log(cfg.vocab):.4f})")

    activations, timeline, sent_per_step, window = None, None, None, []
    for step in range(total_steps):
        lr = lr_at(step, args.lr, args.warmup, total_steps, args.min_lr_ratio)
        ids = order[(step * world + rank) * args.micro_batch :][: args.micro_batch]
        x, y = get_batch(train_tokens, ids, args.seq_len)
        comm.reset()
        if step + 1 == args.timeline_step:
            ledger.record()
        dist.barrier()  # start every rank's clock together
        t0 = time.perf_counter()
        engine.zero_grad()
        if step == 1:
            with saved_activations(model) as saved:
                logits = engine.forward(x)
                loss_sum = token_loss(logits, y, reduction="sum")
            activations = saved.total_bytes
        else:
            logits = engine.forward(x)
            loss_sum = token_loss(logits, y, reduction="sum")
        # Divide by the token count of the whole global batch, so summing the ranks' gradients
        # gives the gradient of the mean loss. No extra division by N needed.
        engine.backward(loss_sum / global_tokens)
        grad_norm = engine.step(lr)
        dt = time.perf_counter() - t0
        del logits
        if step + 1 == args.timeline_step:
            timeline = ledger.stop()
        if step == 1:
            sent_per_step = dict(comm.sent)

        # Logging only, outside the timed region and outside the byte counter.
        t = torch.tensor([loss_sum.item(), y.numel()], dtype=torch.float64)
        dist.all_reduce(t)
        loss_val = (t[0] / t[1]).item()
        write({"type": "train", "step": step + 1, "tokens": (step + 1) * global_tokens, "loss": loss_val,
               "grad_norm": grad_norm, "lr": lr, "step_time": dt, "comm_time": comm.seconds,
               "opt_time": engine.opt_seconds, "tok_per_s": global_tokens / dt, "sent": dict(comm.sent)})
        window = (window + [dt])[-args.print_every:]
        if step < 3 or (step + 1) % args.print_every == 0 or step + 1 == total_steps:
            avg = sum(window) / len(window)
            say(f"step {step + 1}/{total_steps} loss {loss_val:.4f} grad_norm {grad_norm:.3f} lr {lr:.2e} "
                f"{global_tokens / avg:,.0f} tok/s ({dt * 1000:.0f} ms, collectives {comm.seconds * 1000:.0f} ms, "
                f"optimizer {engine.opt_seconds * 1000:.0f} ms)")
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step + 1 == total_steps):
            ev = evaluate(engine, eval_tokens, args.seq_len, args.eval_seqs, rank, world)
            write({"type": "eval", "step": step + 1, "loss": ev})
            say(f"eval step {step + 1}: held-out loss {ev:.4f}")

    # Every rank's peak, to show the ranks are symmetric.
    peaks = [torch.zeros(1, dtype=torch.int64) for _ in range(world)]
    dist.all_gather(peaks, torch.tensor([ledger.peak_total]))
    peaks = [int(p) for p in peaks]
    steady = dict(ledger.now)
    say("measured memory per rank between steps (rank 0):")
    for k in steady:
        say(f"  {k:<10}{mb(steady[k])}   predicted {mb(bill[k])}")
    say(f"  peak      {mb(ledger.peak_total)}   at peak: " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in ledger.peak.items()))
    say(f"  peak over ranks: min {mb(min(peaks))}, max {mb(max(peaks))}")
    say(f"  activations {mb(activations or 0)}   (the same at every stage: ZeRO does not shard them)")
    say(f"sent per rank per step: " + ", ".join(f"{k} {mb(v)}" for k, v in sent_per_step.items() if v)
        + f", total {mb(sum(sent_per_step.values()))} = {sum(sent_per_step.values()) / (engine.bytes_per_number * padded):.3f} x Psi x {engine.bytes_per_number} bytes")
    if rank == 0:
        with open(f"{args.out}/summary.json", "w") as f:
            json.dump({"stage": args.stage, "stage_name": STAGE_NAMES[args.stage], "world": world, "n_params": n_params,
                       "padded_params": padded, "bytes_per_number": engine.bytes_per_number,
                       "units": [{"name": u.name, "numel": u.numel, "padded": u.padded} for u in engine.units],
                       "predicted": bill, "steady": steady, "peak_total": ledger.peak_total, "peak": ledger.peak,
                       "peak_over_ranks": [min(peaks), max(peaks)], "activations": activations,
                       "sent_per_step": sent_per_step, "tokens_per_rank": tokens_per_rank,
                       "fwd_bwd_flops_per_rank": 6 * n_params * tokens_per_rank,
                       "optimizer_numbers_per_rank": engine.owned_numbers()}, f, indent=2)
        with open(f"{args.out}/timeline.json", "w") as f:
            json.dump(timeline or [], f)
        log.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
