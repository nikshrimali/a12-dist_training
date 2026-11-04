# ZeRO from scratch

DDP, ZeRO-1, ZeRO-2 and ZeRO-3, written by hand in about 450 lines of PyTorch and run on 32 "virtual GPUs" (32 CPU processes on one Mac), to measure what each stage does to one GPU's memory, traffic and work.

## Setup

| | |
|---|---|
| Model | a10's GPT, unchanged: 4 layers, width 256, 4 heads, context 128, tied embeddings, 4,231,680 params |
| Data | FineWeb-Edu from `../a10-pipeline/data/fwe` (25M train tokens, a10's 4,096-token BPE) |
| Virtual GPUs | N CPU processes started by `torchrun`, talking over gloo (local TCP); N = 1 to 32 |
| Hardware | MacBook Air M4 (10 CPU cores, 16 GB, no fan), PyTorch 2.15 |
| Precision | bf16 working weights and gradients, fp32 master weights and AdamW state: the ZeRO paper's 2 + 2 + 12 bytes per parameter |
| Batch | 4 sequences x 128 tokens per rank; at N = 32, 16,384 tokens per step |
| Optimizer | hand-written AdamW (checked against `torch.optim.AdamW`), lr 1e-3, 20 warmup steps, cosine to 1e-4, wd 0.1, clip 1.0 |
| Sharding | model cut into units (embeddings + tied head, then one per block), each a flat buffer padded to a multiple of N; rank r owns slice r of every unit |

What each stage keeps per rank (Ψ = parameter count):

| stage | weights | gradients | optimizer | sent per step |
|---|---|---|---|---|
| DDP | 2Ψ | 2Ψ | 12Ψ | 2Ψ (all-reduce) |
| ZeRO-1 | 2Ψ | 2Ψ | 12Ψ/N | 2Ψ (reduce-scatter + all-gather) |
| ZeRO-2 | 2Ψ | 2Ψ/N | 12Ψ/N | 2Ψ, per unit during backward |
| ZeRO-3 | 2Ψ/N | 2Ψ/N | 12Ψ/N | 3Ψ in the paper, 2.66Ψ here (see Findings) |

## Results

All from `runs.sh` (replayed with outputs in `zero.ipynb`).

**Correctness (`test_zero.py`).** After 5 fp32 steps on 4 ranks and on 3 ranks, every stage ends within 3.4e-7 of a single process running `torch.optim.AdamW` on the whole batch; the weights moved 5.0e-3 from their start. At N = 32 in bf16, {{LOSS_SENTENCE}}

**Memory per rank at N = 32**, measured by a ledger of every buffer the engine allocates. Between steps it equals the formula to the byte.

| stage | weights | gradients | optimizer | between steps | peak in a step |
|---|---|---|---|---|---|
| DDP | 8.46 MB | 8.46 MB | 50.78 MB | **67.71 MB** | 67.71 MB |
| ZeRO-1 | 8.46 MB | 8.46 MB | 1.59 MB | **18.51 MB** | 18.78 MB |
| ZeRO-2 | 8.46 MB | 0.26 MB | 1.59 MB | **10.31 MB** | 14.05 MB |
| ZeRO-3 | 0.26 MB | 0.26 MB | 1.59 MB | **2.12 MB** | 9.59 MB |

Activations are 25.75 MB per rank at every stage. ZeRO does not shard them, so at N = 32 they are 12 times ZeRO-3's model states.

**Traffic per rank per step at N = 32**, ring cost, in units of the model's size in bytes: DDP, ZeRO-1 and ZeRO-2 send **1.94** (= 2 x 31/32), ZeRO-3 sends **2.66**, 1.37 times DDP.

**Work per rank.** Forward + backward is 13.0 GFLOP per rank per step at every stage. AdamW updates 4.23M numbers per rank under DDP and 132K under every ZeRO stage at N = 32. {{TIME_SENTENCE}}

Plots in `plots/`: `memory.png`, `scaling.png` (memory against N), `timeline.png` (one step, allocation by allocation), `comm.png`, `work.png`, `loss.png`.

## Findings we did not plan for

- **bf16 is 270x slower than fp32 on this CPU.** A 512 x 256 by 256 x 1024 matmul takes 80 ms in bf16 or fp16 and 0.30 ms in fp32, one thread, because Apple's fast matrix path is fp32-only. We kept bf16 to keep the paper's byte split, and paid in time: about 20 s per step at N = 32 once the fanless laptop warmed up (12.5 s cold).
- **ZeRO-3 sends 2.66Ψ, not 3Ψ.** Unit 0 (embeddings + tied output head, 26% of the parameters) is gathered once per step and kept, because the head runs first in backward and the embedding last. So the backward all-gather covers only the blocks: (1 + 0.744 + 1) x 31/32 = 2.659, which is what we measured.
- **The peaks are set by unit 0.** ZeRO-2's peak gradients (4.00 MB at N = 32) are unit 0's full gradient, open for the whole of backward, plus one block's. A model with more, smaller units, or untied embeddings, would peak lower.
- **ZeRO-1 peaks above DDP at N = 1** (76.17 against 67.71 MB): the reduce-scatter writes into a separate slice buffer, which at N = 1 is the whole gradient. At N = 32 it costs 0.26 MB.

## Status

Complete on the M4. Timings are trends only: 32 processes share 10 cores and gloo runs over local TCP, so they say nothing about NVLink. Memory and bytes sent do not depend on the hardware and are exact. No prefetch, no gradient accumulation, no checkpoint saving (`full_state()` gathers the weights if you want one).

## Files

| file | what |
|---|---|
| `guide.md` | the walkthrough: the 16 bytes, collectives, each ZeRO stage on 8 parameters by hand, then every run explained |
| `zero.py` | the engine: `Ledger` (byte counting), `Comm` (collectives + byte counts), `Unit`, `ZeRO` (stages 0-3), `adam_`, `predicted_bill` |
| `train.py` | one run on N ranks: `--stage 0|1|2|3`, launched with `torchrun` |
| `sweep.py` | every stage at N = 1, 2, 4, 8, 16, 32, collected into `runs/sweep/sweep.json` |
| `test_zero.py` | every stage against DDP, and DDP against `torch.optim.AdamW`, at 4 and 3 ranks |
| `plot.py` | the six figures in `plots/` |
| `model.py`, `data.py` | copied unchanged from a10-pipeline |
| `runs.sh` | every run behind the results, in order, about 100 minutes |
| `zero.ipynb` | replays `runs.sh`'s logs (or reruns them with `RERUN = True`) and shows the plots |

```bash
uv sync
uv run pytest test_zero.py -q                                                  # 5 s
uv run torchrun --nproc_per_node 32 train.py --stage 3 --out runs/s3_n32       # one run
./runs.sh                                                                      # everything
```

If `../a10-pipeline/data/fwe` is missing, build it with `uv run data.py --out data/fwe --train-tokens 25M --eval-tokens 500K` and pass `--data data/fwe`.
