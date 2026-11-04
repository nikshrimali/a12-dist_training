"""Check that every ZeRO stage trains exactly like DDP, and DDP exactly like one process with torch.optim.AdamW.

    uv run pytest test_zero.py -q -s       # about 5 s; -s shows the measured differences

ZeRO changes where numbers live, never what they are. So in fp32, after the same steps on the same
data, every stage must end with the same weights as DDP, and DDP must match a single process that
sees the whole global batch and uses PyTorch's own AdamW and clip_grad_norm_. pytest launches the
worker below with torchrun, once with 4 ranks and once with 3 (3 does not divide the unit sizes,
so the padding code runs too).
"""

import os
import subprocess
import sys

import pytest
import torch

STEPS, MICRO_BATCH, SEQ, LR = 5, 2, 32, 1e-3
CFG = dict(vocab=256, context=SEQ, layers=2, width=64, heads=2)


def worker(out_dir):
    """Runs on every rank: train each stage for STEPS steps, rank 0 saves the gathered weights."""
    import torch.distributed as dist

    from model import GPT, GPTConfig, token_loss
    from zero import ZeRO, Comm, Ledger

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(1)
    data = torch.randint(0, CFG["vocab"], (STEPS, world * MICRO_BATCH, SEQ + 1), generator=gen)
    global_tokens = world * MICRO_BATCH * SEQ
    for stage in range(4):
        torch.manual_seed(0)
        model = GPT(GPTConfig(**CFG))
        engine = ZeRO(model, stage, Comm(), Ledger(), dtype=torch.float32)
        for s in range(STEPS):
            batch = data[s, rank * MICRO_BATCH : (rank + 1) * MICRO_BATCH]
            engine.zero_grad()
            logits = engine.forward(batch[:, :-1])
            engine.backward(token_loss(logits, batch[:, 1:], reduction="sum") / global_tokens)
            engine.step(LR)
        state = engine.full_state()
        if rank == 0:
            torch.save(state, f"{out_dir}/stage{stage}.pt")
    if rank == 0:
        torch.manual_seed(0)
        model = GPT(GPTConfig(**CFG))
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=LR, betas=(0.9, 0.95), eps=1e-8)
        for s in range(STEPS):
            opt.zero_grad()
            token_loss(model(data[s, :, :-1]), data[s, :, 1:]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.save({n: p.detach().clone() for n, p in model.named_parameters()}, f"{out_dir}/reference.pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [4, 3])
def test_every_stage_matches_ddp_and_pytorch(world, tmp_path):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(world),
           "--master_port", str(29600 + world), __file__, "--worker", str(tmp_path)]
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    assert run.returncode == 0, run.stderr[-3000:]
    ref = torch.load(tmp_path / "reference.pt")
    ddp = torch.load(tmp_path / "stage0.pt")
    for name in ref:
        assert torch.allclose(ddp[name], ref[name], atol=1e-5), f"DDP differs from torch AdamW at {name}"
    from model import GPT, GPTConfig
    torch.manual_seed(0)
    init = dict(GPT(GPTConfig(**CFG)).named_parameters())
    moved = max((ref[n] - init[n]).abs().max().item() for n in ref)
    assert moved > 1e-3, "the weights barely moved, so matching them proves nothing"
    for stage in (1, 2, 3):
        got = torch.load(tmp_path / f"stage{stage}.pt")
        for name in ddp:
            diff = (got[name] - ddp[name]).abs().max().item()
            assert diff < 1e-5, f"stage {stage} differs from DDP at {name} by {diff:.2e}"
    worst = {s: max((torch.load(tmp_path / f"stage{s}.pt")[n] - ref[n]).abs().max().item() for n in ref) for s in range(4)}
    print(f"\n{world} ranks: weights moved up to {moved:.1e} from init; largest difference from torch.optim.AdamW: "
          + ", ".join(f"stage {s} {d:.1e}" for s, d in worst.items()))


if __name__ == "__main__" and sys.argv[1:2] == ["--worker"]:
    worker(sys.argv[2])
