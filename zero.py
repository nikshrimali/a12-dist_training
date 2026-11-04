"""DDP and ZeRO stages 1, 2 and 3, written out by hand so every byte and every collective is visible.

    from zero import ZeRO, Comm, Ledger
    engine = ZeRO(model, stage=2, comm=Comm(), ledger=Ledger())
    engine.zero_grad()
    logits = engine.forward(x)
    engine.backward(token_loss(logits, y, reduction="sum") / tokens_in_global_batch)
    grad_norm = engine.step(lr)

The four stages differ only in what each rank keeps for itself and what it borrows from the
others. With N ranks and Psi parameters, mixed precision (bf16 weights and gradients, fp32 master
weights and Adam state), each rank holds:

    stage    name     params (2 bytes)   grads (2 bytes)   optimizer (12 bytes)   sent per step
    0        DDP      Psi                Psi               Psi                    2 Psi
    1        ZeRO-1   Psi                Psi               Psi / N                2 Psi
    2        ZeRO-2   Psi                Psi / N           Psi / N                2 Psi
    3        ZeRO-3   Psi / N            Psi / N           Psi / N                3 Psi (2.744 Psi here)

The model is cut into "units": unit 0 holds the embeddings, the final LayerNorm and the output
head (one unit because the output head reuses the embedding matrix), and every transformer block
is one more unit. Each unit is one flat buffer padded to a multiple of N, and rank r owns slice r
of every unit. Stages 2 and 3 move gradients unit by unit during backward; stage 3 also fetches
each unit's weights just before it runs and drops them right after.

Ledger counts the bytes of every model-state buffer the engine allocates or frees, by category,
and keeps the peak. It leaves out scratch space under 1 MB (see ADAM_CHUNK) and the gradient
autograd computes for a parameter before adding it into our buffer. Comm sends every collective through one place and counts the bytes each rank sends, using
the ring-algorithm cost the ZeRO paper uses.
"""

import math
import time

import torch
import torch.distributed as dist

STAGE_NAMES = {0: "DDP", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
CATEGORIES = ("params", "grads", "optimizer")
# Adam and the gradient norm work on this many numbers at a time, so their scratch space stays at
# 256 KB (Adam's fp32 denominator) and 512 KB (the norm's fp64 copy) instead of the size of a unit.
ADAM_CHUNK = 65536


class Ledger:
    """Bytes held right now, by category, plus the peak total and the split at that peak.

    Call record() before a step to also keep every change as an event, which is how the one-step
    memory timeline is drawn. mark(phase) labels the events that follow.
    """

    def __init__(self):
        self.now = dict.fromkeys(CATEGORIES, 0)
        self.peak_total = 0
        self.peak = dict(self.now)
        self.phase = "init"
        self.events = None

    def alloc(self, cat, nbytes, what):
        self.now[cat] += nbytes
        self._changed(what)

    def free(self, cat, nbytes, what):
        self.now[cat] -= nbytes
        self._changed(what)

    def total(self):
        return sum(self.now.values())

    def reset_peak(self):
        """Start measuring the peak from the current level, e.g. after setup is done."""
        self.peak_total = self.total()
        self.peak = dict(self.now)

    def record(self):
        self.events = []
        self.mark(self.phase)

    def stop(self):
        events, self.events = self.events, None
        return events

    def mark(self, phase):
        self.phase = phase
        self._changed(f"{phase} starts")

    def _changed(self, what):
        if self.total() > self.peak_total:
            self.peak_total = self.total()
            self.peak = dict(self.now)
        if self.events is not None:
            self.events.append({"phase": self.phase, "what": what, **self.now})


class Comm:
    """All collectives in one place, so each rank can count what it sends and how long it waits.

    Bytes follow the ring algorithm: a reduce-scatter or an all-gather of an M-byte buffer sends
    (N - 1) / N * M bytes from every rank, and an all-reduce is one of each, 2 (N - 1) / N * M.
    """

    def __init__(self):
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        self.reset()

    def reset(self):
        self.sent = {"all_reduce": 0, "reduce_scatter": 0, "all_gather": 0}
        self.calls = dict.fromkeys(self.sent, 0)
        self.seconds = 0.0

    def _count(self, kind, nbytes, t0):
        self.sent[kind] += round(nbytes * (self.world - 1) / self.world) * (2 if kind == "all_reduce" else 1)
        self.calls[kind] += 1
        self.seconds += time.perf_counter() - t0

    def all_reduce(self, t):
        """Every rank ends with the sum of everyone's t."""
        t0 = time.perf_counter()
        dist.all_reduce(t)
        self._count("all_reduce", t.numel() * t.element_size(), t0)

    def reduce_scatter(self, out, full):
        """Sum everyone's `full`, then rank r keeps only slice r of the sum, in `out`."""
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(out, full)
        self._count("reduce_scatter", full.numel() * full.element_size(), t0)

    def all_gather(self, full, mine):
        """Every rank sends its slice `mine`, and every rank ends with all slices side by side in `full`."""
        t0 = time.perf_counter()
        dist.all_gather_into_tensor(full, mine)
        self._count("all_gather", full.numel() * full.element_size(), t0)


class Unit:
    """A group of parameters that moves as one flat buffer, and this rank's slice of it.

    Attributes:
        numel: real parameter count. padded: numel rounded up to a multiple of N.
        shard: padded / N, the slice each rank owns. lo, hi: this rank's slice of the flat buffer.
        decay: (start, end) ranges inside this rank's slice that get weight decay. Matrices are
            decayed; LayerNorm gains and biases are not, the same rule as a10's train.py.
    """

    def __init__(self, name, params, world, rank):
        self.name = name
        self.params = params
        self.numel = sum(p.numel() for p in params)
        self.padded = math.ceil(self.numel / world) * world
        self.shard = self.padded // world
        self.lo, self.hi = rank * self.shard, (rank + 1) * self.shard
        self.offsets = []
        self.decay = []
        off = 0
        for p in params:
            self.offsets.append(off)
            if p.dim() >= 2:
                a, b = max(off, self.lo), min(off + p.numel(), self.hi)
                if a < b:
                    self.decay.append((a - self.lo, b - self.lo))
            off += p.numel()

    def point_params_at(self, flat):
        """Make every parameter of the unit a view into `flat`, so the model reads the flat buffer."""
        for p, off in zip(self.params, self.offsets):
            p.data = flat[off : off + p.numel()].view(p.shape)

    def point_grads_at(self, flat):
        """Pre-set every .grad as a view into `flat`. Autograd then adds gradients into it in place."""
        for p, off in zip(self.params, self.offsets):
            p.grad = flat[off : off + p.numel()].view(p.shape)


def model_units(model, world, rank):
    """Cut the GPT into unit 0 (embeddings, final LayerNorm, tied output head) and one unit per block."""
    head = [model.tok_emb.weight, model.pos_emb.weight, model.ln_f.weight, model.ln_f.bias]
    units = [Unit("embed+head", head, world, rank)]
    units += [Unit(f"block{i}", list(b.parameters()), world, rank) for i, b in enumerate(model.blocks)]
    assert sum(u.numel for u in units) == sum(p.numel() for p in model.parameters()), "a parameter is in no unit"
    return units


def adam_(p, m, v, g, lr, t, clip, weight_decay, decay, betas=(0.9, 0.95), eps=1e-8):
    """AdamW on flat fp32 tensors, the same arithmetic as torch.optim.AdamW, applied in place.

    Args:
        p, m, v: fp32 master weights, first and second moment. g: gradient, any dtype.
        lr: learning rate. t: step number, from 1. clip: factor from gradient clipping, at most 1.
        decay: (start, end) ranges of p that get weight decay.
    """
    b1, b2 = betas
    for a, b in decay:
        p[a:b].mul_(1 - lr * weight_decay)
    bc1, bc2 = 1 - b1**t, 1 - b2**t
    for i in range(0, p.numel(), ADAM_CHUNK):
        s = slice(i, i + ADAM_CHUNK)
        m[s].mul_(b1).add_(g[s], alpha=(1 - b1) * clip)
        v[s].mul_(b2).addcmul_(g[s], g[s], value=(1 - b2) * clip * clip)
        denom = v[s].sqrt().div_(math.sqrt(bc2)).add_(eps)
        p[s].addcdiv_(m[s], denom, value=-lr / bc1)


class ZeRO:
    """Train one model on N ranks with DDP (stage 0) or ZeRO stage 1, 2 or 3.

    Args:
        model: the GPT, initialized identically on every rank (same seed).
        stage: 0, 1, 2 or 3.
        comm: a Comm. ledger: a Ledger.
        dtype: dtype of the working weights and gradients, torch.bfloat16 for the real runs and
            torch.float32 for the equivalence test. Master weights and Adam state are always fp32.
        weight_decay, grad_clip: AdamW decay and the max global gradient norm.
    """

    def __init__(self, model, stage, comm, ledger, dtype=torch.bfloat16, weight_decay=0.1, grad_clip=1.0):
        assert stage in STAGE_NAMES
        self.model, self.stage, self.comm, self.ledger, self.dtype = model, stage, comm, ledger, dtype
        self.weight_decay, self.grad_clip = weight_decay, grad_clip
        self.world, self.rank = comm.world, comm.rank
        self.units = model_units(model, self.world, self.rank)
        self.t = 0
        self.full = []     # per unit: the whole flat working weights (stage 3: empty between uses)
        self.mine = []     # per unit, stage 3 only: this rank's slice of the working weights
        self.gfull = []    # per unit: the whole flat gradient (stages 2, 3: only during that unit's backward)
        self.gmine = []    # per unit: this rank's slice of the summed gradient (stages 1, 2, 3)
        self.master, self.m, self.v = [], [], []
        b = self.bytes_per_number = torch.finfo(dtype).bits // 8
        for u in self.units:
            flat = torch.zeros(u.padded, dtype=dtype)
            with torch.no_grad():
                for p, off in zip(u.params, u.offsets):
                    flat[off : off + p.numel()] = p.detach().flatten()
            u.point_params_at(flat)
            self.full.append(flat)
            ledger.alloc("params", b * u.padded, f"{u.name} weights")
            # The optimizer state: all of it for DDP, this rank's slice for every ZeRO stage.
            own = flat if stage == 0 else flat[u.lo : u.hi]
            self.master.append(own.float().clone())
            self.m.append(torch.zeros(own.numel()))
            self.v.append(torch.zeros(own.numel()))
            ledger.alloc("optimizer", 12 * own.numel(), f"{u.name} master weights, Adam m and v")
            if stage == 3:
                self.mine.append(flat[u.lo : u.hi].clone())
                ledger.alloc("params", b * u.shard, f"{u.name} weight slice")
                self._drop_weights(u)
            if stage <= 1:
                g = torch.zeros(u.padded, dtype=dtype)
                u.point_grads_at(g)
                self.gfull.append(g)
                ledger.alloc("grads", b * u.padded, f"{u.name} gradients")
                self.gmine.append(None)
            else:
                self.gfull.append(None)
                self.gmine.append(torch.zeros(u.shard, dtype=dtype))
                ledger.alloc("grads", b * u.shard, f"{u.name} gradient slice")
        self.decay = [self._decay_ranges(u) for u in self.units]
        if stage >= 2:
            self._install_hooks()

    # ---- stage 3: fetch weights just in time, drop them right after --------------------------------

    def _fetch_weights(self, i):
        u, b = self.units[i], self.bytes_per_number
        self.full[i].untyped_storage().resize_(b * u.padded)
        self.ledger.alloc("params", b * u.padded, f"gather {u.name} weights")
        self.comm.all_gather(self.full[i], self.mine[i])

    def _drop_weights(self, u):
        i = self.units.index(u)
        self.full[i].untyped_storage().resize_(0)
        self.ledger.free("params", self.bytes_per_number * u.padded, f"drop {u.name} weights")

    # ---- stages 2 and 3: move gradients unit by unit during backward -------------------------------

    def _install_hooks(self):
        """Hooks that run inside forward and backward.

        Every block gets a forward pre-hook (stage 3: fetch its weights) and a forward hook (stage 3:
        drop the weights; both stages: hang a hook on the block's output that fires when backward
        reaches the block). Every parameter gets a hook that fires once its gradient is complete;
        when a unit's last parameter reports in, its gradient is reduce-scattered and freed.
        """
        self.waiting = [len(u.params) for u in self.units]
        for i, block in enumerate(self.model.blocks, start=1):
            if self.stage == 3:
                block.register_forward_pre_hook(lambda mod, args, i=i: self._fetch_weights(i))
            block.register_forward_hook(lambda mod, args, out, i=i: self._after_forward(i, out))
        for i, u in enumerate(self.units):
            for p in u.params:
                p.register_post_accumulate_grad_hook(lambda p, i=i: self._grad_ready(i))

    def _after_forward(self, i, out):
        if self.stage == 3:
            self._drop_weights(self.units[i])
        if out.requires_grad:
            out.register_hook(lambda g, i=i: self._before_backward(i))

    def _before_backward(self, i):
        if self.stage == 3:
            self._fetch_weights(i)
        self._alloc_full_grad(i)

    def _alloc_full_grad(self, i):
        u = self.units[i]
        self.gfull[i] = torch.zeros(u.padded, dtype=self.dtype)
        u.point_grads_at(self.gfull[i])
        self.ledger.alloc("grads", self.bytes_per_number * u.padded, f"{u.name} full gradient")

    def _grad_ready(self, i):
        self.waiting[i] -= 1
        if self.waiting[i]:
            return
        u = self.units[i]
        self.comm.reduce_scatter(self.gmine[i], self.gfull[i])
        for p in u.params:
            p.grad = None
        self.gfull[i] = None
        self.ledger.free("grads", self.bytes_per_number * u.padded, f"{u.name} full gradient")
        if self.stage == 3:
            self._drop_weights(u)

    # ---- the training step -------------------------------------------------------------------------

    def zero_grad(self):
        """Stages 0 and 1 keep full gradient buffers between steps, so clear them. Stages 2 and 3
        overwrite their gradient slices in the reduce-scatter, so there is nothing to clear."""
        for g in self.gfull:
            if g is not None and self.stage <= 1:
                g.zero_()

    def forward(self, x):
        """Run the model. Stage 3 fetches unit 0 first: the embeddings are needed at the very start
        and, because the output head shares the embedding matrix, again at the very end."""
        self.ledger.mark("forward")
        if self.stage == 3:
            self._fetch_weights(0)
        logits = self.model(x)
        if self.stage == 3 and not torch.is_grad_enabled():
            self._drop_weights(self.units[0])
        return logits

    def backward(self, loss):
        """Backpropagate. Stages 2 and 3 open unit 0's full gradient first, because the output head
        (part of unit 0) is the first thing backward reaches, and unit 0 finishes last."""
        self.ledger.mark("backward")
        if self.stage >= 2:
            self.waiting = [len(u.params) for u in self.units]
            self._alloc_full_grad(0)
        loss.backward()
        if self.stage >= 2:
            assert not any(self.waiting), f"units never finished their gradients: {self.waiting}"

    def step(self, lr):
        """Average the gradients, clip, run AdamW on what this rank owns, and share the new weights.

        Returns:
            The global gradient norm before clipping, the same number on every rank.
        """
        self.ledger.mark("optimizer step")
        self.t += 1
        b = self.bytes_per_number
        # 1. Get each number's summed gradient to the rank that owns it.
        if self.stage == 0:
            for g in self.gfull:
                self.comm.all_reduce(g)
            grads = self.gfull
        else:
            if self.stage == 1:
                for i, u in enumerate(self.units):
                    self.gmine[i] = torch.zeros(u.shard, dtype=self.dtype)
                    self.ledger.alloc("grads", b * u.shard, f"{u.name} gradient slice")
                    self.comm.reduce_scatter(self.gmine[i], self.gfull[i])
            grads = self.gmine
        # 2. Global gradient norm. With sharded gradients no rank sees them all, so every rank sums
        # the squares of its slice and one all-reduce of a single number adds the slices up.
        sq = torch.zeros(1, dtype=torch.float64)
        for g in grads:
            for i in range(0, g.numel(), ADAM_CHUNK):
                c = g[i : i + ADAM_CHUNK].double()
                sq += torch.dot(c, c)
        if self.stage > 0:
            self.comm.all_reduce(sq)
        norm = sq.sqrt().item()
        clip = min(1.0, self.grad_clip / (norm + 1e-6))
        # 3. AdamW on the numbers this rank owns: all of them for DDP, 1/N of them for ZeRO.
        t0 = time.perf_counter()
        for i, u in enumerate(self.units):
            adam_(self.master[i], self.m[i], self.v[i], grads[i], lr, self.t, clip, self.weight_decay, self.decay[i])
        self.opt_seconds = time.perf_counter() - t0
        # 4. Turn the fp32 master weights back into working weights, and share them.
        with torch.no_grad():
            for i, u in enumerate(self.units):
                if self.stage == 0:
                    self.full[i].copy_(self.master[i])
                elif self.stage in (1, 2):
                    mine = self.full[i][u.lo : u.hi]
                    mine.copy_(self.master[i])
                    self.comm.all_gather(self.full[i], mine)
                else:
                    self.mine[i].copy_(self.master[i])  # next forward gathers it
        if self.stage == 1:
            for i, u in enumerate(self.units):
                self.gmine[i] = None
                self.ledger.free("grads", b * u.shard, f"{u.name} gradient slice")
        self.ledger.mark("idle")
        return norm

    def _decay_ranges(self, u):
        """Weight-decay ranges in the coordinates of the optimizer's tensors: the whole flat buffer
        for DDP, this rank's slice for ZeRO."""
        if self.stage > 0:
            return u.decay
        out, off = [], 0
        for p in u.params:
            if p.dim() >= 2:
                out.append((off, off + p.numel()))
            off += p.numel()
        return out

    # ---- reading the result out --------------------------------------------------------------------

    def full_state(self):
        """Gather the fp32 master weights from every rank into {name: tensor}, the same on every rank.

        This is what saving a checkpoint looks like with ZeRO: no rank holds the whole model, so
        writing one file means gathering first (or saving one file per rank).
        """
        names = {id(p): n for n, p in self.model.named_parameters()}
        out = {}
        for i, u in enumerate(self.units):
            if self.stage == 0:
                flat = self.master[i]
            else:
                flat = torch.empty(u.padded)
                dist.all_gather_into_tensor(flat, self.master[i])
            for p, off in zip(u.params, u.offsets):
                out[names[id(p)]] = flat[off : off + p.numel()].view(p.shape).clone()
        return out

    def owned_numbers(self):
        """How many numbers this rank's optimizer updates each step (padding included)."""
        return sum(m.numel() for m in self.master)


def predicted_bill(stage, units, world, bytes_per_number=2):
    """What the formulas in the module docstring say each rank holds between steps, in bytes.

    Uses the padded sizes, so the numbers match the ledger exactly.
    """
    full = sum(u.padded for u in units)
    shard = full // world
    return {
        "params": bytes_per_number * (full if stage < 3 else shard),
        "grads": bytes_per_number * (full if stage < 2 else shard),
        "optimizer": 12 * (full if stage == 0 else shard),
    }
