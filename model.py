"""A small GPT, written out in full so every matmul and every tensor shape is visible.

    from model import GPT, GPTConfig
    model = GPT(GPTConfig())                      # 6 layers, width 384, 6 heads, context 256
    logits = model(x)                             # x: (B, T) token ids -> (B, T, V) scores
    loss = token_loss(logits, y)                  # mean cross-entropy over real tokens

Letters used for dimensions everywhere in this project:
    B  batch: how many sequences we process side by side
    T  time: positions in a sequence (tokens), at most `context`
    C  channels: the width of the vector that represents one token (384)
    H  heads: how many independent attention patterns each layer computes (6)
    D  head dim: channels per head, C / H (64)
    V  vocab: how many different tokens exist (4,096)

Pass `trace=[]` to forward() and every intermediate tensor's name and shape is appended to the
list, which is how train.py --trace-shapes writes shapes.txt.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """The numbers that define the model's size. Changing them changes N, the parameter count.

    Attributes:
        vocab: V, number of distinct tokens. Must match the tokenizer.
        context: longest sequence the model can read, sets the size of the position table.
        layers: L, how many transformer blocks are stacked.
        width: C, channels per token.
        heads: H, attention heads per block. `width` must divide evenly by it.
        attn: "sdpa" uses PyTorch's fused attention kernel, "manual" spells out softmax(QK^T)V.
            Both compute the same numbers; the ladder compares their speed.
    """

    vocab: int = 4096
    context: int = 256
    layers: int = 6
    width: int = 384
    heads: int = 6
    attn: str = "sdpa"


def _record(trace, name, t, meaning):
    """Append one tensor's name, shape and a plain-words meaning to the trace list, if tracing."""
    if trace is not None:
        trace.append((name, tuple(t.shape), str(t.dtype).replace("torch.", ""), meaning))


class Attention(nn.Module):
    """Causal self-attention: each position looks back at earlier positions and mixes their values.

    One linear layer makes queries, keys and values for all heads at once (C -> 3C), the heads
    attend independently, and a second linear layer mixes the heads back together (C -> C).
    Parameters: 3*C*C + C*C = 4*C^2.
    """

    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.attn = cfg.attn
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.proj = nn.Linear(cfg.width, cfg.width, bias=False)

    def forward(self, x, trace=None, p=""):
        """Run attention on a batch.

        Args:
            x: (B, T, C) normalized token vectors.
            trace: optional list collecting shapes.
            p: name prefix for trace entries, e.g. "block0.".

        Returns:
            (B, T, C) the attention output, to be added to the residual stream.
        """
        B, T, C = x.shape
        H, D = self.heads, C // self.heads
        qkv = self.qkv(x)
        _record(trace, p + "attn.qkv", qkv, "queries, keys and values for every token, side by side (3C = 3 x 384)")
        q, k, v = qkv.split(C, dim=2)
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D): give each head its own slice of channels
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        _record(trace, p + "attn.q", q, "queries split into heads: (batch, head, position, channels per head)")
        if self.attn == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
            _record(trace, p + "attn.scores", scores, "how much each position (row) attends to each position (column), per head")
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            y = F.softmax(scores, dim=-1) @ v
        _record(trace, p + "attn.heads_out", y, "each head's weighted mix of values")
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        _record(trace, p + "attn.merged", y, "heads glued back into one C-wide vector per token")
        y = self.proj(y)
        _record(trace, p + "attn.out", y, "attention output after mixing heads")
        return y


class MLP(nn.Module):
    """The feed-forward layer: widen to 4C, apply GELU, narrow back to C. Works on each token alone.

    Parameters: C*4C + 4C*C = 8*C^2, twice the attention layer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.width, 4 * cfg.width, bias=False)
        self.proj = nn.Linear(4 * cfg.width, cfg.width, bias=False)

    def forward(self, x, trace=None, p=""):
        """Args: x (B, T, C). Returns: (B, T, C)."""
        h = F.gelu(self.fc(x))
        _record(trace, p + "mlp.hidden", h, "each token widened to 4C = 1,536 channels, after GELU")
        y = self.proj(h)
        _record(trace, p + "mlp.out", y, "narrowed back to C channels")
        return y


class Block(nn.Module):
    """One transformer block: x + attention(norm(x)), then x + mlp(norm(x)).

    The "x + ..." parts are residual connections: each layer adds a correction to the token
    vectors instead of replacing them, which keeps gradients flowing through deep stacks.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.width)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.width)
        self.mlp = MLP(cfg)

    def forward(self, x, trace=None, p=""):
        """Args: x (B, T, C) residual stream. Returns: (B, T, C) updated residual stream."""
        x = x + self.attn(self.ln1(x), trace, p)
        _record(trace, p + "resid_after_attn", x, "residual stream after adding attention's output")
        x = x + self.mlp(self.ln2(x), trace, p)
        _record(trace, p + "resid_after_mlp", x, "residual stream after adding the MLP's output")
        return x


class GPT(nn.Module):
    """Token ids in, next-token scores (logits) out.

    The embedding table turns ids into vectors; the same table, transposed, turns final vectors
    back into one score per vocabulary entry ("weight tying"). Tying saves V*C parameters and
    means the embedding gets gradient from both ends of the network.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.width)
        self.pos_emb = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln_f = nn.LayerNorm(cfg.width)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        # GPT-2 trick: shrink the layers that write into the residual stream by 1/sqrt(2L), so
        # the stream's size does not grow with depth at initialization.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(m):
        """Start every weight matrix as small random numbers (std 0.02) and LayerNorm at identity."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx, trace=None):
        """Compute next-token scores for every position.

        Args:
            idx: (B, T) int64 token ids, T <= cfg.context.
            trace: optional list. When given, every intermediate tensor is recorded in it.

        Returns:
            (B, T, V) logits. logits[b, t] scores every possible token as the one at t + 1.
        """
        B, T = idx.shape
        _record(trace, "input_ids", idx, "token ids: (sequences in the batch, positions)")
        # Position ids as a full (B, T) grid rather than one (T,) row broadcast over the batch.
        # Same numbers either way, but with the broadcast row torch.compile on Apple GPUs
        # (torch 2.15, MPS) returns a pos_emb gradient B times too large. Found by comparing
        # compiled and eager gradients; see guide Part 15.
        pos = torch.arange(T, device=idx.device).expand(B, T)
        tok = self.tok_emb(idx)
        _record(trace, "tok_emb", tok, "each token id looked up as a C-wide vector")
        x = tok + self.pos_emb(pos)
        _record(trace, "tok_emb + pos_emb", x, "token vectors plus a learned vector for each position")
        for i, block in enumerate(self.blocks):
            x = block(x, trace if i == 0 else None, f"block{i}.")
        if trace is not None:
            trace.append((f"block1 .. block{len(self.blocks) - 1}", tuple(x.shape), str(x.dtype).replace("torch.", ""),
                          "same shapes as block0, repeated"))
        x = self.ln_f(x)
        _record(trace, "ln_f", x, "final normalized token vectors")
        logits = self.lm_head(x)
        _record(trace, "logits", logits, "a score for every vocabulary entry at every position")
        return logits

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=50):
        """Extend a prompt by sampling one token at a time.

        Args:
            idx: (1, T) prompt token ids.
            n_new: how many tokens to add.
            temperature: below 1 makes the model pickier, above 1 more random.
            top_k: only sample among the k highest-scoring tokens.

        Returns:
            (1, T + n_new) token ids.
        """
        for _ in range(n_new):
            logits = self(idx[:, -self.cfg.context :])[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits.float(), dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def token_loss(logits, targets, reduction="mean"):
    """Cross-entropy between the model's scores and the tokens that actually came next.

    For each position: loss = -log(softmax(logits)[correct token]). Targets equal to -100 are
    padding and are skipped, which matters when sequences in a batch have different lengths.

    Args:
        logits: (B, T, V) scores.
        targets: (B, T) int64 ids of the true next tokens, -100 where there is no token.
        reduction: "mean" averages over real tokens, "sum" adds them up. Gradient accumulation
            needs "sum" so it can divide by the token count of the whole step (see accum.py).

    Returns:
        A scalar tensor. Computed in at least fp32, even under autocast, because softmax over
        4,096 entries in bf16 loses digits. fp64 logits stay fp64 (gradcheck.py needs that).
    """
    if logits.dtype != torch.float64:
        logits = logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100, reduction=reduction)


def count_params(model):
    """Count each parameter tensor once. The tied embedding/output matrix is one tensor, so it counts once.

    Returns:
        Total number of trainable numbers N, the N in "6N FLOPs per token".
    """
    return sum(p.numel() for p in model.parameters())
