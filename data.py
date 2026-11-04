"""Download FineWeb-Edu text, train a small BPE tokenizer on it, and write packed token files.

    uv run data.py --out data/fwe --train-tokens 25M --eval-tokens 500K
    uv run data.py --out data/smoke --train-tokens 2M --eval-tokens 100K     # quick test

Writes:
    tokenizer.json   the trained byte-level BPE (4,096 tokens by default)
    train.bin        uint16 token ids, documents joined by the end-of-text token
    eval.bin         same format, built from the first documents of the stream
    meta.json        vocab size, token counts and the end-of-text id

The eval documents come first in the stream and are never used to train the tokenizer or the
model, so the held-out loss is measured on text the run has never seen.
"""

import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# A 10B-token sample of FineWeb-Edu. We read the parquet files row group by row group because
# the `datasets` streaming reader stalls on this dataset (found while building a14-MOE).
PARQUET_GLOB = "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet"
EOT = "<|endoftext|>"


def parse_count(s):
    """Turn a human count like "25M", "500K" or "1.5B" into an int.

    Args:
        s: a number with an optional K, M or B suffix (case does not matter).

    Returns:
        The count as an int, e.g. parse_count("25M") == 25_000_000.
    """
    s = s.upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1], 1)
    return int(float(s.rstrip("KMB")) * mult)


def documents():
    """Yield FineWeb-Edu documents as plain strings, in a fixed order.

    The order is the sorted file list, then row groups in file order, so two runs of this
    script see the same documents and build the same token files.

    Yields:
        One document's text per iteration.
    """
    fs = HfFileSystem()
    for path in sorted(fs.glob(PARQUET_GLOB)):
        pf = pq.ParquetFile(fs.open(path))
        for rg in range(pf.metadata.num_row_groups):
            yield from pf.read_row_group(rg, columns=["text"]).column(0).to_pylist()


def collect(n_chars):
    """Download documents until we hold at least `n_chars` characters of text.

    Args:
        n_chars: how many characters to collect. We over-collect on purpose (about 6
            characters per token we want), because we only learn the real characters per
            token after training the tokenizer.

    Returns:
        A list of document strings.
    """
    docs, have, t0 = [], 0, time.time()
    for text in documents():
        docs.append(text)
        have += len(text)
        if len(docs) % 5000 == 0:
            print(f"  downloaded {len(docs):,} docs, {have / 1e6:.0f}M chars, {time.time() - t0:.0f}s", flush=True)
        if have >= n_chars:
            break
    print(f"downloaded {len(docs):,} docs, {have / 1e6:.0f}M chars in {time.time() - t0:.0f}s")
    return docs


def train_tokenizer(texts, vocab_size):
    """Train a byte-level BPE tokenizer, the same kind GPT-2 uses but with a smaller vocabulary.

    Byte-level means every text can be encoded: the tokenizer starts from the 256 possible
    bytes and then learns `vocab_size - 257` merges, each one gluing the most frequent
    adjacent pair into a new token ("t" + "h" -> "th", then "th" + "e" -> "the", ...).

    Args:
        texts: list of training strings.
        vocab_size: final number of tokens, including the 256 bytes and the end-of-text token.

    Returns:
        A trained `tokenizers.Tokenizer`.
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def tokenize_into(tok, docs, want, eot_id, batch=512):
    """Tokenize documents in order and pack them into one long array of token ids.

    Each document is followed by the end-of-text token, so the model sees where one document
    stops and the next starts. The last document is cut so the array has exactly `want` ids.

    Args:
        tok: the trained tokenizer.
        docs: iterator of document strings. It is consumed only as far as needed.
        want: number of tokens to produce.
        eot_id: id of the end-of-text token.
        batch: documents per call to the (multi-threaded) batch encoder.

    Returns:
        (ids, n_docs): a uint16 numpy array of length `want` (or shorter if `docs` ran out),
        and how many documents were used.
    """
    out, have, n_docs = [], 0, 0
    chunk = []
    for text in docs:
        chunk.append(text)
        if len(chunk) < batch:
            continue
        for enc in tok.encode_batch(chunk):
            ids = enc.ids + [eot_id]
            out.append(np.asarray(ids[: want - have], dtype=np.uint16))
            have += len(out[-1])
            n_docs += 1
            if have >= want:
                return np.concatenate(out), n_docs
        chunk = []
    for enc in tok.encode_batch(chunk):
        ids = enc.ids + [eot_id]
        out.append(np.asarray(ids[: want - have], dtype=np.uint16))
        have += len(out[-1])
        n_docs += 1
        if have >= want:
            break
    return np.concatenate(out), n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tokens", default="25M")
    ap.add_argument("--eval-tokens", default="500K")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--tokenizer-chars", default="40M", help="characters of training text used to learn the BPE merges")
    args = ap.parse_args()
    want = {"eval": parse_count(args.eval_tokens), "train": parse_count(args.train_tokens)}
    os.makedirs(args.out, exist_ok=True)

    docs = collect(6 * (want["eval"] + want["train"]))

    # Hold out roughly the first eval-sized slice of documents so the tokenizer never sees it.
    eval_chars, n_eval_docs = 0, 0
    while eval_chars < 6 * want["eval"]:
        eval_chars += len(docs[n_eval_docs])
        n_eval_docs += 1
    tok_texts, n = [], 0
    for text in docs[n_eval_docs:]:
        tok_texts.append(text)
        n += len(text)
        if n >= parse_count(args.tokenizer_chars):
            break
    t0 = time.time()
    tok = train_tokenizer(tok_texts, args.vocab_size)
    tok.save(f"{args.out}/tokenizer.json")
    eot_id = tok.token_to_id(EOT)
    print(f"trained a {tok.get_vocab_size():,}-token BPE on {n / 1e6:.0f}M chars in {time.time() - t0:.0f}s")

    # Eval uses the first n_eval documents, which sit inside the slice the tokenizer never saw
    # (the 6 chars/token estimate over-reserves). Train starts right after them.
    eval_ids, n_eval = tokenize_into(tok, iter(docs), want["eval"], eot_id)
    assert n_eval <= n_eval_docs, "eval documents leaked into tokenizer training"
    train_ids, n_train = tokenize_into(tok, iter(docs[n_eval:]), want["train"], eot_id)
    if len(train_ids) < want["train"]:
        raise SystemExit(f"ran out of text at {len(train_ids):,} train tokens, raise the 6 chars/token margin")
    eval_ids.tofile(f"{args.out}/eval.bin")
    train_ids.tofile(f"{args.out}/train.bin")

    chars = sum(len(d) for d in docs[n_eval : n_eval + n_train])
    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu sample-10BT",
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot_id,
        "dtype": "uint16",
        "tokens": {"eval": len(eval_ids), "train": len(train_ids)},
        "docs": {"eval": n_eval, "train": n_train},
        "chars_per_token": round(chars / len(train_ids), 3),
    }
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    sample = "The mitochondria is the powerhouse of the cell."
    enc = tok.encode(sample)
    print(f"example: {sample!r}\n  -> {enc.tokens}\n  -> {enc.ids}")


if __name__ == "__main__":
    main()
