"""Dataset construction and loading.

Layout on disk (paths from configs/default.yaml):
  data/dataset.parquet                     one row per kept document:
      doc_id, split, prompt_token_ids, prompt_text, real_continuation_token_ids
  data/synthetic/<generator>/shard-*.parquet   one row per generated doc:
      doc_id, generator_name, seed, attempt, temperature, top_p, gen_max_tokens,
      synthetic_continuation_token_ids

All token ids are GPT-2 BPE. The prompt is NEVER part of any classifier input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.config import REPO_ROOT
from src.utils import derive_seed, doc_hash, get_logger

log = get_logger("data")


def get_gpt2_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tok.model_max_length = int(1e9)  # tokenization only; silence length warning
    return tok


def iter_openwebtext(hf_name: str = "Skylion007/openwebtext") -> Iterator[str]:
    """Stream raw document texts from HF."""
    from datasets import load_dataset
    ds = load_dataset(hf_name, split="train", streaming=True)
    for ex in ds:
        yield ex["text"]


def build_records(
    text_iter: Iterable[str],
    tokenizer,
    n_docs: int,
    prompt_len: int,
    cont_len: int,
    dedup_prefix: int = 256,
    tokenize_batch: int = 512,
) -> list[dict]:
    """Tokenize, filter (>= prompt_len + cont_len tokens), dedup, slice.

    text_iter is injectable so tests can feed a synthetic corpus.
    """
    min_len = prompt_len + cont_len
    records: list[dict] = []
    seen: set[str] = set()
    n_scanned = 0
    batch: list[str] = []

    def flush(texts: list[str]) -> bool:
        nonlocal n_scanned
        if not texts:
            return False
        enc = tokenizer(texts)["input_ids"]
        n_scanned += len(texts)
        for ids in enc:
            if len(ids) < min_len:
                continue
            h = doc_hash(ids, dedup_prefix)
            if h in seen:
                continue
            seen.add(h)
            prompt_ids = ids[:prompt_len]
            cont_ids = ids[prompt_len:min_len]
            records.append({
                "doc_id": h,
                "prompt_token_ids": prompt_ids,
                "prompt_text": tokenizer.decode(prompt_ids),
                "real_continuation_token_ids": cont_ids,
            })
            if len(records) >= n_docs:
                return True
        return False

    for text in text_iter:
        batch.append(text)
        if len(batch) >= tokenize_batch:
            done = flush(batch)
            batch = []
            if done:
                break
            if n_scanned % 51200 == 0:
                log.info("scanned %d docs, kept %d/%d", n_scanned, len(records), n_docs)
    else:
        flush(batch)

    if len(records) < n_docs:
        raise RuntimeError(
            f"source exhausted: only {len(records)}/{n_docs} documents "
            f"with >= {min_len} GPT-2 tokens")
    log.info("kept %d documents (scanned %d)", len(records), n_scanned)
    return records


def retokenize_truncate(texts: list[str], tokenizer, cont_len: int) -> list[list[int] | None]:
    """Re-tokenize generator output with GPT-2 and truncate to exactly cont_len
    tokens. Returns None for generations that come up short (caller retries).

    This is the ONLY preprocessing applied to synthetic continuations, so both
    classes end up as exactly-cont_len GPT-2 token sequences.
    """
    enc = tokenizer(texts)["input_ids"]
    out: list[list[int] | None] = []
    for ids in enc:
        out.append(ids[:cont_len] if len(ids) >= cont_len else None)
    return out


def assign_splits(n: int, n_train: int, n_val: int, n_test: int, seed: int) -> list[str]:
    """Random BY-DOCUMENT split. Each document lands in exactly one split, so a
    prompt/continuation pair can never leak across splits."""
    assert n >= n_train + n_val + n_test, (n, n_train, n_val, n_test)
    rng = np.random.default_rng(derive_seed(seed, "split"))
    order = rng.permutation(n)
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:n_train + n_val + n_test]] = "test"
    labels[order[n_train + n_val + n_test:]] = "unused"
    return labels.tolist()


def save_dataset(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "doc_id": pa.array([r["doc_id"] for r in records], pa.string()),
        "split": pa.array([r["split"] for r in records], pa.string()),
        "prompt_token_ids": pa.array([r["prompt_token_ids"] for r in records],
                                     pa.list_(pa.int32())),
        "prompt_text": pa.array([r["prompt_text"] for r in records], pa.string()),
        "real_continuation_token_ids": pa.array(
            [r["real_continuation_token_ids"] for r in records], pa.list_(pa.int32())),
    })
    pq.write_table(table, path)
    log.info("wrote %s (%d rows)", path, len(records))


def load_dataset_df(cfg) -> pd.DataFrame:
    path = REPO_ROOT / cfg.paths.dataset
    df = pq.read_table(path).to_pandas()
    return df


def synthetic_dir(cfg) -> Path:
    gen = cfg.generation.model.replace("/", "__")
    return REPO_ROOT / cfg.paths.synthetic_dir / gen


def load_synthetic_df(cfg) -> pd.DataFrame:
    d = synthetic_dir(cfg)
    shards = sorted(d.glob("shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no synthetic shards in {d}; run generate_synthetic first")
    df = pd.concat([pq.read_table(s).to_pandas() for s in shards], ignore_index=True)
    df = df.drop_duplicates(subset="doc_id", keep="last")
    return df


def append_synthetic_shard(cfg, rows: list[dict]) -> Path:
    d = synthetic_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    idx = len(list(d.glob("shard-*.parquet")))
    path = d / f"shard-{idx:05d}.parquet"
    table = pa.table({
        "doc_id": pa.array([r["doc_id"] for r in rows], pa.string()),
        "generator_name": pa.array([r["generator_name"] for r in rows], pa.string()),
        "seed": pa.array([r["seed"] for r in rows], pa.int64()),
        "attempt": pa.array([r["attempt"] for r in rows], pa.int32()),
        "sampling_params": pa.array([json.dumps(r["sampling_params"]) for r in rows],
                                    pa.string()),
        "synthetic_continuation_token_ids": pa.array(
            [r["synthetic_continuation_token_ids"] for r in rows], pa.list_(pa.int32())),
    })
    pq.write_table(table, path)
    return path


def load_pairs(cfg, splits: list[str] | None = None) -> pd.DataFrame:
    """Join real and synthetic continuations by doc_id.

    Returns df with doc_id, split, real_ids, synth_ids. Verifies BOTH classes
    have exactly cfg.data.cont_len tokens for every document — length must not
    be a usable feature.
    """
    real = load_dataset_df(cfg)
    if splits is not None:
        real = real[real["split"].isin(splits)]
    synth = load_synthetic_df(cfg)[["doc_id", "synthetic_continuation_token_ids"]]
    df = real.merge(synth, on="doc_id", how="inner", validate="one_to_one")
    missing = len(real) - len(df)
    if missing:
        raise RuntimeError(
            f"{missing} documents in splits {splits} lack synthetic continuations; "
            "finish generate_synthetic first (it is resumable)")
    L = int(cfg.data.cont_len)
    real_lens = df["real_continuation_token_ids"].map(len)
    synth_lens = df["synthetic_continuation_token_ids"].map(len)
    assert (real_lens == L).all(), "real continuation length != cont_len"
    assert (synth_lens == L).all(), "synthetic continuation length != cont_len"
    return df.rename(columns={
        "real_continuation_token_ids": "real_ids",
        "synthetic_continuation_token_ids": "synth_ids",
    })[["doc_id", "split", "real_ids", "synth_ids"]]


class PairedContinuations(torch.utils.data.Dataset):
    """Yields the (real, synthetic) continuation pair for one document.

    The training loop unrolls each item into two examples (labels 1=real,
    0=synthetic), which makes every batch exactly 50/50 class-balanced.
    Token ids only — masking happens later, freshly, in the train/eval step.
    """

    def __init__(self, pairs_df: pd.DataFrame):
        self.doc_ids = pairs_df["doc_id"].tolist()
        self.real = [np.array(x, dtype=np.int64) for x in pairs_df["real_ids"]]
        self.synth = [np.array(x, dtype=np.int64) for x in pairs_df["synth_ids"]]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, i: int):
        return {
            "doc_id": self.doc_ids[i],
            "real_ids": torch.from_numpy(self.real[i]),
            "synth_ids": torch.from_numpy(self.synth[i]),
        }


def collate_pairs(items: list[dict]) -> dict:
    return {
        "doc_id": [it["doc_id"] for it in items],
        "real_ids": torch.stack([it["real_ids"] for it in items]),
        "synth_ids": torch.stack([it["synth_ids"] for it in items]),
    }
