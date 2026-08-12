"""Both classes must have identical length distributions; load_pairs enforces it."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.config import Config
from src.data import PairedContinuations, collate_pairs, load_pairs


def _write_fixture(tmp_path, monkeypatch, cont_len=32, synth_len=32, n=6):
    rng = np.random.default_rng(0)
    doc_ids = [f"d{i:03d}" for i in range(n)]
    splits = ["train"] * (n - 4) + ["val"] * 2 + ["test"] * 2
    real = [rng.integers(0, 50257, cont_len).tolist() for _ in range(n)]
    synth = [rng.integers(0, 50257, synth_len).tolist() for _ in range(n)]

    ds = tmp_path / "dataset.parquet"
    pq.write_table(pa.table({
        "doc_id": doc_ids, "split": splits,
        "prompt_token_ids": [rng.integers(0, 50257, 8).tolist() for _ in range(n)],
        "prompt_text": ["p"] * n,
        "real_continuation_token_ids": real,
    }), ds)
    gen_dir = tmp_path / "synthetic" / "gen"
    gen_dir.mkdir(parents=True)
    pq.write_table(pa.table({
        "doc_id": doc_ids,
        "synthetic_continuation_token_ids": synth,
    }), gen_dir / "shard-00000.parquet")

    import src.config as config_mod
    import src.data as data_mod
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data_mod, "REPO_ROOT", tmp_path)
    return Config({
        "paths": {"dataset": "dataset.parquet", "synthetic_dir": "synthetic"},
        "generation": {"model": "gen"},
        "data": {"cont_len": cont_len},
    })


def test_identical_length_distributions(tmp_path, monkeypatch):
    cfg = _write_fixture(tmp_path, monkeypatch)
    pairs = load_pairs(cfg)
    real_lens = pairs["real_ids"].map(len).to_numpy()
    synth_lens = pairs["synth_ids"].map(len).to_numpy()
    assert (real_lens == synth_lens).all()
    assert set(real_lens) == {32} and set(synth_lens) == {32}


def test_length_mismatch_rejected(tmp_path, monkeypatch):
    cfg = _write_fixture(tmp_path, monkeypatch, cont_len=32, synth_len=30)
    with pytest.raises(AssertionError, match="synthetic continuation length"):
        load_pairs(cfg)


def test_missing_synthetic_rejected(tmp_path, monkeypatch):
    cfg = _write_fixture(tmp_path, monkeypatch)
    shard = tmp_path / "synthetic" / "gen" / "shard-00000.parquet"
    df = pq.read_table(shard).to_pandas().iloc[:-1]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), shard)
    with pytest.raises(RuntimeError, match="lack synthetic"):
        load_pairs(cfg)


def test_batches_are_balanced_and_equal_shape(tmp_path, monkeypatch):
    cfg = _write_fixture(tmp_path, monkeypatch)
    ds = PairedContinuations(load_pairs(cfg, ["train"]))
    batch = collate_pairs([ds[i] for i in range(2)])
    assert batch["real_ids"].shape == batch["synth_ids"].shape
    x = torch.cat([batch["real_ids"], batch["synth_ids"]])
    y = torch.cat([torch.ones(2), torch.zeros(2)])
    assert x.shape[0] == 4 and float(y.mean()) == 0.5  # 50/50 by construction
