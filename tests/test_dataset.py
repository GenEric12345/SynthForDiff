"""Dataset construction: filtering, dedup, split hygiene, length invariants.

Uses a dummy word-level tokenizer so unit tests never touch the network; the
real GPT-2 path is exercised by the smoke test.
"""
import numpy as np
import pytest

from src.data import assign_splits, build_records, retokenize_truncate


class DummyTokenizer:
    """Whitespace 'BPE': token id = hash of word."""

    def __call__(self, texts):
        return {"input_ids": [[hash(w) % 50000 for w in t.split()] for t in texts]}

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def make_corpus():
    rng = np.random.default_rng(0)
    texts = []
    for i in range(80):
        n = int(rng.integers(5, 60))
        texts.append(" ".join(f"tok{i}_{j}" for j in range(n)))
    texts.append(texts[0])                    # exact duplicate
    texts.append(texts[1] + " extra tail")    # same first tokens => dedup hash hit
    return texts


def test_build_records_filter_dedup_slice():
    tok = DummyTokenizer()
    prompt_len, cont_len = 8, 24
    corpus = make_corpus()
    eligible = sum(len(t.split()) >= prompt_len + cont_len for t in corpus[:80])
    records = build_records(corpus, tok, n_docs=eligible, prompt_len=prompt_len,
                            cont_len=cont_len, dedup_prefix=16, tokenize_batch=7)
    assert len(records) == eligible
    ids = [r["doc_id"] for r in records]
    assert len(set(ids)) == len(ids)  # dedup
    for r in records:
        assert len(r["prompt_token_ids"]) == prompt_len
        assert len(r["real_continuation_token_ids"]) == cont_len


def test_build_records_raises_when_exhausted():
    with pytest.raises(RuntimeError, match="exhausted"):
        build_records(["short text"] * 5, DummyTokenizer(), n_docs=3,
                      prompt_len=8, cont_len=24)


def test_no_docid_leaks_across_splits():
    n = 500
    labels = assign_splits(n, 300, 100, 50, seed=7)
    assert len(labels) == n
    doc_ids = [f"doc{i}" for i in range(n)]
    by_split = {}
    for d, s in zip(doc_ids, labels):
        by_split.setdefault(s, set()).add(d)
    assert len(by_split["train"]) == 300
    assert len(by_split["val"]) == 100
    assert len(by_split["test"]) == 50
    assert len(by_split.get("unused", set())) == 50
    for a in by_split:
        for b in by_split:
            if a != b:
                assert not (by_split[a] & by_split[b]), (a, b)


def test_splits_deterministic_under_seed():
    assert assign_splits(200, 100, 50, 50, seed=7) == assign_splits(200, 100, 50, 50, seed=7)
    assert assign_splits(200, 100, 50, 50, seed=7) != assign_splits(200, 100, 50, 50, seed=8)


def test_retokenize_truncate_exact_length():
    tok = DummyTokenizer()
    cont_len = 24
    texts = [" ".join(f"w{i}" for i in range(40)),   # long -> truncated
             " ".join(f"w{i}" for i in range(24)),   # exact
             " ".join(f"w{i}" for i in range(10))]   # short -> None (retry)
    out = retokenize_truncate(texts, tok, cont_len)
    assert len(out[0]) == cont_len
    assert len(out[1]) == cont_len
    assert out[2] is None
