"""GPT-2 retokenization/truncation produces exactly 960 tokens (real tokenizer)."""
import pytest

from src.data import retokenize_truncate

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def gpt2():
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    except Exception as e:  # offline CI
        pytest.skip(f"gpt2 tokenizer unavailable: {e}")


def test_exactly_960_tokens(gpt2):
    cont_len = 960
    long_text = ("The quick brown fox jumps over the lazy dog. " * 400)
    out = retokenize_truncate([long_text], gpt2, cont_len)[0]
    assert out is not None
    assert len(out) == cont_len
    # truncation is a prefix of the full tokenization
    full = gpt2(long_text)["input_ids"]
    assert out == full[:cont_len]


def test_short_generation_flagged_for_retry(gpt2):
    out = retokenize_truncate(["too short"], gpt2, 960)[0]
    assert out is None


def test_all_ids_below_mask_index(gpt2):
    from src.masking import MASK_INDEX
    out = retokenize_truncate(["hello world " * 1000], gpt2, 960)[0]
    assert max(out) < MASK_INDEX
