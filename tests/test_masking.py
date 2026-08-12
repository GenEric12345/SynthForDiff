"""Mask-rate correctness and forward-process conventions (must match MDLM)."""
import pytest
import torch

from src.masking import (GPT2_VOCAB_SIZE, MASK_INDEX, VOCAB_SIZE_WITH_MASK,
                         apply_random_mask, sample_t)


def test_mask_token_convention():
    # MDLM appends [MASK] after the GPT-2 vocab (third_party/mdlm/diffusion.py:85-90)
    assert GPT2_VOCAB_SIZE == 50257
    assert MASK_INDEX == 50257
    assert VOCAB_SIZE_WITH_MASK == 50258


@pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
def test_empirical_mask_rate(t):
    """10k sequences: empirical mask fraction within 0.5% of t."""
    torch.manual_seed(0)
    x = torch.randint(0, GPT2_VOCAB_SIZE, (10_000, 1024))
    xt = apply_random_mask(x, t)
    frac = (xt == MASK_INDEX).float().mean().item()
    assert abs(frac - t) < 0.005, (t, frac)


def test_unmasked_tokens_unchanged():
    torch.manual_seed(0)
    x = torch.randint(0, GPT2_VOCAB_SIZE, (64, 256))
    xt = apply_random_mask(x, 0.5)
    keep = xt != MASK_INDEX
    assert torch.equal(xt[keep], x[keep])
    assert torch.equal(apply_random_mask(x, 0.0), x)
    assert (apply_random_mask(x, 1.0) == MASK_INDEX).all()


def test_fresh_randomness_every_call():
    """Masks must be resampled on every access, never cached."""
    torch.manual_seed(0)
    x = torch.randint(0, GPT2_VOCAB_SIZE, (8, 512))
    a = apply_random_mask(x, 0.5)
    b = apply_random_mask(x, 0.5)
    assert not torch.equal(a, b)


def test_per_example_t_broadcast():
    torch.manual_seed(0)
    x = torch.randint(0, GPT2_VOCAB_SIZE, (2, 20_000))
    t = torch.tensor([[0.1], [0.9]])
    xt = apply_random_mask(x, t)
    fracs = (xt == MASK_INDEX).float().mean(dim=1)
    assert abs(fracs[0].item() - 0.1) < 0.02
    assert abs(fracs[1].item() - 0.9) < 0.02


def test_matches_mdlm_q_xt_semantics():
    """Same RNG stream => identical output to MDLM's q_xt
    (third_party/mdlm/diffusion.py:575-586)."""
    x = torch.randint(0, GPT2_VOCAB_SIZE, (16, 128))
    move_chance = torch.tensor(0.37)

    g1 = torch.Generator()
    g1.manual_seed(123)
    ours = apply_random_mask(x, move_chance, generator=g1)

    g2 = torch.Generator()
    g2.manual_seed(123)
    move_indices = torch.rand(*x.shape, generator=g2) < move_chance  # mdlm q_xt
    reference = torch.where(move_indices, torch.tensor(MASK_INDEX), x)
    assert torch.equal(ours, reference)


def test_sample_t_range():
    torch.manual_seed(0)
    t = sample_t(100_000, 0.05, 0.95)
    assert t.min() >= 0.05 and t.max() <= 0.95
    assert abs(t.mean().item() - 0.5) < 0.01
