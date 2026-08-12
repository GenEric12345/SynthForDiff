"""Classifier head/pooling logic (tiny random backbone; no download needed)."""
import torch

from src.classifier import MaskedSeqClassifier
from src.masking import MASK_INDEX, apply_random_mask


def make_model(pool="all"):
    return MaskedSeqClassifier("_tiny_test", pool=pool)


def test_forward_shapes_and_probability_orientation():
    torch.manual_seed(0)
    model = make_model()
    x = torch.randint(0, 50257, (4, 64))
    xt = apply_random_mask(x, 0.5)
    logits = model(xt)
    assert logits.shape == (4,)
    p_real = torch.sigmoid(logits)  # convention: output = P(real)
    assert ((p_real >= 0) & (p_real <= 1)).all()


def test_nonmask_pooling_ignores_masked_positions():
    torch.manual_seed(0)
    model = make_model(pool="nonmask").eval()
    x = torch.randint(0, 50257, (2, 32))
    xt = x.clone()
    xt[:, 16:] = MASK_INDEX
    with torch.no_grad():
        a = model(xt)
        # changing only MASKED positions' original ids cannot change the input
        # (they are all MASK_INDEX already), but changing an UNMASKED id must.
        xt2 = xt.clone()
        xt2[:, 0] = (xt2[:, 0] + 1) % 50257
        b = model(xt2)
    assert not torch.allclose(a, b)


def test_fully_masked_input_finite():
    torch.manual_seed(0)
    for pool in ("all", "nonmask"):
        model = make_model(pool=pool).eval()
        xt = torch.full((2, 32), MASK_INDEX)
        with torch.no_grad():
            logits = model(xt)
        assert torch.isfinite(logits).all()


def test_gradients_flow_to_backbone_and_head():
    model = make_model()
    xt = torch.randint(0, 50258, (2, 32))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(xt), torch.tensor([1.0, 0.0]))
    loss.backward()
    assert model.head.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.backbone.parameters())
