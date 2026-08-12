"""The pure-torch flash_attn shim must match reference math exactly
(it stands in for the fused kernels the mdlm-owt remote code calls)."""
import math

import torch

from src.flash_attn_shim import apply_rotary_emb_qkv_, flash_attn_varlen_qkvpacked_func


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def test_rotary_matches_rotate_half_identity():
    torch.manual_seed(0)
    b, s, h, d = 2, 16, 4, 32
    qkv = torch.randn(b, s, 3, h, d)
    ang = torch.rand(s, d // 2) * 6.28
    cos, sin = ang.cos(), ang.sin()
    out = apply_rotary_emb_qkv_(qkv.clone(), cos, sin)

    cos_full = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    sin_full = torch.cat([sin, sin], dim=-1)[None, :, None, :]
    for i in (0, 1):  # q and k rotated
        expected = qkv[:, :, i] * cos_full + _rotate_half(qkv[:, :, i]) * sin_full
        assert torch.allclose(out[:, :, i], expected, atol=1e-5)
    assert torch.equal(out[:, :, 2], qkv[:, :, 2])  # v untouched


def test_attention_matches_naive_softmax():
    torch.manual_seed(0)
    b, s, h, d = 2, 8, 3, 16
    qkv = torch.randn(b * s, 3, h, d)
    cu = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu, s, 0.0, causal=False)
    assert out.shape == (b * s, h, d)

    qkv_b = qkv.view(b, s, 3, h, d)
    for bi in range(b):
        for hi in range(h):
            q = qkv_b[bi, :, 0, hi]
            k = qkv_b[bi, :, 1, hi]
            v = qkv_b[bi, :, 2, hi]
            attn = torch.softmax(q @ k.T / math.sqrt(d), dim=-1)
            expected = attn @ v
            got = out.view(b, s, h, d)[bi, :, hi]
            assert torch.allclose(got, expected, atol=1e-5)
