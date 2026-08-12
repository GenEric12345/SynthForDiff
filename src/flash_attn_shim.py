"""Pure-PyTorch stand-in for the two flash_attn entry points used by the
kuleshov-group/mdlm-owt remote code (modeling_mdlm.py), so the UNMODIFIED
pretrained checkpoint loads and runs on machines without flash-attn/CUDA.

install() registers fake `flash_attn` modules in sys.modules ONLY if the real
flash_attn is not importable. On a GPU box with flash-attn installed (the
intended production setup, matching third_party/mdlm/requirements.yaml) this
module is a no-op and the real fused kernels are used.

Functions replicated (semantics checked against flash-attn's documented
behavior and mdlm's call sites in modeling_mdlm.py):

* flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
    qkv: (B, S, 3, H, D); cos/sin: (S, D/2); non-interleaved (GPT-NeoX style)
    rotary applied to Q and K over the full head dim, V untouched.
* flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
      qkv, cu_seqlens, max_seqlen, dropout_p, causal=...)
    qkv: (total_tokens, 3, H, D) -> out (total_tokens, H, D), softmax scale
    D**-0.5. mdlm always calls it with equal-length sequences.

Correctness of both is unit-tested in tests/test_flash_attn_shim.py against
naive reference implementations.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


def _apply_rotary_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Non-interleaved rotary: x (..., S, D)-like with cos/sin (S, D/2)."""
    d_half = cos.shape[-1]
    x1, x2 = x[..., :d_half], x[..., d_half: 2 * d_half]
    # broadcast cos/sin (S, D/2) over (B, S, H, D/2): insert head dim
    c = cos[None, :, None, :].to(x.dtype)
    s = sin[None, :, None, :].to(x.dtype)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = torch.cat([o1, o2, x[..., 2 * d_half:]], dim=-1)
    return out


def apply_rotary_emb_qkv_(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          *args, **kwargs) -> torch.Tensor:
    """qkv: (B, S, 3, H, D). Applies rotary to q and k; v is untouched."""
    q = _apply_rotary_half(qkv[:, :, 0], cos, sin)
    k = _apply_rotary_half(qkv[:, :, 1], cos, sin)
    v = qkv[:, :, 2]
    return torch.stack([q, k, v], dim=2)


def flash_attn_varlen_qkvpacked_func(qkv: torch.Tensor, cu_seqlens: torch.Tensor,
                                     max_seqlen: int, dropout_p: float = 0.0,
                                     causal: bool = False, **kwargs) -> torch.Tensor:
    """qkv: (total, 3, H, D) with cu_seqlens boundaries -> (total, H, D)."""
    seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1])
    if not bool((seq_lens == max_seqlen).all()):
        # generic (ragged) path — not used by mdlm, kept for safety
        outs = []
        for i in range(len(seq_lens)):
            chunk = qkv[cu_seqlens[i]: cu_seqlens[i + 1]]
            outs.append(_attend(chunk.unsqueeze(0), dropout_p, causal).squeeze(0))
        return torch.cat(outs, dim=0)
    b = len(seq_lens)
    s = int(max_seqlen)
    out = _attend(qkv.view(b, s, 3, qkv.shape[-2], qkv.shape[-1]), dropout_p, causal)
    return out.reshape(b * s, qkv.shape[-2], qkv.shape[-1])


def _attend(qkv: torch.Tensor, dropout_p: float, causal: bool) -> torch.Tensor:
    """qkv: (B, S, 3, H, D) -> (B, S, H, D) via scaled_dot_product_attention."""
    q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))  # (B, H, S, D)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)
    return o.transpose(1, 2)


def install() -> bool:
    """Register the shim if real flash_attn is unavailable. Returns True if shimmed."""
    try:
        import flash_attn  # noqa: F401
        return False
    except ImportError:
        pass
    import importlib.machinery

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        # a real ModuleSpec so importlib.util.find_spec(name) doesn't raise
        # (transformers probes it; without installed metadata it then treats
        # flash-attn as unavailable, which is what we want)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return m

    fa = _mod("flash_attn")
    fa.__version__ = "0.0.0+shim"
    layers = _mod("flash_attn.layers")
    rotary = _mod("flash_attn.layers.rotary")
    iface = _mod("flash_attn.flash_attn_interface")
    rotary.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_
    iface.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    layers.rotary = rotary
    fa.layers = layers
    fa.flash_attn_interface = iface
    sys.modules["flash_attn"] = fa
    sys.modules["flash_attn.layers"] = layers
    sys.modules["flash_attn.layers.rotary"] = rotary
    sys.modules["flash_attn.flash_attn_interface"] = iface
    return True
