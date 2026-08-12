"""Time-aware real-vs-synthetic classifier on masked sequences.

Backbone: kuleshov-group/mdlm-owt (a 12-block DiT trained with the MDLM
objective on masked OpenWebText at all mask ratios — i.e. exactly our input
distribution), loaded via AutoModelForMaskedLM(..., trust_remote_code=True).

Head: mean-pool the final hidden states (output of the last DiT block; the
checkpoint's `hidden_states` list is [embedding, block1..block12]) over ALL
positions, then Linear(hidden -> 1).  `pool='nonmask'` is the ablation that
pools over surviving (non-[MASK]) positions only.

ORIENTATION CONVENTION (used everywhere in this repo):
    label 1 = REAL, label 0 = SYNTHETIC, classifier output = P(real) =
    sigmoid(logit). AUC is computed with real as the positive class.

Note: mdlm-owt was trained with time_conditioning=False (the checkpoint zeroes
sigma internally; see modeling_mdlm.py DITBackbone.forward), so we pass
timesteps=zeros. The mask ratio is implicitly visible to the model through the
fraction of [MASK] tokens in the input.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.masking import MASK_INDEX, VOCAB_SIZE_WITH_MASK

DEFAULT_BACKBONE = "kuleshov-group/mdlm-owt"


class _TinyTestBackbone(nn.Module):
    """Small randomly-initialized stand-in exposing the same interface as the
    mdlm-owt HF model. FOR UNIT TESTS ONLY (cfg model.backbone == '_tiny_test'):
    lets the pytest suite and CPU-only integration checks run without
    downloading the checkpoint. Never a silent substitute: scripts default to
    mdlm-owt and log the backbone in the resolved config.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.config = type("C", (), {"hidden_dim": hidden_dim})()
        self.embed = nn.Embedding(VOCAB_SIZE_WITH_MASK, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=4 * hidden_dim,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, input_ids, timesteps=None, output_hidden_states=False,
                return_dict=True):
        h = self.encoder(self.embed(input_ids))
        out = type("O", (), {})()
        out.hidden_states = [h]
        out.logits = None
        return out


def load_backbone(name: str, dtype: torch.dtype = torch.float32) -> nn.Module:
    if name == "_tiny_test":
        return _TinyTestBackbone()
    # Make the unmodified checkpoint importable without flash-attn (CPU/tests);
    # no-op when real flash-attn is installed (GPU production path).
    from src import flash_attn_shim
    flash_attn_shim.install()
    from transformers import AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=dtype)
    assert model.config.vocab_size == VOCAB_SIZE_WITH_MASK, (
        f"backbone vocab {model.config.vocab_size} != {VOCAB_SIZE_WITH_MASK}; "
        "mask-token convention would be broken")
    return model


class MaskedSeqClassifier(nn.Module):
    def __init__(self, backbone_name: str = DEFAULT_BACKBONE, pool: str = "all"):
        super().__init__()
        assert pool in ("all", "nonmask")
        self.pool = pool
        self.backbone_name = backbone_name
        self.backbone = load_backbone(backbone_name)
        self.head = nn.Linear(self.backbone.config.hidden_dim, 1)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """xt: (B, L) int64 in [0, MASK_INDEX]. Returns logits (B,); P(real)=sigmoid."""
        timesteps = torch.zeros(xt.shape[0], device=xt.device, dtype=torch.float32)
        out = self.backbone(xt, timesteps=timesteps, output_hidden_states=True,
                            return_dict=True)
        h = out.hidden_states[-1]  # (B, L, H) — output of final block
        if self.pool == "all":
            pooled = h.mean(dim=1)
        else:
            keep = (xt != MASK_INDEX).unsqueeze(-1).to(h.dtype)  # (B, L, 1)
            denom = keep.sum(dim=1).clamp(min=1.0)
            pooled = (h * keep).sum(dim=1) / denom
            # fully-masked rows have no surviving tokens: fall back to all-pool
            all_masked = keep.sum(dim=1).squeeze(-1) == 0
            if bool(all_masked.any()):
                pooled = torch.where(all_masked.unsqueeze(-1), h.mean(dim=1), pooled)
        # the mdlm remote code autocasts its blocks to bf16 on CUDA on its own;
        # keep the head numerically consistent regardless of outer autocast
        return self.head(pooled.to(self.head.weight.dtype)).squeeze(-1)

    def param_groups(self, lr_backbone: float, lr_head: float):
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ]


def save_checkpoint(path, model, optimizer=None, scheduler=None, step=0, extra=None):
    payload = {
        "model": model.state_dict(),
        "backbone_name": model.backbone_name,
        "pool": model.pool,
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload["torch_rng"] = torch.get_rng_state()
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    # atomic write: a job killed mid-save must not leave a truncated checkpoint
    from pathlib import Path
    import os
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_classifier(path, map_location="cpu") -> tuple[MaskedSeqClassifier, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = MaskedSeqClassifier(ckpt["backbone_name"], pool=ckpt["pool"])
    model.load_state_dict(ckpt["model"])
    return model, ckpt
