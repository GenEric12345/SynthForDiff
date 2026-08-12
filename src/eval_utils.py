"""Shared scoring/AUC machinery for evaluate, controls, and t_min annotation.

Orientation: label 1 = real, classifier score = P(real); AUC uses real as the
positive class (see src/classifier.py docstring).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.masking import apply_random_mask
from src.utils import derive_seed


@torch.no_grad()
def score_documents(
    model,
    tokens: np.ndarray,          # (N, L) int64 GPT-2 ids
    t: float,
    n_masks: int,
    batch_size: int,
    device: torch.device,
    mask_seed: int,
    use_bf16: bool = True,
) -> np.ndarray:
    """Mean P(real) over `n_masks` fresh mask realizations per document. (N,)"""
    model.eval()
    x_all = torch.from_numpy(np.ascontiguousarray(tokens))
    gen = torch.Generator(device=device)
    gen.manual_seed(mask_seed)
    out = np.zeros(len(tokens), dtype=np.float64)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if (
        use_bf16 and device.type == "cuda") else torch.autocast("cpu", enabled=False)
    for start in range(0, len(tokens), batch_size):
        x = x_all[start:start + batch_size].to(device)
        acc = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
        for _ in range(n_masks):
            xt = apply_random_mask(x, t, generator=gen)  # fresh mask every call
            with autocast:
                logits = model(xt)
            acc += torch.sigmoid(logits.float()).double()
        out[start:start + batch_size] = (acc / n_masks).cpu().numpy()
    return out


def paired_bootstrap_auc(
    scores_real: np.ndarray,
    scores_synth: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """AUC + 95% CI, bootstrapping over DOCUMENTS.

    scores_real[i] and scores_synth[i] belong to the same document (same
    prompt), so resampling is paired: a resampled document contributes both its
    real and its synthetic score.
    """
    n = len(scores_real)
    assert len(scores_synth) == n
    y = np.concatenate([np.ones(n), np.zeros(n)])
    s = np.concatenate([scores_real, scores_synth])
    auc = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sb = np.concatenate([scores_real[idx], scores_synth[idx]])
        boots[b] = roc_auc_score(y, sb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def permutation_null_auc(
    scores_real: np.ndarray,
    scores_synth: np.ndarray,
    n_perm: int,
    seed: int,
    percentile: float = 97.5,
) -> tuple[float, np.ndarray]:
    """Label-permutation null distribution of AUC; returns (percentile, all AUCs)."""
    n = len(scores_real)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    s = np.concatenate([scores_real, scores_synth])
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_perm)
    for p in range(n_perm):
        yp = rng.permutation(y)
        aucs[p] = roc_auc_score(yp, s)
    return float(np.percentile(aucs, percentile)), aucs


def eval_mask_seed(global_seed: int, purpose: str, t: float) -> int:
    return derive_seed(global_seed, purpose, f"{t:.4f}")
