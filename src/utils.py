"""Seeding, logging, small shared helpers."""
from __future__ import annotations

import hashlib
import logging
import random
import sys

import numpy as np
import torch


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s %(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(global_seed: int, *scope: str | int) -> int:
    """Deterministic per-purpose seed, e.g. derive_seed(cfg.seed, 'gen', doc_id, attempt).

    Keeps independent randomness streams (splits / masking / generation retries)
    decoupled while everything remains a pure function of the global seed.
    """
    payload = f"{global_seed}|" + "|".join(str(s) for s in scope)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def doc_hash(token_ids, n: int = 256) -> str:
    """Dedup / doc_id hash over the first `n` token ids."""
    b = np.asarray(list(token_ids[:n]), dtype=np.int64).tobytes()
    return hashlib.sha1(b).hexdigest()[:16]
