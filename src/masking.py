"""Masked-diffusion forward process, matching the MDLM reference implementation.

Conventions verified against third_party/mdlm (github.com/kuleshov-group/mdlm):

* Mask token id (third_party/mdlm/diffusion.py:85-90): the GPT-2 tokenizer has no
  mask_token, so MDLM appends one AFTER the vocab:
      mask_index = tokenizer.vocab_size          # = 50257 for GPT-2
      vocab_size = tokenizer.vocab_size + 1      # = 50258
  This matches the pretrained checkpoint: kuleshov-group/mdlm-owt has
  config.vocab_size == 50258, so id 50257 is [MASK].

* Forward process q(x_t | x_0) (third_party/mdlm/diffusion.py:575-586, `q_xt`):
      move_indices = torch.rand(*x.shape) < move_chance
      xt = torch.where(move_indices, mask_index, x)
  i.e. every token is INDEPENDENTLY replaced by [MASK] with prob move_chance.

* Log-linear schedule (third_party/mdlm/noise_schedule.py:126-144 LogLinearNoise):
  sigma(t) = -log1p(-(1-eps)t)  =>  move_chance = 1 - exp(-sigma) = (1-eps)*t
  with eps=1e-3. We use move_chance = t exactly (the eps->0 limit); the deviation
  from MDLM's training-time (0.999*t) is <= 0.1% of t and the experiment's
  semantics ("each token masked with probability t") are exact by construction.

Masks are sampled with FRESH randomness on every call. Never cache masked
sequences anywhere in this codebase.
"""
from __future__ import annotations

import torch

GPT2_VOCAB_SIZE = 50257
MASK_INDEX = 50257           # == gpt2 vocab_size; appended [MASK], see docstring
VOCAB_SIZE_WITH_MASK = 50258


def apply_random_mask(
    x: torch.Tensor,
    t: torch.Tensor | float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x_0=x) with per-token mask probability t.

    Args:
        x: int64 tensor (..., L) of GPT-2 token ids in [0, 50256].
        t: scalar, or tensor broadcastable to x.shape (e.g. (B, 1) for a
           per-example mask ratio).
        generator: optional torch.Generator for reproducible masks.

    Returns:
        Same shape as x; masked positions replaced with MASK_INDEX (50257).
    """
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=x.device)
    t = t.to(device=x.device, dtype=torch.float32)
    u = torch.rand(x.shape, device=x.device, generator=generator)
    move_indices = u < t
    return torch.where(move_indices, torch.full_like(x, MASK_INDEX), x)


def sample_t(
    n: int,
    t_min: float,
    t_max: float,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """t ~ Uniform(t_min, t_max), shape (n,)."""
    u = torch.rand(n, device=device, generator=generator)
    return t_min + (t_max - t_min) * u
