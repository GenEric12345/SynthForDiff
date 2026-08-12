"""Config handling: single YAML + argparse dot-notation overrides.

Usage pattern shared by every script:

    parser = base_parser("description")
    ... add script-specific args ...
    args = parser.parse_args()
    cfg = load_config(args)          # default.yaml (+ smoke.yaml if --smoke-test) (+ --config) (+ --set k=v)
    setup_run(cfg, script_name)      # seeds everything, dumps resolved config

Overrides use dot notation:  --set train.lr_head=5e-4 --set data.n_train=1000
Values are YAML-parsed, so `--set eval.t_grid=[0.0,0.5]` works.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        if isinstance(v, dict) and not isinstance(v, Config):
            # wrap IN PLACE so `cfg.a.b = x` mutates this config, not a copy
            v = Config(v)
            self[k] = v
        return v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def to_plain(self) -> dict:
        def conv(x):
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            return x
        return conv(self)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, default=None,
                   help="Extra YAML config layered on top of configs/default.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="Layer configs/smoke.yaml on top (tiny sizes, few steps)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="Dot-notation config override, YAML-parsed")
    p.add_argument("--seed", type=int, default=None, help="Override global seed")
    return p


def load_config(args: argparse.Namespace) -> Config:
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    if getattr(args, "smoke_test", False):
        with open(CONFIG_DIR / "smoke.yaml") as f:
            _deep_update(cfg, yaml.safe_load(f))
        cfg["smoke_test"] = True
    if getattr(args, "config", None):
        with open(args.config) as f:
            _deep_update(cfg, yaml.safe_load(f))
    for ov in getattr(args, "overrides", []):
        if "=" not in ov:
            raise ValueError(f"--set expects KEY=VALUE, got {ov!r}")
        key, _, raw = ov.partition("=")
        val = yaml.safe_load(raw)
        if isinstance(val, str):
            # PyYAML reads bare scientific notation ("5e-4") as a string
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        _set_dotted(cfg, key.strip(), val)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("smoke_test", False)
    return Config(copy.deepcopy(cfg))


def setup_run(cfg: Config, script_name: str) -> None:
    """Seed all RNGs and dump the fully-resolved config for reproducibility."""
    from src.utils import seed_everything, get_logger

    seed_everything(cfg.seed)
    out_dir = REPO_ROOT / "results" / "run_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{script_name}_{stamp}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_plain(), f, sort_keys=False)
    get_logger(script_name).info(
        "seed=%d | resolved config dumped to %s", cfg.seed, os.path.relpath(path, REPO_ROOT))
