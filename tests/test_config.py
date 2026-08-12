"""Config plumbing: nested attribute writes must persist (regression test for
the --resume/--fixed-t/--wandb flags), and --set overrides must be YAML-typed."""
import argparse

from src.config import Config, load_config


def test_nested_attribute_write_persists():
    cfg = Config({"train": {"resume": None, "lr": 1e-3}})
    cfg.train.resume = "auto"
    assert cfg["train"]["resume"] == "auto"
    assert cfg.train.resume == "auto"
    assert cfg.to_plain()["train"]["resume"] == "auto"


def test_load_config_and_set_overrides():
    args = argparse.Namespace(
        smoke_test=False, config=None, seed=None,
        overrides=["train.lr_head=5e-4", "eval.t_grid=[0.0,0.5]",
                   "model.pool=nonmask"])
    cfg = load_config(args)
    assert cfg.train.lr_head == 5e-4
    assert cfg.eval.t_grid == [0.0, 0.5]
    assert cfg.model.pool == "nonmask"
    # untouched defaults still present and typed
    assert isinstance(cfg.data.n_train, int)
    assert cfg.model.backbone == "kuleshov-group/mdlm-owt"


def test_smoke_layer_applies():
    args = argparse.Namespace(smoke_test=True, config=None, seed=42, overrides=[])
    cfg = load_config(args)
    assert cfg.data.n_train == 400
    assert cfg.seed == 42
    assert cfg.smoke_test is True
    # smoke must not touch the science-critical lengths
    assert cfg.data.prompt_len == 128
    assert cfg.data.cont_len == 960
