"""Tests for YAML configs and CLI-override semantics."""

import argparse

import pytest

from src.utils.config import load_config, parse_args_with_config

yaml = pytest.importorskip("yaml")


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    return p


def test_load_base_configs():
    for name in ["base.yaml", "benchmark.yaml", "reproduce.yaml", "multiseed.yaml"]:
        cfg = load_config(f"configs/{name}")
        assert isinstance(cfg, dict)
        assert "seed" in cfg or "seeds" in cfg


def test_cli_overrides_config(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("epochs: 2\nseed: 7\n", encoding="utf-8")
    args = parse_args_with_config(_parser(), ["--config", str(cfg_path)])
    assert args.epochs == 2
    assert args.seed == 7
    args = parse_args_with_config(_parser(), ["--config", str(cfg_path), "--epochs", "9"])
    assert args.epochs == 9
    assert args.seed == 7


def test_unknown_config_keys_warn_not_silent(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("nonexistent_flag: 123\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="nonexistent_flag"):
        args = parse_args_with_config(_parser(), ["--config", str(cfg_path)])
    assert args.epochs == 35
