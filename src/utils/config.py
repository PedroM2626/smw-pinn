"""YAML config loader with CLI-override semantics.

Pattern used by entry points::

    parser = argparse.ArgumentParser(...)
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=35)
    ...
    args = parse_args_with_config(parser, args=None)

If ``--config`` points to a YAML file, its keys become parser defaults, so
explicit CLI flags still win over the file.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: Optional[str]) -> dict[str, Any]:
    """Load a YAML config file into a plain dict (empty dict if path is None)."""
    if path is None:
        return {}
    if yaml is None:
        raise ImportError("pyyaml is required for --config (pip install pyyaml)")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping at top level")
    return data


def parse_args_with_config(
    parser: argparse.ArgumentParser, args: list[str] | None = None
) -> argparse.Namespace:
    """Two-pass parse: read --config first, apply as defaults, re-parse."""
    known, _ = parser.parse_known_args(args)
    config_path = getattr(known, "config", None)
    if config_path:
        cfg = load_config(config_path)
        # Only keys that match an existing argument take effect; warn loudly
        # on the rest so typos (e.g. `epoch: 10`) never pass silently.
        valid = {a.dest for a in parser._actions}
        unknown = sorted(k for k in cfg if k not in valid)
        if unknown:
            warnings.warn(
                f"Ignoring unknown config keys in {config_path}: {unknown}. "
                f"Valid keys: {sorted(valid)}.",
                UserWarning,
                stacklevel=2,
            )
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in valid})
    return parser.parse_args(args)
