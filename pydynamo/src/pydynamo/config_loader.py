"""Load and merge YAML config with defaults."""
import os
from pathlib import Path

import yaml


def get_config_dir():
    """Return path to pydynamo config directory."""
    base = Path(__file__).resolve().parent.parent
    return base / "config"


def load_config(config_path: str, command: str = None) -> dict:
    """
    Load config from path and merge with command defaults.
    User config overrides defaults.
    """
    config_dir = get_config_dir()
    defaults = {}
    if command and (config_dir / f"{command}_defaults.yaml").exists():
        with open(config_dir / f"{command}_defaults.yaml") as f:
            defaults = yaml.safe_load(f) or {}
    user_cfg = {}
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(defaults, user_cfg)


def _deep_merge(a: dict, b: dict) -> dict:
    """Merge b into a; b overrides a."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
