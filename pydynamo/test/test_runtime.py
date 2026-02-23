"""Tests for runtime logging path resolution."""
from pathlib import Path
from types import SimpleNamespace

from pydynamo.runtime import resolve_log_paths


def test_resolve_log_paths_defaults_to_yaml_directory(tmp_path: Path):
    """Without explicit paths, defaults should be colocated with YAML file."""
    cfg_path = tmp_path / "reconstruction.yaml"
    cfg_path.write_text("pixel_size: 1.0\n", encoding="utf-8")

    log_path, err_path = resolve_log_paths(
        args=SimpleNamespace(log_file=None),
        config={},
        config_path=str(cfg_path),
    )
    assert log_path == str(tmp_path / "reconstruction.log")
    assert err_path == str(tmp_path / "reconstruction.error.log")


def test_resolve_log_paths_relative_paths_are_yaml_relative(tmp_path: Path):
    """Relative log paths should be resolved under YAML directory."""
    cfg_path = tmp_path / "crop.yaml"
    cfg_path.write_text("pixel_size: 1.0\n", encoding="utf-8")

    log_path, err_path = resolve_log_paths(
        args=SimpleNamespace(log_file=None),
        config={"log_file": "logs/run.log", "error_log_file": "logs/run.error.log"},
        config_path=str(cfg_path),
    )
    assert log_path == str((tmp_path / "logs" / "run.log").resolve())
    assert err_path == str((tmp_path / "logs" / "run.error.log").resolve())
