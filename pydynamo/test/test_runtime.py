"""Tests for runtime logging path resolution."""
import logging
from pathlib import Path
from types import SimpleNamespace

from pydynamo.runtime import log_command_inputs, progress_timing_text, resolve_log_paths


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


def test_log_command_inputs_emits_yaml_block(caplog):
    """Command inputs should be logged with key config values."""
    logger = logging.getLogger("pydynamo.test.runtime")
    args = SimpleNamespace(log_level="info", log_file=None, json_errors=False)
    with caplog.at_level(logging.INFO):
        log_command_inputs(
            logger,
            "alignment",
            config={"particles": "a.star", "nmask": "mask.mrc"},
            config_path="/tmp/a.yaml",
            args=args,
            rest=["--foo", "bar"],
        )
    text = "\n".join(r.message for r in caplog.records)
    assert "Command inputs:" in text
    assert "particles: a.star" in text
    assert "nmask: mask.mrc" in text


def test_progress_timing_text_includes_eta_and_rss_fields():
    """Progress text should include timing and rss observability keys."""
    txt = progress_timing_text(0.0, processed=1, total=2)
    assert "elapsed=" in txt
    assert "eta=" in txt
    assert "eta_at=" in txt
    assert "rss_cur=" in txt
    assert "rss_avg=" in txt
    assert "rss_peak=" in txt
