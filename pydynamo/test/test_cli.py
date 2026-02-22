"""Tests for pydynamo CLI.

Run from project root after: cd pydynamo && pip install -e .
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_help():
    """pydynamo --help lists commands."""
    r = subprocess.run(
        [sys.executable, "-m", "pydynamo.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0
    assert "crop" in r.stdout
    assert "reconstruction" in r.stdout
    assert "alignment" in r.stdout
    assert "classification" in r.stdout


def test_missing_config_exits_nonzero():
    """Missing --i exits with non-zero code."""
    r = subprocess.run(
        [sys.executable, "-m", "pydynamo.cli", "crop"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode != 0


def test_nonexistent_config_exits_1():
    """Nonexistent config file exits with code 1."""
    r = subprocess.run(
        [sys.executable, "-m", "pydynamo.cli", "crop", "--i", "/nonexistent/config.yaml"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode != 0
