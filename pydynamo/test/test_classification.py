"""Tests for pydynamo classification command."""
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
import yaml

from pydynamo.commands import classification


def _write_mrc(path: Path, data: np.ndarray):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))


def test_classification_runs_single_iteration(tmp_path: Path):
    """classification command produces iteration outputs."""
    ref = np.zeros((12, 12, 12), dtype=np.float32)
    ref[5:7, 5:7, 5:7] = 1.0
    part = ref.copy()

    ref_path = tmp_path / "ref.mrc"
    part_path = tmp_path / "particle_000001.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, part)

    particles_df = pd.DataFrame(
        {
            "tag": [1],
            "rlnImageName": [str(part_path)],
            "rlnMicrographName": ["tomo1"],
            "x": [6.0],
            "y": [6.0],
            "z": [6.0],
        }
    )
    particles_star = tmp_path / "particles.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    out_dir = tmp_path / "mra_out"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(out_dir),
        "max_iterations": 1,
        "swap": True,
        "cone_step": 180,
        "cone_range": [0, 180],
        "inplane_step": 360,
        "inplane_range": [0, 360],
        "shift_search": 0,
        "multigrid_levels": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    cfg_path = tmp_path / "classification.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg_path), [], args)
    assert rc == 0
    assert (out_dir / "ite_001" / "average_ref_001.mrc").exists()
    assert (out_dir / "ite_001" / "refined_table_ref_001.tbl").exists()


def test_classification_passes_alignment_kwargs(tmp_path: Path, monkeypatch):
    """classification forwards shift/multigrid/device to aligner."""
    ref = np.ones((8, 8, 8), dtype=np.float32)
    ref_path = tmp_path / "ref.mrc"
    part_path = tmp_path / "particle_000001.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, ref)

    particles_df = pd.DataFrame(
        {
            "tag": [1],
            "rlnImageName": [str(part_path)],
            "rlnMicrographName": ["tomo1"],
            "x": [4.0],
            "y": [4.0],
            "z": [4.0],
        }
    )
    particles_star = tmp_path / "particles.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    captured = {}

    def _fake_align(*_args, **kwargs):
        captured.update(kwargs)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(classification, "align_one_particle", _fake_align)

    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(tmp_path / "mra_out"),
        "max_iterations": 1,
        "swap": True,
        "shift_search": 2,
        "multigrid_levels": 2,
        "device": "cpu",
    }
    cfg_path = tmp_path / "classification_kwargs.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg_path), [], args)
    assert rc == 0
    assert captured.get("shift_search") == 2
    assert captured.get("multigrid_levels") == 2
    assert captured.get("device") == "cpu"


def test_classification_auto_uses_all_gpus_for_dispatch(tmp_path: Path, monkeypatch):
    """auto device dispatches tasks across all detected GPU ids."""
    ref = np.ones((8, 8, 8), dtype=np.float32)
    ref_path = tmp_path / "ref.mrc"
    part1 = tmp_path / "particle_000001.mrc"
    part2 = tmp_path / "particle_000002.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part1, ref)
    _write_mrc(part2, ref)

    particles_df = pd.DataFrame(
        {
            "tag": [1, 2],
            "rlnImageName": [str(part1), str(part2)],
            "rlnMicrographName": ["tomo1", "tomo1"],
            "x": [4.0, 4.0],
            "y": [4.0, 4.0],
            "z": [4.0, 4.0],
        }
    )
    particles_star = tmp_path / "particles_2.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    monkeypatch.setattr(classification, "_get_cuda_info", lambda: (True, 2))
    captured_ids = []

    def _fake_align(*_args, **kwargs):
        captured_ids.append(kwargs.get("device_id"))
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(classification, "align_one_particle", _fake_align)

    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(tmp_path / "mra_out_multi_gpu"),
        "max_iterations": 1,
        "swap": True,
        "device": "auto",
    }
    cfg_path = tmp_path / "classification_auto_gpu.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg_path), [], args)
    assert rc == 0
    assert set(captured_ids) == {0, 1}
