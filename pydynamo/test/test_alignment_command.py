"""Tests for pydynamo alignment command."""
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from pydynamo.commands import alignment


def _write_mrc(path: Path, data: np.ndarray):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))


def test_alignment_command_emits_output_star(tmp_path: Path):
    """alignment command writes refined star output."""
    ref = np.zeros((10, 10, 10), dtype=np.float32)
    ref[4:6, 4:6, 4:6] = 1.0
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
            "x": [5.0],
            "y": [5.0],
            "z": [5.0],
        }
    )
    particles_star = tmp_path / "particles.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    output_star = tmp_path / "aligned.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
        "device_id": 0,
    }
    cfg_path = tmp_path / "alignment.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert output_star.exists()
    out_df = starfile.read(str(output_star), always_dict=False)
    assert len(out_df) == 1
    assert "cc" in out_df.columns


def test_alignment_resolve_execution_devices_auto_cpu(monkeypatch):
    """auto falls back to CPU when CUDA unavailable."""
    monkeypatch.setattr(alignment, "_get_cuda_info", lambda: (False, 0))
    dev, gids = alignment._resolve_execution_devices("auto")
    assert dev == "cpu"
    assert gids == []


def test_alignment_resolve_execution_devices_auto_all_gpus(monkeypatch):
    """auto uses all detected GPU ids by default."""
    monkeypatch.setattr(alignment, "_get_cuda_info", lambda: (True, 3))
    dev, gids = alignment._resolve_execution_devices("auto")
    assert dev == "cuda"
    assert gids == [0, 1, 2]


def test_alignment_auto_dispatches_across_all_gpus(tmp_path: Path, monkeypatch):
    """alignment auto mode distributes tasks across detected GPU ids."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_multi.mrc"
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
    particles_star = tmp_path / "particles_multi.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    monkeypatch.setattr(alignment, "_get_cuda_info", lambda: (True, 2))
    captured_ids = []

    def _fake_align(*_args, **kwargs):
        captured_ids.append(kwargs.get("device_id"))
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(alignment, "align_one_particle", _fake_align)

    output_star = tmp_path / "aligned_multi.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "auto",
    }
    cfg_path = tmp_path / "alignment_multi_gpu.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert set(captured_ids) == {0, 1}
