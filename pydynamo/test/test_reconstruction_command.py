"""Tests for reconstruction command path resolution."""
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from pydynamo.commands import reconstruction


def _write_mrc(path: Path, data: np.ndarray):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))


def test_reconstruction_uses_subtomogram_dir_for_relative_star_paths(tmp_path: Path):
    """If STAR has relative rlnImageName, reconstruction should resolve under subtomograms dir."""
    sub_dir = tmp_path / "out_subtomograms"
    sub_dir.mkdir(parents=True, exist_ok=True)
    vol = np.random.randn(8, 8, 8).astype(np.float32)
    _write_mrc(sub_dir / "particle_000001.mrc", vol)

    star_df = pd.DataFrame(
        {
            "rlnImageName": ["particle_000001.mrc"],  # relative basename
            "rlnMicrographName": ["tomo1"],
            "rlnAngleRot": [0.0],
            "rlnAngleTilt": [0.0],
            "rlnAnglePsi": [0.0],
            "rlnOriginXAngst": [0.0],
            "rlnOriginYAngst": [0.0],
            "rlnOriginZAngst": [0.0],
            "averaged": [1],
        }
    )
    particles_star = tmp_path / "particles.star"
    starfile.write(star_df, str(particles_star), overwrite=True)

    out_mrc = tmp_path / "average.mrc"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(sub_dir),
        "output": str(out_mrc),
        "sidelength": 8,
        "symmetry": "c1",
        "pixel_size": 1.0,
    }
    cfg_path = tmp_path / "reconstruction.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = reconstruction.run(str(cfg_path), [], args)
    assert rc == 0
    assert out_mrc.exists()
    with mrcfile.open(str(out_mrc), permissive=True) as mrc:
        assert abs(float(mrc.voxel_size.x) - 1.0) < 1e-6


def test_reconstruction_applies_nmask(tmp_path: Path):
    """reconstruction should apply nmask to particle contributions."""
    sub_dir = tmp_path / "subs"
    sub_dir.mkdir(parents=True, exist_ok=True)
    vol = np.ones((8, 8, 8), dtype=np.float32)
    _write_mrc(sub_dir / "particle_000001.mrc", vol)
    mask = np.zeros((8, 8, 8), dtype=np.float32)
    mask[:4, :, :] = 1.0
    mask_path = tmp_path / "mask.mrc"
    _write_mrc(mask_path, mask)

    star_df = pd.DataFrame(
        {
            "rlnImageName": ["particle_000001.mrc"],
            "rlnMicrographName": ["tomo1"],
            "rlnAngleRot": [0.0],
            "rlnAngleTilt": [0.0],
            "rlnAnglePsi": [0.0],
            "rlnOriginXAngst": [0.0],
            "rlnOriginYAngst": [0.0],
            "rlnOriginZAngst": [0.0],
            "averaged": [1],
        }
    )
    particles_star = tmp_path / "particles_mask.star"
    starfile.write(star_df, str(particles_star), overwrite=True)

    out_mrc = tmp_path / "average_masked.mrc"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(sub_dir),
        "output": str(out_mrc),
        "sidelength": 8,
        "symmetry": "c1",
        "pixel_size": 1.0,
        "nmask": str(mask_path),
    }
    cfg_path = tmp_path / "reconstruction_mask.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = reconstruction.run(str(cfg_path), [], args)
    assert rc == 0
    with mrcfile.open(str(out_mrc), permissive=True) as mrc:
        out = np.array(mrc.data, copy=True)
    assert float(out[:4, :, :].mean()) > 0.9
    assert float(np.abs(out[4:, :, :]).max()) < 1e-6
