"""Tests for pydynamo crop."""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
import starfile
import pytest

from pydynamo.commands import crop as crop_cmd
from pydynamo.core.crop import crop_volume, load_tomogram, save_subtomo
from pydynamo.commands.crop import _resolve_num_workers
from pydynamo.io import read_dynamo_tbl, read_vll_to_df, create_dynamo_table


def test_crop_volume_basic():
    """Crop from center of volume."""
    vol = np.random.randn(32, 32, 32).astype(np.float32)
    pos = (17, 17, 17)  # 1-based center
    sub, report = crop_volume(vol, 10, pos, fill=0)
    assert sub.shape == (10, 10, 10)
    assert "out_of_scope" in report


def test_crop_volume_out_of_scope():
    """Crop with partial out-of-scope -> shrink or zeros."""
    vol = np.random.randn(20, 20, 20).astype(np.float32)
    pos = (5, 5, 5)  # Near corner
    sub, report = crop_volume(vol, 16, pos, fill=-1)
    assert report.get("out_of_scope", False) or sub.size > 0


def test_crop_volume_fill_zero():
    """Fill=0: output full size with zeros for out-of-scope."""
    vol = np.ones((24, 24, 24), dtype=np.float32)
    pos = (12, 12, 12)
    sub, _ = crop_volume(vol, 20, pos, fill=0)
    assert sub.shape == (20, 20, 20)


def test_load_save_subtomo():
    """Load and save MRC."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.mrc"
        data = np.random.randn(8, 8, 8).astype(np.float32)
        save_subtomo(data, str(path))
        loaded = load_tomogram(str(path))
        np.testing.assert_array_almost_equal(loaded, data)


def test_read_vll():
    """Read VLL file."""
    with tempfile.NamedTemporaryFile(suffix=".vll", delete=False) as f:
        f.write(b"/path/to/tomo1.mrc\n/path/to/tomo2.mrc\n")
        vll_path = f.name
    try:
        df = read_vll_to_df(vll_path)
        assert len(df) == 2
        assert "rlnMicrographName" in df.columns
        assert "tomo_path" in df.columns
    finally:
        Path(vll_path).unlink(missing_ok=True)


def test_read_write_dynamo_tbl():
    """Read/write Dynamo tbl."""
    pytest.importorskip("eulerangles")
    with tempfile.TemporaryDirectory() as tmp:
        tbl_path = Path(tmp) / "test.tbl"
        coords = np.random.rand(5, 3) * 100
        create_dynamo_table(coords, output_file=str(tbl_path))
        df = read_dynamo_tbl(str(tbl_path))
        assert len(df) == 5
        np.testing.assert_allclose(df[["x", "y", "z"]].values, coords, rtol=1e-4)


def test_resolve_num_workers_auto():
    """num_workers<=0 resolves to all detected CPUs."""
    assert _resolve_num_workers(0) >= 1
    assert _resolve_num_workers(-1) >= 1


def test_crop_tbl_outputs_relion_star(tmp_path: Path):
    """tbl input should produce RELION-style STAR fields, not raw Dynamo columns."""
    tomo = np.random.randn(32, 32, 32).astype(np.float32)
    tomo_path = tmp_path / "tomo1.mrc"
    with mrcfile.new(str(tomo_path), overwrite=True) as mrc:
        mrc.set_data(tomo)

    vll_path = tmp_path / "tomograms.vll"
    vll_path.write_text(str(tomo_path) + "\n", encoding="utf-8")

    tbl_path = tmp_path / "particles.tbl"
    create_dynamo_table(
        np.array([[16.0, 16.0, 16.0]]),
        output_file=str(tbl_path),
        micrograph_names=["tomo1"],
    )

    output_star = tmp_path / "particles.star"
    cfg_path = tmp_path / "crop.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"particles: {tbl_path}",
                f"vll: {vll_path}",
                "tomograms: null",
                f"output_star: {output_star}",
                f"output_dir: {tmp_path / 'out'}",
                "sidelength: 8",
                "fill: 0",
                "num_workers: 1",
                "pixel_size: 4.0",
                "tomogram_size: [32, 32, 32]",
            ]
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = crop_cmd.run(str(cfg_path), [], args)
    assert rc == 0

    df = starfile.read(str(output_star), always_dict=False)
    assert "rlnImageName" in df.columns
    assert "rlnAngleRot" in df.columns
    assert "rlnCenteredCoordinateXAngst" in df.columns
    assert "tdrot" not in df.columns


def test_crop_grouped_loads_tomogram_once_for_single_tomo(tmp_path: Path, monkeypatch):
    """Single-tomogram crop should load tomogram once and process grouped tasks."""
    tomo = np.random.randn(32, 32, 32).astype(np.float32)
    tomo_path = tmp_path / "tomo1.mrc"
    with mrcfile.new(str(tomo_path), overwrite=True) as mrc:
        mrc.set_data(tomo)

    vll_path = tmp_path / "tomograms.vll"
    vll_path.write_text(str(tomo_path) + "\n", encoding="utf-8")

    tbl_path = tmp_path / "particles.tbl"
    create_dynamo_table(
        np.array([[16.0, 16.0, 16.0], [18.0, 18.0, 18.0]]),
        output_file=str(tbl_path),
        micrograph_names=["tomo1", "tomo1"],
    )

    output_star = tmp_path / "particles.star"
    out_dir = tmp_path / "out"
    cfg_path = tmp_path / "crop_grouped.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"particles: {tbl_path}",
                f"vll: {vll_path}",
                f"output_star: {output_star}",
                f"output_dir: {out_dir}",
                "sidelength: 8",
                "fill: 0",
                "num_workers: 4",
            ]
        ),
        encoding="utf-8",
    )

    open_count = {"n": 0}
    orig_open = crop_cmd.mrcfile.open

    def _counting_open(*args, **kwargs):
        open_count["n"] += 1
        return orig_open(*args, **kwargs)

    monkeypatch.setattr(crop_cmd.mrcfile, "open", _counting_open)
    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = crop_cmd.run(str(cfg_path), [], args)
    assert rc == 0
    assert open_count["n"] == 1
