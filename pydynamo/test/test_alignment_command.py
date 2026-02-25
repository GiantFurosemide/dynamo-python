"""Tests for pydynamo alignment command."""
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from pydynamo.commands import alignment
from pydynamo.io import create_dynamo_table
from pydynamo.io import read_dynamo_tbl


def _write_mrc(path: Path, data: np.ndarray):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))


def test_alignment_command_emits_output_star(tmp_path: Path):
    """alignment command writes refined star and reconstructed average."""
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
    assert (tmp_path / "average.mrc").exists()
    out_df = starfile.read(str(output_star), always_dict=False)
    assert len(out_df) == 1
    assert "rlnAngleRot" in out_df.columns
    assert "rlnOriginXAngst" in out_df.columns
    assert "tdrot" not in out_df.columns
    assert "dx" not in out_df.columns


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


def test_alignment_uses_subtomogram_dir_for_relative_star_paths(tmp_path: Path):
    """If STAR has relative rlnImageName, alignment should resolve under subtomograms dir."""
    sub_dir = tmp_path / "out_subtomograms"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref.mrc"
    part_path = sub_dir / "particle_000001.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, ref)

    particles_df = pd.DataFrame(
        {
            "tag": [1],
            "rlnImageName": ["particle_000001.mrc"],  # relative
            "rlnMicrographName": ["tomo1"],
            "x": [4.0],
            "y": [4.0],
            "z": [4.0],
        }
    )
    particles_star = tmp_path / "particles_relative.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    output_star = tmp_path / "aligned_relative.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(sub_dir),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "alignment_relative.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert output_star.exists()
    out_df = starfile.read(str(output_star), always_dict=False)
    assert len(out_df) == 1


def test_alignment_logs_periodic_progress(tmp_path: Path, capsys):
    """alignment emits periodic progress logs controlled by progress_log_every."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_progress.mrc"
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
    particles_star = tmp_path / "particles_progress.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    output_star = tmp_path / "aligned_progress.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
        "progress_log_every": 1,
    }
    cfg_path = tmp_path / "alignment_progress.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Alignment progress" in out


def test_alignment_loads_nmask_from_yaml(tmp_path: Path, monkeypatch):
    """alignment should load YAML nmask and pass it to aligner."""
    ref = np.ones((8, 8, 8), dtype=np.float32)
    ref_path = tmp_path / "ref_mask.mrc"
    part_path = tmp_path / "particle_mask.mrc"
    mask_path = tmp_path / "mask.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, ref)
    mask = np.zeros((8, 8, 8), dtype=np.float32)
    mask[2:6, 2:6, 2:6] = 1.0
    _write_mrc(mask_path, mask)

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
    particles_star = tmp_path / "particles_mask.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    captured = {}

    def _fake_align(*_args, **kwargs):
        captured["args"] = _args
        captured["kwargs"] = kwargs
        captured["mask"] = kwargs.get("mask") if "mask" in kwargs else (_args[2] if len(_args) > 2 else None)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(alignment, "align_one_particle", _fake_align)

    output_star = tmp_path / "aligned_mask.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "cone_step": 180,
        "tdrot_step": 30,
        "tdrot_range": [0, 120],
        "inplane_step": 360,
        "shift_search": 0,
        "shift_mode": "ellipsoid_center",
        "subpixel": False,
        "cc_mode": "roseman_local",
        "cc_local_window": 7,
        "cc_local_eps": 1e-7,
        "device": "cpu",
        "nmask": str(mask_path),
    }
    cfg_path = tmp_path / "alignment_mask.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert captured["mask"] is not None
    assert captured["mask"].dtype == np.bool_
    kw = captured.get("kwargs", {})
    assert kw.get("tdrot_step", 30) == 30
    assert kw.get("tdrot_range", (0, 120)) == (0, 120)
    assert kw.get("shift_mode", "ellipsoid_center") == "ellipsoid_center"
    assert kw.get("subpixel", False) is False
    assert kw.get("cc_mode", "roseman_local") == "roseman_local"
    assert kw.get("cc_local_window", 7) == 7
    assert kw.get("cc_local_eps", 1e-7) == 1e-7
    assert kw.get("angle_sampling_mode", "dynamo") == "dynamo"


def test_alignment_writes_average_to_configured_path(tmp_path: Path):
    """alignment should write reconstructed average to output_average path."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_avg.mrc"
    part_path = tmp_path / "particle_avg.mrc"
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
    particles_star = tmp_path / "particles_avg.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    output_star = tmp_path / "aligned_avg.star"
    output_avg = tmp_path / "custom" / "average_custom.mrc"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "output_average": str(output_avg),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "alignment_avg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert output_avg.exists()


def test_alignment_star_output_stays_relion_schema_for_tbl_input(tmp_path: Path):
    """tbl input + star output should emit RELION schema only (no Dynamo internal columns)."""
    ref = np.zeros((10, 10, 10), dtype=np.float32)
    ref[4:6, 4:6, 4:6] = 1.0
    part = ref.copy()

    sub_dir = tmp_path / "subtomos"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ref_path = tmp_path / "ref_tbl.mrc"
    part_path = sub_dir / "particle_000001.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, part)

    tbl_path = tmp_path / "particles.tbl"
    create_dynamo_table(
        coordinates=np.array([[5.0, 5.0, 5.0]], dtype=float),
        angles_zyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        micrograph_names=["tomo1"],
        origins=np.array([[0.0, 0.0, 0.0]], dtype=float),
        output_file=str(tbl_path),
    )

    output_star = tmp_path / "aligned_from_tbl.star"
    cfg = {
        "particles": str(tbl_path),
        "subtomograms": str(sub_dir),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "output_star": str(output_star),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "alignment_tbl.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert output_star.exists()

    out_df = starfile.read(str(output_star), always_dict=False)
    assert len(out_df) == 1
    assert "rlnAngleRot" in out_df.columns
    assert "rlnOriginXAngst" in out_df.columns
    assert "rlnImageName" in out_df.columns
    # Guardrail: do not leak Dynamo internal fields into STAR.
    forbidden = {"tag", "tdrot", "tilt", "narot", "dx", "dy", "dz", "cc", "cc2", "ref"}
    assert forbidden.isdisjoint(set(out_df.columns))


def test_alignment_writes_tbl_and_star_when_both_configured(tmp_path: Path):
    """When output_table is .tbl and output_star is set, both outputs should be produced."""
    sub_dir = tmp_path / "subtomos"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ref = np.zeros((10, 10, 10), dtype=np.float32)
    ref[4:6, 4:6, 4:6] = 1.0
    part = ref.copy()

    ref_path = tmp_path / "ref_dual.mrc"
    part_path = sub_dir / "particle_000001.mrc"
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
    particles_star = tmp_path / "particles_dual.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    output_tbl = tmp_path / "refined.tbl"
    output_star = tmp_path / "refined.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(sub_dir),
        "reference": str(ref_path),
        "output_table": str(output_tbl),
        "output_star": str(output_star),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "alignment_dual.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert output_tbl.exists()
    assert output_star.exists()

    tbl_df = read_dynamo_tbl(str(output_tbl))
    assert len(tbl_df) == 1
    assert "tdrot" in tbl_df.columns
    assert "dx" in tbl_df.columns
    assert "cc" in tbl_df.columns

    star_df = starfile.read(str(output_star), always_dict=False)
    assert "rlnAngleRot" in star_df.columns
    assert "tdrot" not in star_df.columns


def test_alignment_tbl_streaming_handles_many_metadata_rows(tmp_path: Path):
    """Streaming tbl path should handle larger metadata counts without schema regressions."""
    sub_dir = tmp_path / "subtomos_large"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    part = ref.copy()

    ref_path = tmp_path / "ref_large.mrc"
    part_path = sub_dir / "particle_000001.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, part)

    n_rows = 64
    particles_df = pd.DataFrame(
        {
            "tag": list(range(1, n_rows + 1)),
            "rlnImageName": ["particle_000001.mrc"] * n_rows,
            "rlnMicrographName": ["tomo1"] * n_rows,
            "x": [4.0] * n_rows,
            "y": [4.0] * n_rows,
            "z": [4.0] * n_rows,
        }
    )
    particles_star = tmp_path / "particles_large.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    output_tbl = tmp_path / "refined_large.tbl"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(sub_dir),
        "reference": str(ref_path),
        "output_table": str(output_tbl),
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "multigrid_levels": 1,
        "device": "cpu",
        "progress_log_every": 16,
    }
    cfg_path = tmp_path / "alignment_large.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    tbl_df = read_dynamo_tbl(str(output_tbl))
    assert len(tbl_df) == n_rows


def test_alignment_command_builds_wedge_mask_for_scoring(tmp_path: Path, monkeypatch):
    """alignment command should pass wedge_mask to aligner when enabled."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_wedge.mrc"
    part_path = tmp_path / "particle_wedge.mrc"
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
    particles_star = tmp_path / "particles_wedge.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    captured = {}

    def _fake_align(*_args, **kwargs):
        captured["wedge_mask"] = kwargs.get("wedge_mask")
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(alignment, "align_one_particle", _fake_align)

    output_star = tmp_path / "aligned_wedge.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "apply_wedge_scoring": True,
        "wedge_ftype": 1,
        "wedge_ymin": -30,
        "wedge_ymax": 30,
        "wedge_xmin": -60,
        "wedge_xmax": 60,
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "alignment_wedge.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert captured["wedge_mask"] is not None


def test_alignment_command_multigrid_wedge_scoring_does_not_broadcast_mismatch(tmp_path: Path):
    """multigrid + wedge scoring should run without coarse-stage shape broadcast failures."""
    ref = np.zeros((24, 24, 24), dtype=np.float32)
    ref[10:14, 10:14, 10:14] = 1.0
    ref_path = tmp_path / "ref_mg_wedge.mrc"
    part_path = tmp_path / "particle_mg_wedge.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, ref.copy())

    particles_df = pd.DataFrame(
        {
            "tag": [1],
            "rlnImageName": [str(part_path)],
            "rlnMicrographName": ["tomo1"],
            "x": [12.0],
            "y": [12.0],
            "z": [12.0],
        }
    )
    particles_star = tmp_path / "particles_mg_wedge.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    output_star = tmp_path / "aligned_mg_wedge.star"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_star),
        "apply_wedge_scoring": True,
        "wedge_ftype": 1,
        "wedge_ymin": -30,
        "wedge_ymax": 30,
        "wedge_xmin": -60,
        "wedge_xmax": 60,
        "multigrid_levels": 2,
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "alignment_mg_wedge.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert output_star.exists()


def test_alignment_tbl_fsampling_mode_passes_table_fsampling(tmp_path: Path, monkeypatch):
    """tbl input with fsampling_mode=table should pass per-row fsampling metadata."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_tbl_fsampling.mrc"
    part_path = tmp_path / "particle_tbl_fsampling.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, ref)

    tbl_path = tmp_path / "particles_fsampling.tbl"
    row = np.zeros(35, dtype=float)
    row[0] = 1
    row[1] = 1
    row[2] = 1
    row[12] = 1   # ftype
    row[13] = -30
    row[14] = 30
    row[15] = -60
    row[16] = 60
    row[19] = 1
    row[23] = 4
    row[24] = 4
    row[25] = 4
    row[33] = 1
    with open(tbl_path, "w", encoding="utf-8") as f:
        f.write(" ".join(format(float(v), ".6g") for v in row) + "\n")

    captured = {"fsampling_modes": [], "fsamplings": []}

    def _fake_align(*_args, **kwargs):
        captured["fsampling_modes"].append(kwargs.get("fsampling_mode"))
        captured["fsamplings"].append(kwargs.get("fsampling"))
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(alignment, "align_one_particle", _fake_align)

    output_tbl = tmp_path / "out.tbl"
    cfg = {
        "particles": str(tbl_path),
        "subtomograms": str(tmp_path),
        "reference": str(ref_path),
        "output_table": str(output_tbl),
        "fsampling_mode": "table",
        "wedge_apply_to": "auto",
        "shift_search": 0,
        "cone_step": 180,
        "inplane_step": 360,
        "device": "cpu",
    }
    cfg_path = tmp_path / "alignment_tbl_fsampling.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False, log_file=None)
    rc = alignment.run(str(cfg_path), [], args)
    assert rc == 0
    assert any(m == "table" for m in captured["fsampling_modes"])
    non_none = [v for v in captured["fsamplings"] if v is not None]
    assert len(non_none) >= 1
    assert int(float(non_none[0]["ftype"])) == 1
