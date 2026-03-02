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
    """classification command produces iteration outputs and final average."""
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
    assert (out_dir / "average.mrc").exists()


def test_classification_passes_alignment_kwargs(tmp_path: Path, monkeypatch):
    """classification forwards shift/multigrid/device to aligner."""
    ref = np.ones((8, 8, 8), dtype=np.float32)
    ref_path = tmp_path / "ref.mrc"
    part_path = tmp_path / "particle_000001.mrc"
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
    particles_star = tmp_path / "particles.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    captured = {}

    def _fake_align(*_args, **kwargs):
        captured["args"] = _args
        captured["kwargs"] = kwargs
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
        "tdrot_step": 20,
        "tdrot_range": [0, 200],
        "shift_search": 2,
        "shift_mode": "ellipsoid_follow",
        "subpixel": False,
        "cc_mode": "roseman_local",
        "cc_local_window": 7,
        "cc_local_eps": 1e-7,
        "multigrid_levels": 2,
        "device": "cpu",
        "nmask": str(mask_path),
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
    assert captured.get("mask") is not None
    kw = captured.get("kwargs", {})
    assert kw.get("tdrot_step", 20) == 20
    assert kw.get("tdrot_range", (0, 200)) == (0, 200)
    assert kw.get("shift_mode", "ellipsoid_follow") == "ellipsoid_follow"
    assert kw.get("subpixel", False) is False
    assert kw.get("cc_mode", "roseman_local") == "roseman_local"
    assert kw.get("cc_local_window", 7) == 7
    assert kw.get("cc_local_eps", 1e-7) == 1e-7
    assert kw.get("angle_sampling_mode", "dynamo") == "dynamo"


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


def test_classification_uses_subtomogram_dir_for_relative_star_paths(tmp_path: Path):
    """Relative STAR paths should resolve under subtomograms directory."""
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

    out_dir = tmp_path / "mra_out_relative"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(sub_dir),
        "references": [str(ref_path)],
        "output_dir": str(out_dir),
        "max_iterations": 1,
        "swap": True,
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "classification_relative.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg_path), [], args)
    assert rc == 0
    assert (out_dir / "ite_001" / "refined_table_ref_001.tbl").exists()


def test_classification_writes_average_to_configured_path(tmp_path: Path):
    """classification should write final average to output_average path."""
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

    out_dir = tmp_path / "mra_out_avg"
    out_avg = tmp_path / "custom" / "classification_average.mrc"
    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(out_dir),
        "output_average": str(out_avg),
        "max_iterations": 1,
        "swap": True,
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "multigrid_levels": 1,
        "device": "cpu",
    }
    cfg_path = tmp_path / "classification_avg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg_path), [], args)
    assert rc == 0
    assert out_avg.exists()


def test_classification_chunk_size_equivalence(tmp_path: Path):
    """chunk_size=1 and chunk_size=100 produce equivalent outputs (p_017, f_016)."""
    sub_dir = tmp_path / "subtomos"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_chunk.mrc"
    _write_mrc(ref_path, ref)
    n_particles = 4
    particles_df = pd.DataFrame(
        {
            "tag": list(range(1, n_particles + 1)),
            "rlnImageName": [str(sub_dir / f"particle_{i:06d}.mrc") for i in range(1, n_particles + 1)],
            "rlnMicrographName": ["tomo1"] * n_particles,
            "x": [4.0] * n_particles,
            "y": [4.0] * n_particles,
            "z": [4.0] * n_particles,
        }
    )
    for i in range(1, n_particles + 1):
        _write_mrc(sub_dir / f"particle_{i:06d}.mrc", ref)
    particles_star = tmp_path / "particles_chunk.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    def _run(chunk_size: int):
        out_dir = tmp_path / f"mra_c{chunk_size}"
        cfg = {
            "particles": str(particles_star),
            "subtomograms": str(sub_dir),
            "references": [str(ref_path)],
            "output_dir": str(out_dir),
            "max_iterations": 1,
            "swap": True,
            "cone_step": 90,
            "inplane_step": 90,
            "shift_search": 1,
            "multigrid_levels": 1,
            "device": "cpu",
            "num_workers": 2,
            "chunk_size": chunk_size,
        }
        cfg_path = tmp_path / f"class_c{chunk_size}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        args = SimpleNamespace(log_level="warning", json_errors=False)
        rc = classification.run(str(cfg_path), [], args)
        assert rc == 0
        tbl_path = out_dir / "ite_001" / "refined_table_ref_001.tbl"
        avg_path = out_dir / "ite_001" / "average_ref_001.mrc"
        assert tbl_path.exists()
        assert avg_path.exists()
        with open(tbl_path) as f:
            n_rows = sum(1 for line in f if line.strip())
        avg = mrcfile.open(str(avg_path), mode="r", permissive=True).data
        return n_rows, avg

    n1, avg1 = _run(1)
    n2, avg2 = _run(100)
    assert n1 == n2, "per-ref row counts should match"
    np.testing.assert_allclose(avg1, avg2, atol=1e-4, rtol=1e-3)


def test_classification_resume_from_checkpoint_runs_remaining_iterations_only(tmp_path: Path, monkeypatch):
    """Resume should continue from latest checkpoint, not restart from ite_001."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_resume.mrc"
    part_path = tmp_path / "particle_resume.mrc"
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
    particles_star = tmp_path / "particles_resume.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    out_dir = tmp_path / "mra_out_resume"
    cfg1 = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(out_dir),
        "max_iterations": 1,
        "swap": True,
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "multigrid_levels": 1,
        "device": "cpu",
    }
    cfg1_path = tmp_path / "classification_resume_1.yaml"
    with open(cfg1_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg1, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg1_path), [], args)
    assert rc == 0
    assert (out_dir / "ite_001" / "checkpoint.yaml").exists()

    calls = {"n": 0}

    def _fake_align(*_args, **_kwargs):
        calls["n"] += 1
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(classification, "align_one_particle", _fake_align)

    cfg2 = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(out_dir),
        "max_iterations": 2,
        "resume": True,
        "swap": True,
        "cone_step": 180,
        "inplane_step": 360,
        "shift_search": 0,
        "multigrid_levels": 1,
        "device": "cpu",
    }
    cfg2_path = tmp_path / "classification_resume_2.yaml"
    with open(cfg2_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg2, f, sort_keys=False)

    rc = classification.run(str(cfg2_path), [], args)
    assert rc == 0
    assert calls["n"] == 1
    assert (out_dir / "ite_002" / "checkpoint.yaml").exists()
    assert (out_dir / "ite_002" / "refined_table_ref_001.tbl").exists()


def test_classification_builds_wedge_mask_for_scoring(tmp_path: Path, monkeypatch):
    """classification should pass wedge_mask to aligner when enabled."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_cls_wedge.mrc"
    part_path = tmp_path / "particle_cls_wedge.mrc"
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
    particles_star = tmp_path / "particles_cls_wedge.star"
    starfile.write(particles_df, str(particles_star), overwrite=True)

    captured = {}

    def _fake_align(*_args, **kwargs):
        captured["wedge_mask"] = kwargs.get("wedge_mask")
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    monkeypatch.setattr(classification, "align_one_particle", _fake_align)

    cfg = {
        "particles": str(particles_star),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(tmp_path / "mra_out_cls_wedge"),
        "max_iterations": 1,
        "swap": True,
        "apply_wedge_scoring": True,
        "wedge_ftype": 1,
        "wedge_ymin": -30,
        "wedge_ymax": 30,
        "wedge_xmin": -60,
        "wedge_xmax": 60,
        "device": "cpu",
    }
    cfg_path = tmp_path / "classification_wedge.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg_path), [], args)
    assert rc == 0
    assert captured["wedge_mask"] is not None


def test_classification_tbl_fsampling_mode_passes_table_fsampling(tmp_path: Path, monkeypatch):
    """classification should pass table fsampling metadata when enabled."""
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    ref[3:5, 3:5, 3:5] = 1.0
    ref_path = tmp_path / "ref_cls_tbl_fs.mrc"
    part_path = tmp_path / "particle_cls_tbl_fs.mrc"
    _write_mrc(ref_path, ref)
    _write_mrc(part_path, ref)

    tbl_path = tmp_path / "particles_cls_fsampling.tbl"
    row = np.zeros(35, dtype=float)
    row[0] = 1
    row[1] = 1
    row[2] = 1
    row[12] = 2   # ftype
    row[13] = -20
    row[14] = 20
    row[15] = -40
    row[16] = 40
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

    monkeypatch.setattr(classification, "align_one_particle", _fake_align)

    cfg = {
        "particles": str(tbl_path),
        "subtomograms": str(tmp_path),
        "references": [str(ref_path)],
        "output_dir": str(tmp_path / "mra_out_tbl_fs"),
        "max_iterations": 1,
        "swap": True,
        "fsampling_mode": "table",
        "wedge_apply_to": "auto",
        "device": "cpu",
    }
    cfg_path = tmp_path / "classification_tbl_fsampling.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    args = SimpleNamespace(log_level="info", json_errors=False)
    rc = classification.run(str(cfg_path), [], args)
    assert rc == 0
    assert any(m == "table" for m in captured["fsampling_modes"])
    non_none = [v for v in captured["fsamplings"] if v is not None]
    assert len(non_none) >= 1
    assert int(float(non_none[0]["ftype"])) == 2
