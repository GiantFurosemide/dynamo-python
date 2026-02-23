"""Tests for synthetic data generator."""
from pathlib import Path

import mrcfile
import numpy as np
import starfile
import yaml

from pydynamo.scripts.generate_synthetic import run


def test_generate_synthetic_small_dataset(tmp_path: Path):
    """gen_synthetic creates tomogram/subtomograms/classification outputs."""
    template = np.zeros((16, 16, 16), dtype=np.float32)
    template[7:9, 7:9, 7:9] = 1.0
    template_path = tmp_path / "template.mrc"
    with mrcfile.new(str(template_path), overwrite=True) as mrc:
        mrc.set_data(template)
        mrc.voxel_size = 4.0

    out_root = tmp_path / "synthetic"
    cfg = {
        "template": str(template_path),
        "output_root": str(out_root),
        "n_particles": 4,
        "n_noise": 2,
        "tomogram_size": [48, 48, 48],
        "apply_missing_wedge": True,
        "wedge_ftype": 1,
        "wedge_ymin": -48,
        "wedge_ymax": 48,
        "noise_sigma": 0.01,
        "particle_scale_ratio": 2.0,
        "seed": 1,
    }
    cfg_path = tmp_path / "synthetic.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run(config_path=str(cfg_path))

    tomo_dir = out_root / "out_tomograms"
    sub_dir = out_root / "out_subtomograms"
    class_dir = out_root / "out_tomograms4classification"
    assert (tomo_dir / "tomo1.mrc").exists()
    assert (tomo_dir / "particles.tbl").exists()
    assert (sub_dir / "particles.star").exists()
    assert (class_dir / "particles.star").exists()

    cls_df = starfile.read(str(class_dir / "particles.star"), always_dict=False)
    assert len(cls_df) == 6
