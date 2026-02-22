"""pydynamo alignment — align subtomograms against reference(s)."""
import logging
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
import yaml

from ..core.align import align_one_particle
from ..io import read_dynamo_tbl

logger = logging.getLogger(__name__)


def run(config_path: str, rest: list, args) -> int:
    """Run alignment command. Returns exit code."""
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    config = _load_config(config_path, args)

    particles = config.get("particles")
    subtomograms = config.get("subtomograms")
    reference = config.get("reference")
    output_table = config.get("output_table")
    output_star = config.get("output_star") or output_table
    vll_path = config.get("vll") or config.get("vll_path")
    cone_step = float(config.get("cone_step", 15))
    cone_range = tuple(config.get("cone_range", [0, 180]))
    inplane_step = float(config.get("inplane_step", 15))
    inplane_range = tuple(config.get("inplane_range", [0, 360]))
    shift_search = int(config.get("shift_search", 3))
    lowpass = config.get("lowpass")
    pixel_size = float(config.get("pixel_size", 1.0))
    multigrid_levels = int(config.get("multigrid_levels", 1))
    device = str(config.get("device", "auto"))

    if not all([particles, subtomograms, reference, output_table]):
        _err("Missing required: particles, subtomograms, reference, output_table", args)
        return 1

    # Load reference
    with mrcfile.open(reference, mode="r", permissive=True) as mrc:
        ref_vol = mrc.data.copy().astype(np.float32)

    # Load particle table
    if particles.endswith(".star"):
        tbl_df = starfile.read(particles, always_dict=False)
        if isinstance(tbl_df, dict):
            tbl_df = tbl_df.get("particles", list(tbl_df.values())[0])
        paths_col = "rlnImageName" if "rlnImageName" in tbl_df.columns else None
    else:
        tbl_df = read_dynamo_tbl(particles, vll_path=vll_path)
        paths_col = None

    if paths_col and paths_col in tbl_df.columns:
        paths_list = tbl_df[paths_col].tolist()
    else:
        sub_path = Path(subtomograms)
        paths_list = sorted(sub_path.glob("*.mrc")) + sorted(sub_path.glob("*.mrcs"))
        paths_list = [str(p) for p in paths_list]
        if len(paths_list) < len(tbl_df):
            paths_list = (paths_list * (len(tbl_df) // len(paths_list) + 1))[: len(tbl_df)]

    base_dir = Path(particles).parent if isinstance(particles, str) else Path(".")
    rows = []
    for i, p_path in enumerate(paths_list):
        full_path = base_dir / p_path if not Path(p_path).is_absolute() else Path(p_path)
        try:
            with mrcfile.open(str(full_path), mode="r", permissive=True) as mrc:
                part = mrc.data.copy().astype(np.float32)
        except Exception as e:
            logger.warning("Skip %s: %s", full_path, e)
            continue
        if part.shape != ref_vol.shape:
            logger.warning("Shape mismatch %s vs ref", part.shape)
            continue

        tdrot, tilt, narot, dx, dy, dz, cc = align_one_particle(
            part, ref_vol,
            cone_step=cone_step, cone_range=cone_range,
            inplane_step=inplane_step, inplane_range=inplane_range,
            shift_search=shift_search, lowpass_angstrom=lowpass, pixel_size=pixel_size,
            multigrid_levels=multigrid_levels, device=device,
        )
        row = tbl_df.iloc[i].to_dict() if i < len(tbl_df) else {}
        row.update({
            "tag": int(row.get("tag", i + 1)),
            "tdrot": tdrot, "tilt": tilt, "narot": narot,
            "dx": dx, "dy": dy, "dz": dz,
            "cc": cc, "cc2": cc,
            "aligned": 1, "averaged": 1,
            "ref": int(row.get("ref", 1)),
        })
        if paths_col:
            row["rlnImageName"] = str(p_path)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(output_star)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(out_path).endswith(".star"):
        starfile.write(out_df, str(out_path))
    else:
        from ..io import create_dynamo_table, convert_euler
        coords = out_df[["x", "y", "z"]].values if "x" in out_df.columns else np.zeros((len(out_df), 3))
        ang_zxz = np.column_stack([out_df["tdrot"], out_df["tilt"], out_df["narot"]])
        ang_zyz = convert_euler(ang_zxz, src_convention="dynamo", dst_convention="relion", degrees=True)
        create_dynamo_table(
            coords,
            angles_zyz=ang_zyz,
            micrograph_names=out_df["rlnMicrographName"].tolist() if "rlnMicrographName" in out_df.columns else None,
            origins=out_df[["dx", "dy", "dz"]].values,
            output_file=str(out_path),
        )

    logger.info("Aligned %d particles, output: %s", len(rows), output_star)
    return 0


def _load_config(path: str, args=None) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        if args:
            _err(f"Config file not found: {path}", args)
        raise


def _err(msg: str, args, code: int = 1):
    if getattr(args, "json_errors", False):
        import json
        print(json.dumps({"error": msg, "code": code}), file=sys.stderr)
    else:
        print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)
