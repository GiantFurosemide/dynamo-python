"""
Dynamo tbl/vll/star I/O. Strictly follows TomoPANDA-pick/utils/io_dynamo.py.
"""
import os
import numpy as np
import pandas as pd
import starfile

from .io_eular import convert_euler

COLUMNS_NAME = {
    1: "tag",
    2: "aligned",
    3: "averaged",
    4: "dx",
    5: "dy",
    6: "dz",
    7: "tdrot",
    8: "tilt",
    9: "narot",
    10: "cc",
    11: "cc2",
    12: "cpu",
    13: "ftype",
    14: "ymintilt",
    15: "ymaxtilt",
    16: "xmintilt",
    17: "xmaxtilt",
    18: "fs1",
    19: "fs2",
    20: "tomo",
    21: "reg",
    22: "class",
    23: "annotation",
    24: "x",
    25: "y",
    26: "z",
    27: "dshift",
    28: "daxis",
    29: "dnarot",
    30: "dcc",
    31: "otag",
    32: "npar",
    34: "ref",
    35: "sref",
}


def read_vll_to_df(vll_path: str) -> pd.DataFrame:
    """Read VLL file: one path per line. Returns DataFrame with rlnMicrographName, tomo_path."""
    names = []
    tomo_paths = []
    with open(vll_path) as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith("#"):
                continue
            base = os.path.basename(p)
            noext, _ = os.path.splitext(base)
            names.append(noext)
            tomo_paths.append(p)
    return pd.DataFrame({"rlnMicrographName": names, "tomo_path": tomo_paths})


def read_dynamo_tbl(tbl_path: str, vll_path=None) -> pd.DataFrame:
    """Read Dynamo .tbl. If vll_path given, map tomo id to rlnMicrographName."""
    rows = []
    max_cols = 0
    with open(tbl_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            parsed = []
            for tok in tokens:
                s = tok.replace("I", "i")
                if "i" in s or "j" in s:
                    s_complex = s.replace("i", "j")
                    try:
                        parsed.append(float(complex(s_complex).real))
                    except Exception:
                        left = s.split("+", 1)[0]
                        try:
                            parsed.append(float(left))
                        except Exception:
                            parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(tok))
                    except Exception:
                        parsed.append(np.nan)
            rows.append(parsed)
            max_cols = max(max_cols, len(parsed))

    normalized = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [np.nan] * (max_cols - len(r))
        normalized.append(r)
    data = np.asarray(normalized, dtype=float)
    ncols = data.shape[1]
    col_names = [COLUMNS_NAME.get(i, f"col{i}") for i in range(1, ncols + 1)]
    df = pd.DataFrame(data, columns=col_names)

    if vll_path is not None and "tomo" in df.columns:
        vll_df = read_vll_to_df(vll_path)
        mapping = {i + 1: vll_df["rlnMicrographName"].iloc[i] for i in range(len(vll_df))}
        mapped = df["tomo"].round().astype(int).map(mapping)
        mapped = mapped.fillna(df["tomo"].astype(int).astype(str))
        df = df.drop(columns=["tomo"])
        df.insert(19, "rlnMicrographName", mapped.values)
    return df


def create_dynamo_table(
    coordinates,
    angles_zyz=None,
    micrograph_names=None,
    origins=None,
    output_file="particles.tbl",
    ref=1,
) -> pd.DataFrame:
    """Create Dynamo .tbl from coordinates, angles, etc."""
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    num_particles = coords.shape[0]

    if angles_zyz is None:
        angles_zyz_arr = np.zeros((num_particles, 3), dtype=float)
    else:
        angles_zyz_arr = np.asarray(angles_zyz, dtype=float)
        if angles_zyz_arr.shape != (num_particles, 3):
            raise ValueError("angles_zyz must have shape (N, 3)")
    angles_zxz = convert_euler(
        angles_zyz_arr, src_convention="relion", dst_convention="dynamo", degrees=True
    )
    angles_zxz = np.atleast_2d(angles_zxz)
    tdrot, tilt, narot = angles_zxz[:, 0], angles_zxz[:, 1], angles_zxz[:, 2]

    if micrograph_names is None:
        tomo_ids = np.ones(num_particles, dtype=int)
    else:
        if len(micrograph_names) != num_particles:
            raise ValueError("micrograph_names length must match coordinates")
        name_to_id = {}
        next_id = 1
        tomo_ids_list = []
        for name in micrograph_names:
            if name not in name_to_id:
                name_to_id[name] = next_id
                next_id += 1
            tomo_ids_list.append(name_to_id[name])
        tomo_ids = np.asarray(tomo_ids_list, dtype=int)

    num_cols = 35
    T = np.zeros((num_particles, num_cols), dtype=float)
    T[:, 0] = np.arange(1, num_particles + 1, dtype=float)
    T[:, 1] = 1.0
    T[:, 2] = 0.0
    if origins is not None:
        origins_arr = np.asarray(origins, dtype=float)
        if origins_arr.shape != (num_particles, 3):
            raise ValueError("origins must have shape (N, 3)")
        T[:, 3:6] = origins_arr
    T[:, 6], T[:, 7], T[:, 8] = tdrot, tilt, narot
    T[:, 19] = tomo_ids.astype(float)
    T[:, 23] = coords[:, 0]
    T[:, 24] = coords[:, 1]
    T[:, 25] = coords[:, 2]
    T[:, 33] = float(ref)

    col_names = [COLUMNS_NAME.get(i, f"col{i}") for i in range(1, num_cols + 1)]
    df = pd.DataFrame(T, columns=col_names)

    int_cols_0based = {0, 1, 2, 12, 19, 20, 21, 22, 30, 31, 33, 34}
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_file, "w") as fh:
        for row in T:
            parts = [
                str(int(round(row[j]))) if j in int_cols_0based else format(float(row[j]), ".6g")
                for j in range(len(row))
            ]
            fh.write(" ".join(parts) + "\n")
    return df


def dynamo_df_to_relion(
    df: pd.DataFrame,
    pixel_size=None,
    tomogram_size=None,
    output_centered=True,
) -> pd.DataFrame:
    """Convert Dynamo DataFrame to RELION-style DataFrame."""
    x_binned = df["x"] if "x" in df.columns else df[COLUMNS_NAME.get(24, "x")]
    y_binned = df["y"] if "y" in df.columns else df[COLUMNS_NAME.get(25, "y")]
    z_binned = df["z"] if "z" in df.columns else df[COLUMNS_NAME.get(26, "z")]

    if output_centered:
        if pixel_size is None or tomogram_size is None:
            raise ValueError("pixel_size and tomogram_size required when output_centered=True")
        tomogram_size = np.asarray(tomogram_size, dtype=float)
        if tomogram_size.shape != (3,):
            raise ValueError(f"tomogram_size must be shape (3,), got {tomogram_size.shape}")
        tomogram_center = tomogram_size / 2.0
        x_out = (x_binned.values - tomogram_center[0]) * float(pixel_size)
        y_out = (y_binned.values - tomogram_center[1]) * float(pixel_size)
        z_out = (z_binned.values - tomogram_center[2]) * float(pixel_size)
        coord_x, coord_y, coord_z = (
            "rlnCenteredCoordinateXAngst",
            "rlnCenteredCoordinateYAngst",
            "rlnCenteredCoordinateZAngst",
        )
    else:
        x_out, y_out, z_out = x_binned.values, y_binned.values, z_binned.values
        coord_x, coord_y, coord_z = "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"

    tdrot = df["tdrot"] if "tdrot" in df.columns else df[COLUMNS_NAME.get(7, "tdrot")]
    tilt = df["tilt"] if "tilt" in df.columns else df[COLUMNS_NAME.get(8, "tilt")]
    narot = df["narot"] if "narot" in df.columns else df[COLUMNS_NAME.get(9, "narot")]
    angles_zxz = np.stack([tdrot.values, tilt.values, narot.values], axis=1)
    angles_zyz = convert_euler(
        angles_zxz, src_convention="dynamo", dst_convention="relion", degrees=True
    )
    angles_zyz = np.atleast_2d(angles_zyz)

    dx = df["dx"] if "dx" in df.columns else df[COLUMNS_NAME.get(4, "dx")]
    dy = df["dy"] if "dy" in df.columns else df[COLUMNS_NAME.get(5, "dy")]
    dz = df["dz"] if "dz" in df.columns else df[COLUMNS_NAME.get(6, "dz")]
    if pixel_size is not None:
        origin_x = -dx.values * float(pixel_size)
        origin_y = -dy.values * float(pixel_size)
        origin_z = -dz.values * float(pixel_size)
    else:
        origin_x = np.zeros_like(dx.values)
        origin_y = np.zeros_like(dy.values)
        origin_z = np.zeros_like(dz.values)

    names = (
        df["rlnMicrographName"].astype(str)
        if "rlnMicrographName" in df.columns
        else (df["tomo"].astype(int).astype(str) if "tomo" in df.columns else pd.Series(["1"] * len(df)))
    )

    return pd.DataFrame(
        {
            coord_x: x_out,
            coord_y: y_out,
            coord_z: z_out,
            "rlnAngleRot": angles_zyz[:, 0],
            "rlnAngleTilt": angles_zyz[:, 1],
            "rlnAnglePsi": angles_zyz[:, 2],
            "rlnOriginXAngst": origin_x,
            "rlnOriginYAngst": origin_y,
            "rlnOriginZAngst": origin_z,
            "rlnMicrographName": names.values,
        }
    )


def dynamo_tbl_vll_to_relion_star(
    tbl_path,
    vll_path=None,
    output_file="particles.star",
    pixel_size=None,
    tomogram_size=None,
    output_centered=True,
) -> pd.DataFrame:
    """Convert Dynamo tbl (+ vll) to RELION star."""
    dynamo_df = read_dynamo_tbl(tbl_path, vll_path=vll_path)
    relion_df = dynamo_df_to_relion(
        dynamo_df,
        pixel_size=pixel_size,
        tomogram_size=tomogram_size,
        output_centered=output_centered,
    )
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    starfile.write(relion_df, output_file)
    return relion_df


def relion_star_to_dynamo_tbl(
    star_path,
    pixel_size,
    tomogram_size=None,
    output_file="particles.tbl",
) -> pd.DataFrame:
    """Convert RELION star to Dynamo tbl."""
    star_data = starfile.read(star_path, always_dict=False)
    particles_df = None
    optics_df = None
    if isinstance(star_data, dict):
        if "particles" in star_data:
            particles_df = star_data["particles"]
        else:
            df_candidates = [v for v in star_data.values() if isinstance(v, pd.DataFrame)]
            if df_candidates:
                particles_df = df_candidates[0]
            else:
                raise ValueError(f"No DataFrame in STAR dict: {star_path}")
        if "optics" in star_data:
            optics_df = star_data["optics"]
    elif isinstance(star_data, pd.DataFrame):
        particles_df = star_data
    else:
        raise ValueError(f"Unexpected starfile type: {type(star_data)}")

    df = particles_df
    required = [
        "rlnCenteredCoordinateXAngst",
        "rlnCenteredCoordinateYAngst",
        "rlnCenteredCoordinateZAngst",
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in STAR: {missing}")

    coords_angstrom = np.stack(
        [df["rlnCenteredCoordinateXAngst"].values, df["rlnCenteredCoordinateYAngst"].values, df["rlnCenteredCoordinateZAngst"].values],
        axis=1,
    )
    coords_pixels_centered = coords_angstrom / float(pixel_size)
    if tomogram_size is not None:
        tomogram_size = np.asarray(tomogram_size, dtype=float)
        coords_pixels = coords_pixels_centered + (tomogram_size / 2.0)
    elif optics_df is not None and all(c in optics_df.columns for c in ["rlnImageSizeX", "rlnImageSizeY", "rlnImageSizeZ"]):
        ts = np.array([optics_df["rlnImageSizeX"].iloc[0], optics_df["rlnImageSizeY"].iloc[0], optics_df["rlnImageSizeZ"].iloc[0]])
        coords_pixels = coords_pixels_centered + (ts / 2.0)
    else:
        coords_pixels = coords_pixels_centered

    angles_zyz = np.stack([df["rlnAngleRot"].values, df["rlnAngleTilt"].values, df["rlnAnglePsi"].values], axis=1)
    if all(c in df.columns for c in ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]):
        origins_angstrom = np.stack([df["rlnOriginXAngst"].values, df["rlnOriginYAngst"].values, df["rlnOriginZAngst"].values], axis=1)
        origins_pixels = -origins_angstrom / float(pixel_size)
    else:
        origins_pixels = np.zeros((len(df), 3), dtype=float)

    if "rlnMicrographName" in df.columns:
        micrograph_names = df["rlnMicrographName"].astype(str).tolist()
    elif "rlnTomoName" in df.columns:
        micrograph_names = df["rlnTomoName"].astype(str).tolist()
    else:
        micrograph_names = None
    if micrograph_names is None:
        raise ValueError("No rlnMicrographName or rlnTomoName in STAR")

    dynamo_df = create_dynamo_table(
        coordinates=coords_pixels,
        angles_zyz=angles_zyz,
        micrograph_names=micrograph_names,
        origins=origins_pixels,
        output_file=output_file,
    )
    dynamo_df["rlnMicrographName"] = micrograph_names
    return dynamo_df
