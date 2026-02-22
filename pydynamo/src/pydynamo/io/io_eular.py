"""Euler angle conversion (RELION ZYZ <-> Dynamo ZXZ). Follows TomoPANDA-pick/utils/io_eular."""
import numpy as np

try:
    from eulerangles import convert_eulers
except ImportError:
    convert_eulers = None


def convert_euler(
    angles,
    src_convention="ZYZ",
    dst_convention="ZXZ",
    degrees=True,
    intrinsic=True,
):
    """Convert Euler angles between conventions (e.g. RELION ZYZ <-> Dynamo ZXZ)."""
    if convert_eulers is None:
        raise ImportError("eulerangles package required. Install with: pip install eulerangles")

    arr = np.asarray(angles, dtype=float)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
        single = True

    src_lower = src_convention.lower()
    dst_lower = dst_convention.lower()
    convention_map = {"zyz": "relion", "zxz": "dynamo"}
    src_meta = convention_map.get(src_lower, src_lower)
    dst_meta = convention_map.get(dst_lower, dst_lower)

    if src_meta in ("relion", "dynamo") and dst_meta in ("relion", "dynamo"):
        out = convert_eulers(arr, source_meta=src_meta, target_meta=dst_meta)
    else:
        from eulerangles import angles2matrix, matrix2angles

        mat = angles2matrix(arr, axes=src_convention.upper(), intrinsic=intrinsic, degrees=degrees)
        out = matrix2angles(mat, axes=dst_convention.upper(), intrinsic=intrinsic, degrees=degrees)

    if single:
        return out.reshape(3,)
    return out
