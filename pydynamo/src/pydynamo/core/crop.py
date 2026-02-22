"""
Subvolume crop. Matches Dynamo dynamo_crop3d behavior.
Position: 1-based center (Dynamo tbl col 24-26).
Fill: -1=shrink+warning, -2=shrink, 0=zeros for out-of-scope, 1=empty if out-of-scope.
"""
import logging
import os
from typing import Tuple, Union

import mrcfile
import numpy as np

logger = logging.getLogger(__name__)


def crop_volume(
    volume: np.ndarray,
    sidelength: Union[int, Tuple[int, int, int]],
    position: Tuple[float, float, float],
    fill: int = -1,
) -> Tuple[Union[np.ndarray, None], dict]:
    """
    Crop a 3D subvolume centered at position. Matches dynamo_crop3d.

    Parameters
    ----------
    volume : ndarray
        3D volume, shape (nz, ny, nx) as in MRC.
    sidelength : int or (int, int, int)
        Cube edge or (dz, dy, dx).
    position : (float, float, float)
        Center in 1-based voxel coordinates (Dynamo convention).
    fill : int
        -1: shrink + warning; -2: shrink; 0: zeros; 1: empty if out-of-scope.

    Returns
    -------
    subvolume : ndarray or None
    report : dict with 'out_of_scope'
    """
    report = {"out_of_scope": False}
    s1, s2, s3 = volume.shape
    size_volume = np.array([s1, s2, s3], dtype=float)
    pos = np.floor(np.asarray(position, dtype=float))

    if np.isscalar(sidelength):
        sidelength = (sidelength,) * 3
    sidelength = np.array(sidelength, dtype=float)

    # ind{i} = max(1, p-r/2) : min(s, p+r/2-1) for each dim (1-based inclusive)
    ind_start = np.ceil(np.maximum(1, pos - sidelength / 2)).astype(int)
    ind_end = np.floor(np.minimum(size_volume, pos + sidelength / 2 - 1)).astype(int)

    if np.any(ind_end < ind_start):
        if fill == 1:
            return None, {"out_of_scope": True}
        logger.warning("Particle totally out of bounds")
        report["out_of_scope"] = True
        return np.zeros(tuple(int(s) for s in sidelength), dtype=volume.dtype), report

    # 0-based Python slices
    sl = (
        slice(ind_start[0] - 1, ind_end[0]),
        slice(ind_start[1] - 1, ind_end[1]),
        slice(ind_start[2] - 1, ind_end[2]),
    )
    subvolume = volume[sl].copy()

    # Out-of-scope if ideal box would exceed volume
    ideal_low = pos - sidelength / 2
    ideal_high = pos + sidelength / 2 - 1
    if np.any(ideal_low < 1) or np.any(ideal_high > size_volume):
        report["out_of_scope"] = True
        if fill == -1:
            logger.warning("Out of scope access to original volume in dynamo_crop")
        if fill == 1:
            return None, report
        if fill == 0:
            # Place cropped region in zero-padded box of expected size
            # first_inscope = index in ideal range where actual range starts (MATLAB)
            expected = tuple(int(s) for s in sidelength)
            out = np.zeros(expected, dtype=volume.dtype)
            ideal_start = np.ceil(ideal_low).astype(int)
            first_inscope = np.maximum(0, ind_start - ideal_start)
            sz = subvolume.shape
            out[
                first_inscope[0] : first_inscope[0] + sz[0],
                first_inscope[1] : first_inscope[1] + sz[1],
                first_inscope[2] : first_inscope[2] + sz[2],
            ] = subvolume
            return out, report

    return subvolume, report


def load_tomogram(path: str) -> np.ndarray:
    """Load MRC/MRCS tomogram."""
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        return mrc.data.copy()


def save_subtomo(data: np.ndarray, path: str) -> None:
    """Save subtomogram as MRC."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
