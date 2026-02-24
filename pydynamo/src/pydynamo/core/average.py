"""
Subtomogram average. Applies inverse transform (Euler ZXZ + shift) and sums.
Wedge/Fourier compensation: optional, simplified.
"""
import numpy as np
from scipy.ndimage import map_coordinates, shift
from scipy.spatial.transform import Rotation


def euler_zxz_to_rotation_matrix(tdrot: float, tilt: float, narot: float, degrees: bool = True) -> np.ndarray:
    """Dynamo ZXZ (tdrot, tilt, narot) to 3x3 rotation matrix (intrinsic)."""
    r = Rotation.from_euler("ZXZ", [tdrot, tilt, narot], degrees=degrees)
    return r.as_matrix()


def apply_inverse_transform(
    volume: np.ndarray,
    tdrot: float,
    tilt: float,
    narot: float,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """
    Apply inverse of (rotation, shift) to bring particle into reference frame.
    Dynamo convention: ZXZ Euler in degrees; dx,dy,dz in voxels.
    """
    # Forward model in align path is approximately:
    #   particle ~= Shift(dx,dy,dz) * Rotate(R) * reference
    # so inverse must apply reverse composition:
    #   reference ~= Rotate(R)^(-1) * Shift^(-1) * particle
    #
    # 1) inverse translation
    unshifted = shift(volume, (-dx, -dy, -dz), order=1, mode="constant", cval=0)

    # 2) exact inverse rotation from rotation object (avoid Euler sign-negation pitfalls)
    r = Rotation.from_euler("ZXZ", [tdrot, tilt, narot], degrees=True)
    r_inv = r.inv().as_matrix()
    center = np.array(unshifted.shape) / 2.0 - 0.5
    coords = np.mgrid[
        : unshifted.shape[0],
        : unshifted.shape[1],
        : unshifted.shape[2],
    ].astype(float) - center.reshape(3, 1, 1, 1)
    coords_flat = coords.reshape(3, -1)
    rotated = r_inv @ coords_flat
    rotated += center.reshape(3, 1)
    coords_new = rotated.reshape(3, unshifted.shape[0], unshifted.shape[1], unshifted.shape[2])

    out = map_coordinates(unshifted, coords_new, order=1, mode="constant", cval=0)
    return out.astype(np.float32)


def average_particles(
    particles: list[np.ndarray],
    angles: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """
    Average particles after applying inverse transform.
    angles: (N, 3) tdrot, tilt, narot
    shifts: (N, 3) dx, dy, dz
    """
    if len(particles) == 0:
        raise ValueError("No particles to average")
    ref_shape = particles[0].shape
    for p in particles:
        if p.shape != ref_shape:
            raise ValueError(f"Particle shape mismatch: {p.shape} vs {ref_shape}")

    acc = np.zeros(ref_shape, dtype=np.float64)
    for i, vol in enumerate(particles):
        tr = apply_inverse_transform(
            vol,
            angles[i, 0],
            angles[i, 1],
            angles[i, 2],
            shifts[i, 0],
            shifts[i, 1],
            shifts[i, 2],
        )
        acc += tr
    return (acc / len(particles)).astype(np.float32)


def apply_symmetry(volume: np.ndarray, sym: str) -> np.ndarray:
    """Apply cyclic symmetry (c2, c4, etc). c1 = no op."""
    sym = sym.lower()
    if sym == "c1" or sym == "" or sym is None:
        return volume
    if sym.startswith("c") and sym[1:].isdigit():
        n = int(sym[1:])
        if n <= 1:
            return volume
        center = np.array(volume.shape) / 2.0 - 0.5
        angles = np.linspace(0, 360, n, endpoint=False)[1:]  # skip 0
        out = volume.copy().astype(np.float64)
        for ang in angles:
            r = Rotation.from_euler("Z", ang, degrees=True)
            r_inv = r.as_matrix()
            coords = np.mgrid[
                : volume.shape[0],
                : volume.shape[1],
                : volume.shape[2],
            ].astype(float) - center.reshape(3, 1, 1, 1)
            coords_flat = coords.reshape(3, -1)
            rotated = r_inv @ coords_flat
            rotated += center.reshape(3, 1)
            coords_new = rotated.reshape(3, *volume.shape)
            rot_vol = map_coordinates(volume, coords_new, order=1, mode="constant", cval=0)
            out += rot_vol
        return (out / n).astype(np.float32)
    return volume
