"""Tests for pydynamo reconstruction (average)."""
import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from pydynamo.core.average import (
    apply_inverse_transform,
    average_particles,
    apply_symmetry,
    euler_zxz_to_rotation_matrix,
)
from pydynamo.core.align import rotate_volume


def test_rotation_matrix():
    """Euler to matrix."""
    r = euler_zxz_to_rotation_matrix(0, 0, 0)
    np.testing.assert_array_almost_equal(r, np.eye(3))


def test_apply_inverse_transform():
    """Inverse transform round-trip."""
    vol = np.random.randn(12, 12, 12).astype(np.float32)
    tdrot, tilt, narot = 10, 20, 5
    dx, dy, dz = 1, -1, 0
    transformed = apply_inverse_transform(vol, tdrot, tilt, narot, dx, dy, dz)
    assert transformed.shape == vol.shape


def test_apply_inverse_transform_recovers_align_forward_model():
    """Inverse transform should approximately invert align's rotate+shift model."""
    ref = np.zeros((32, 32, 32), dtype=np.float32)
    ref[10:22, 12:20, 13:19] = 1.0
    tdrot, tilt, narot = 30.0, 20.0, 15.0
    dx, dy, dz = 2.0, -1.0, 1.0

    particle = rotate_volume(ref, tdrot, tilt, narot)
    particle = ndi_shift(particle, (dx, dy, dz), order=1, mode="constant", cval=0.0).astype(np.float32)
    recovered = apply_inverse_transform(particle, tdrot, tilt, narot, dx, dy, dz)

    # Validate geometry by correlation, tolerant to interpolation loss.
    a = ref.ravel().astype(np.float64)
    b = recovered.ravel().astype(np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    corr = float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))
    assert corr > 0.75


def test_average_particles():
    """Average multiple particles."""
    particles = [np.random.randn(8, 8, 8).astype(np.float32) for _ in range(3)]
    angles = np.zeros((3, 3))
    shifts = np.zeros((3, 3))
    avg = average_particles(particles, angles, shifts)
    assert avg.shape == (8, 8, 8)


def test_apply_symmetry_c1():
    """C1 symmetry = no change."""
    vol = np.random.randn(8, 8, 8).astype(np.float32)
    out = apply_symmetry(vol, "c1")
    np.testing.assert_array_almost_equal(out, vol)
