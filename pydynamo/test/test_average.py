"""Tests for pydynamo reconstruction (average)."""
import numpy as np
import pytest

from pydynamo.core.average import (
    apply_inverse_transform,
    average_particles,
    apply_symmetry,
    euler_zxz_to_rotation_matrix,
)


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
