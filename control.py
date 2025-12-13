"""Forward kinematics helpers for the UR5/UR5e arms.

This module mirrors the Denavit-Hartenberg convention used by the
provided :mod:`IK_utils` inverse kinematics so both directions stay
consistent.  The main entry point, :func:`urFwdKin`, returns a 4x4
homogeneous transform from ``base_link`` to ``tool0`` for a given set of
joint angles.  Numerical Jacobians are also provided because the
resolved-rate controller needs them to map Cartesian twists to joint
velocities.
"""

from __future__ import annotations

import numpy as np


def _rotz(theta: float) -> np.ndarray:
    """Return a Z-axis rotation matrix."""

    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(alpha: float) -> np.ndarray:
    """Return an X-axis rotation matrix."""

    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Construct a Denavit-Hartenberg transform."""

    T = np.eye(4)
    T[:3, :3] = _rotz(theta) @ _rotx(alpha)
    T[0, 3] = a
    T[2, 3] = d
    return T


def _get_ur_params(robot_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the UR DH parameters (a, alpha, d) for the requested arm."""

    if robot_type not in {"ur5", "ur5e"}:
        raise ValueError("robot_type must be either 'ur5' or 'ur5e'")

    # Nominal UR5 parameters.
    d = np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823])
    a = np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0])
    alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

    if robot_type == "ur5e":
        d = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
        a = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])

    return a, alpha, d


def urFwdKin(q: np.ndarray, robot_type: str = "ur5e") -> np.ndarray:
    """Compute the base_link -> tool0 transform for a UR arm."""

    q = np.asarray(q, dtype=float).flatten()
    if q.size != 6:
        raise ValueError("urFwdKin expects a 6-element joint vector")

    a, alpha, d = _get_ur_params(robot_type)
    g = np.eye(4)
    for i in range(6):
        g = g @ _dh(a[i], alpha[i], d[i], q[i])
    return g


def position_jacobian(q: np.ndarray, robot_type: str = "ur5e", eps: float = 1e-4) -> np.ndarray:
    """Numerically differentiate :func:`urFwdKin` to get dp/dq."""

    q = np.asarray(q, dtype=float).flatten()
    J = np.zeros((3, 6))
    p0 = urFwdKin(q, robot_type)[:3, 3]
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        pi = urFwdKin(q + dq, robot_type)[:3, 3]
        J[:, i] = (pi - p0) / eps
    return J


__all__ = ["urFwdKin", "position_jacobian"]
