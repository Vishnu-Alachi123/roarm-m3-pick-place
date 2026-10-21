#!/usr/bin/env python3
"""
URDF-accurate forward/inverse kinematics for the RoArm-M3 (pure NumPy).

The joint chain below is copied from simulation/roarm_m3_scene.xml (the URDF
exported from roarm_description). simulation/test_kinematics.py checks this
module against MuJoCo's own forward kinematics, so the two cannot drift apart
unnoticed.

Frame: world frame of the MuJoCo scene. Origin on the table directly below the
base axis, x forward, y left, z up (metres). The base joint sits at z=0.0701.

Joint order: [base, shoulder, elbow, wrist, roll, gripper], radians.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# (body name, pos in parent frame, quaternion w x y z, (lower, upper) limit)
# Every joint rotates about the body's local +Z axis.
_CHAIN = [
    ("link1", (0.0, 0.0, 0.0701),            (1.0, 0.0, 0.0, 0.0),                   (-3.1416, 3.1416)),
    ("link2", (0.0, 0.0, 0.051959),          (0.499998, -0.5, -0.5, -0.500002),      (-1.5708, 1.5708)),
    ("link3", (0.236815, 0.030002, 0.0),     (0.707105, 0.0, 0.0, 0.707108),         (-1.0, 2.95)),
    ("link4", (0.0, -0.144586, 0.0),         (1.0, 0.0, 0.0, 0.0),                   (-1.5708, 1.5708)),
    ("link5", (0.015147, -0.053653, 0.0),    (0.499998, 0.5, 0.5, -0.500002),        (-3.1416, 3.1416)),
]
_GRIPPER_BODY = ((0.0, 0.018821, 0.052035), (0.499998, -0.5, -0.5, -0.500002))  # pos, quat in link5
_TCP_IN_GRIPPER = (0.0, -0.011, 0.0774)   # hand_tcp site in gripper_link frame

JOINT_LIMITS = [lim for *_, lim in _CHAIN] + [(0.0, 1.5)]
N_ARM_JOINTS = 5


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _transform(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


_BODY_T = [_transform(_quat_to_mat(q), p) for _, p, q, _ in _CHAIN]
_GRIP_T = _transform(_quat_to_mat(_GRIPPER_BODY[1]), _GRIPPER_BODY[0])


def _link5_transform(q: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    for i in range(N_ARM_JOINTS):
        Rz = np.eye(4)
        Rz[:3, :3] = _rz(q[i])
        T = T @ _BODY_T[i] @ Rz
    return T


def tool_point_in_link5(gripper: Optional[float] = None) -> np.ndarray:
    """Tool point in the link5 frame.

    gripper=None (default): the pinch centre of the gripper, measured from the
    visible gripper mesh: on the roll axis (link5 z), about 10 cm out. It does
    not move when the gripper opens or closes. (The URDF's hand_tcp marker sits
    about 7 cm to the side of the real gripper, so it is not used here.)
    gripper=<angle>: the URDF's hand_tcp site at that gripper-joint angle (used
    to compare against MuJoCo)."""
    if gripper is None:
        return TOOL_POINT_LINK5.copy()
    Rz = np.eye(4)
    Rz[:3, :3] = _rz(gripper)
    return (_GRIP_T @ Rz @ np.append(_TCP_IN_GRIPPER, 1.0))[:3]


DEFAULT_TOOL_GRIPPER = None
# Measured from the gripper meshes in the link5 frame: the hand extends along +z
# (the roll axis) to z = 0.12 m, and its jaws open along x.
TOOL_POINT_LINK5 = np.array([-0.008, 0.0, 0.105])
_APPROACH_IN_LINK5 = np.array([0.0, 0.0, 1.0])


def forward_kinematics(q: Sequence[float], tool_gripper: Optional[float] = DEFAULT_TOOL_GRIPPER
                       ) -> Tuple[np.ndarray, float]:
    """q = [base, shoulder, elbow, wrist, roll(, gripper)] -> (tool xyz, pitch).
    pitch is the approach direction's angle below horizontal, measured in the
    arm's own plane: 0 = pointing straight out from the base, +pi/2 = straight
    down, beyond pi/2 = folded back over toward the base."""
    T = _link5_transform(q)
    pos = (T @ np.append(tool_point_in_link5(tool_gripper), 1.0))[:3]
    approach = T[:3, :3] @ _APPROACH_IN_LINK5
    radial = approach[0] * math.cos(q[0]) + approach[1] * math.sin(q[0])
    pitch = math.atan2(-approach[2], radial)
    return pos, pitch


def _clamp(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i in range(len(out)):
        lo, hi = JOINT_LIMITS[i]
        out[i] = min(max(out[i], lo), hi)
    return out


def _residual(q4: np.ndarray, roll: float, target: np.ndarray, pitch: float,
              tool_gripper: Optional[float]) -> np.ndarray:
    pos, p = forward_kinematics([q4[0], q4[1], q4[2], q4[3], roll], tool_gripper)
    return np.array([pos[0] - target[0], pos[1] - target[1], pos[2] - target[2], 0.1 * (p - pitch)])


def _newton(q0: np.ndarray, roll: float, target: np.ndarray, pitch: float,
            tool_gripper: float, tol: float = 1e-5, iters: int = 80):
    q = _clamp(q0)
    for _ in range(iters):
        r = _residual(q, roll, target, pitch, tool_gripper)
        if np.linalg.norm(r[:3]) < tol and abs(r[3]) < 1e-4:
            return q, True
        J = np.zeros((4, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += 1e-6
            J[:, i] = (_residual(dq, roll, target, pitch, tool_gripper) - r) / 1e-6
        lam = 1e-4
        step = np.linalg.solve(J.T @ J + lam * np.eye(4), -J.T @ r)
        q = _clamp(q + np.clip(step, -0.5, 0.5))
    r = _residual(q, roll, target, pitch, tool_gripper)
    return q, (np.linalg.norm(r[:3]) < tol * 10 and abs(r[3]) < 1e-3)


def inverse_kinematics(x: float, y: float, z: float, pitch: float = 0.0, roll: float = 0.0,
                       q_current: Optional[Sequence[float]] = None,
                       tool_gripper: float = DEFAULT_TOOL_GRIPPER
                       ) -> Optional[Tuple[float, float, float, float, float]]:
    """Solve (x, y, z, pitch) -> (base, shoulder, elbow, wrist, roll), or None
    if the target is out of reach or violates a joint limit at that pitch.
    Several starting guesses are tried; the solution closest to q_current wins,
    so consecutive path points stay on the same arm configuration."""
    target = np.array([x, y, z], dtype=float)
    azimuth = math.atan2(y, x)
    ref = np.array(q_current[:4], dtype=float) if q_current is not None else np.array([azimuth, 0.3, 1.0, 0.0])

    seeds: List[np.ndarray] = [ref]
    for sh in (-0.6, 0.0, 0.6, 1.2):
        for el in (0.2, 1.0, 1.8, 2.6):
            for wr in (-0.8, 0.0, 0.8):
                seeds.append(np.array([azimuth, sh, el, wr]))

    for s in seeds:
        q, ok = _newton(s, roll, target, pitch, tool_gripper)
        if ok:
            return float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(roll)
    return None
