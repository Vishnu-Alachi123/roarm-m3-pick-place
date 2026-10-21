#!/usr/bin/env python3
"""
MuJoCo simulation of the RoArm-M3, built from the real URDF/meshes exported
from the ROS2 workspace (roarm_description package). Runs natively on macOS
— no Docker/X11 needed for this path.

Joint order matches armfile.py: [base, shoulder, elbow, wrist, roll, gripper]
so an ArmPosition written against this simulator can later be replayed on
the real arm via RoArm3Controller with no changes to the position data.
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(__file__).parent / "roarm_m3_scene.xml"

JOINT_NAMES = [
    "base_link_to_link1",   # base
    "link1_to_link2",       # shoulder
    "link2_to_link3",       # elbow
    "link3_to_link4",       # wrist
    "link4_to_link5",       # roll
    "link5_to_gripper_link",  # gripper
]

# From the URDF's <limit lower=".." upper="..">, radians.
JOINT_LIMITS = [
    (-3.1416, 3.1416),  # base
    (-1.5708, 1.5708),  # shoulder
    (-1.0, 2.95),        # elbow
    (-1.5708, 1.5708),  # wrist
    (-3.1416, 3.1416),  # roll
    (0.0, 1.5),          # gripper
]

HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.6]


class RoArmSim:
    """Mirrors RoArm3Controller's interface (armfile.py) so pick-and-place
    logic written against the simulator ports to real hardware unchanged."""

    def __init__(self, launch_viewer: bool = True):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_idx = [
            self.model.joint(name).qposadr[0] for name in JOINT_NAMES
        ]
        self.viewer = (
            mujoco.viewer.launch_passive(self.model, self.data)
            if launch_viewer
            else None
        )
        self.move_joints(HOME, settle=True)

    def move_joints(self, angles, duration: float = 1.0, settle: bool = False):
        """Smoothly interpolate to target joint angles (radians), same
        [base, shoulder, elbow, wrist, roll, gripper] order as armfile.py."""
        for angle, (lo, hi) in zip(angles, JOINT_LIMITS):
            if not (lo <= angle <= hi):
                raise ValueError(f"joint angle {angle:.3f} outside limit [{lo}, {hi}]")

        start = np.array([self.data.qpos[i] for i in self.joint_qpos_idx])
        target = np.array(angles)
        steps = max(1, int(duration / self.model.opt.timestep))

        for step in range(steps):
            t = (step + 1) / steps
            interp = start + (target - start) * t
            for idx, val in zip(self.joint_qpos_idx, interp):
                self.data.qpos[idx] = val
            mujoco.mj_forward(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

        if settle:
            time.sleep(0.1)

    def get_joint_angles(self):
        return [self.data.qpos[i] for i in self.joint_qpos_idx]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def demo():
    """Same motion shape as armfile.py's pick-and-place cycle: home -> reach
    -> lower -> grip -> lift -> place -> release -> home."""
    sim = RoArmSim()
    try:
        sequence = [
            ("home", HOME),
            ("above target", [0.5, 0.3, 0.6, 0.0, 0.0, 0.6]),
            ("lower to pick", [0.5, 0.5, 0.9, 0.0, 0.0, 0.6]),
            ("close gripper", [0.5, 0.5, 0.9, 0.0, 0.0, 1.4]),
            ("lift", [0.5, 0.1, 0.4, 0.0, 0.0, 1.4]),
            ("move to place", [-0.5, 0.3, 0.6, 0.0, 0.0, 1.4]),
            ("lower to place", [-0.5, 0.5, 0.9, 0.0, 0.0, 1.4]),
            ("open gripper", [-0.5, 0.5, 0.9, 0.0, 0.0, 0.6]),
            ("home", HOME),
        ]
        for label, angles in sequence:
            print(f"-> {label}: {[round(a, 2) for a in angles]}")
            sim.move_joints(angles, duration=1.2)
            time.sleep(0.3)
        print("Demo sequence complete. Close the viewer window to exit.")
        while sim.viewer is not None and sim.viewer.is_running():
            time.sleep(0.1)
    finally:
        sim.close()


if __name__ == "__main__":
    demo()
