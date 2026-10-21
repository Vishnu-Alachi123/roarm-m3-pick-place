#!/usr/bin/env python3
"""
Scripted pick-and-place in the MuJoCo physics scene, driven by the URDF-accurate IK.

  python3 scripted_pick.py            # headless, prints success/failure
  mjpython scripted_pick.py --view    # watch it in the MuJoCo viewer (macOS)

Sequence: open -> hover over cube -> descend -> close fingers -> lift -> move -> lower -> release.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "workspace"))
import roarm_kinematics as K  # noqa: E402

ARM_ACT = ["base_act", "shoulder_act", "elbow_act", "wrist_act", "roll_act"]
ARM_JOINTS = ["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"]
FINGER_CLOSED_CMD = 0.025
PITCH = math.pi / 2   # straight down; reachable from x = 0.08 m to 0.34 m at table height
FINGER_LEN, FINGER_DEPTH = 0.018, 0.008


class PickWorld:
    def __init__(self, view: bool = False):
        self.model = mujoco.MjModel.from_xml_path(str(HERE / "roarm_m3_pick.xml"))
        self.data = mujoco.MjData(self.model)
        self.act = [self.model.actuator(n).id for n in ARM_ACT]
        self.fl = self.model.actuator("finger_l_act").id
        self.fr = self.model.actuator("finger_r_act").id
        self.q_adr = [self.model.joint(j).qposadr[0] for j in ARM_JOINTS]
        cube = self.model.joint("cube_free")
        self.cube_adr = cube.qposadr[0]
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if view else None
        self.q_cmd = np.array([0.0, 0.3, 1.2, 0.0, 0.0])
        self._apply(self.q_cmd, 0.0)
        for _ in range(500):
            self.step()

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)

    def _apply(self, q, grip):
        for a, v in zip(self.act, q):
            self.data.ctrl[a] = v
        self.data.ctrl[self.fl] = grip
        self.data.ctrl[self.fr] = grip

    def cube_pos(self):
        return self.data.qpos[self.cube_adr:self.cube_adr + 3].copy()

    def joints(self):
        return np.array([self.data.qpos[a] for a in self.q_adr])

    def move_joints(self, q_target, grip, seconds=1.2):
        q_target = np.asarray(q_target, dtype=float)
        start, g0 = self.q_cmd.copy(), float(self.data.ctrl[self.fl])
        n = max(1, int(seconds / self.model.opt.timestep))
        for i in range(n):
            t = (i + 1) / n
            t = t * t * (3 - 2 * t)
            self._apply(start + (q_target - start) * t, g0 + (grip - g0) * t)
            self.step()
        self.q_cmd = q_target
        for _ in range(250):
            self.step()

    def move_tool(self, x, y, z, grip, pitch=PITCH, roll=0.0, seconds=1.2):
        sol = K.inverse_kinematics(x, y, z, pitch=pitch, roll=roll, q_current=self.q_cmd)
        if sol is None:
            raise RuntimeError(f"IK failed for ({x:.3f}, {y:.3f}, {z:.3f}) pitch {math.degrees(pitch):.0f} deg")
        self.move_joints(sol, grip, seconds)
        return sol


def run(view: bool = False, cube_xy=(0.22, 0.0), place_xy=(0.22, 0.10)):
    w = PickWorld(view)
    cx, cy = cube_xy
    w.data.qpos[w.cube_adr:w.cube_adr + 3] = [cx, cy, 0.0125]
    mujoco.mj_forward(w.model, w.data)
    for _ in range(200):
        w.step()

    # lowest point of the tilted finger box sits 3 mm above the table
    grasp_z = 0.003 + FINGER_LEN * math.sin(PITCH) + FINGER_DEPTH * math.cos(PITCH)
    print(f"cube at {np.round(w.cube_pos(), 3)}")
    w.move_tool(cx, cy, 0.10, grip=0.0)
    w.move_tool(cx, cy, grasp_z, grip=0.0, seconds=1.5)
    w.move_tool(cx, cy, grasp_z, grip=FINGER_CLOSED_CMD, seconds=1.0)
    w.move_tool(cx, cy, 0.12, grip=FINGER_CLOSED_CMD, seconds=1.5)
    held = w.cube_pos()
    lifted = held[2] > 0.06
    print(f"after lift: cube at {np.round(held, 3)}  -> {'HELD' if lifted else 'DROPPED'}")
    px, py = place_xy
    w.move_tool(px, py, 0.12, grip=FINGER_CLOSED_CMD, seconds=1.5)
    w.move_tool(px, py, grasp_z, grip=FINGER_CLOSED_CMD, seconds=1.5)
    w.move_tool(px, py, grasp_z, grip=0.0, seconds=0.8)
    w.move_tool(px, py, 0.12, grip=0.0, seconds=1.2)
    final = w.cube_pos()
    err = float(np.linalg.norm(final[:2] - np.array(place_xy)))
    placed = lifted and err < 0.03 and final[2] < 0.03
    print(f"final cube at {np.round(final, 3)}, {err * 1000:.1f} mm from the place target -> {'PLACED' if placed else 'FAILED'}")
    if w.viewer is not None:
        while w.viewer.is_running():
            time.sleep(0.1)
    return placed


def run_interactive():
    """Pick the cube from wherever it is, put it down, wait while you move it, repeat."""
    import mujoco.viewer  # noqa: F401
    w = PickWorld(view=True)
    grasp_z = 0.003 + FINGER_LEN * math.sin(PITCH) + FINGER_DEPTH * math.cos(PITCH)
    print("Drag the cube anywhere on the table (double-click it, then Ctrl + right-drag).")
    print("Each round the arm picks it up from wherever it is. Close the viewer to quit.")
    n = 0
    while w.viewer.is_running():
        for sec in range(8, 0, -1):
            print(f"  picking in {sec} s ...   ", end="\r")
            for _ in range(int(1.0 / w.model.opt.timestep)):
                w.step()
            if not w.viewer.is_running():
                return
        cx, cy, cz = w.cube_pos()
        if not (0.10 <= math.hypot(cx, cy) <= 0.29 and cx > 0.05 and cz < 0.05):
            print(f"\n  cube at ({cx:.2f}, {cy:.2f}) is outside the arm's reach (keep it 10 to 29 cm from the base, in front of it) or off the table; move it and wait.")
            continue
        n += 1
        px, py = (0.22, -0.12) if cy > 0 else (0.22, 0.12)
        print(f"\nround {n}: picking from ({cx:.3f}, {cy:.3f}), placing at ({px:.2f}, {py:.2f})")
        try:
            w.move_tool(cx, cy, 0.10, 0.0)
            w.move_tool(cx, cy, grasp_z, 0.0, seconds=1.5)
            w.move_tool(cx, cy, grasp_z, FINGER_CLOSED_CMD, seconds=1.0)
            w.move_tool(cx, cy, 0.12, FINGER_CLOSED_CMD, seconds=1.5)
            w.move_tool(px, py, 0.12, FINGER_CLOSED_CMD, seconds=1.5)
            w.move_tool(px, py, grasp_z, FINGER_CLOSED_CMD, seconds=1.5)
            w.move_tool(px, py, grasp_z, 0.0, seconds=0.8)
            w.move_tool(px, py, 0.12, 0.0, seconds=1.2)
        except RuntimeError as e:
            print("  skipped:", e)
            w.move_tool(0.22, 0.0, 0.12, 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--interactive", action="store_true", help="move the cube yourself; the arm picks it up wherever it is")
    args = ap.parse_args()
    if args.interactive:
        run_interactive()
        sys.exit(0)
    if args.view:
        import mujoco.viewer  # noqa: F401
    sys.exit(0 if run(args.view) else 1)
