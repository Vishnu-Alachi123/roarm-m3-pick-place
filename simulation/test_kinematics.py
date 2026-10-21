#!/usr/bin/env python3
"""Checks workspace/roarm_kinematics.py against MuJoCo's own forward kinematics."""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workspace"))
import roarm_kinematics as K  # noqa: E402

model = mujoco.MjModel.from_xml_path(str(ROOT / "simulation" / "roarm_m3_scene.xml"))
data = mujoco.MjData(model)
JOINTS = ["base_link_to_link1", "link1_to_link2", "link2_to_link3",
          "link3_to_link4", "link4_to_link5", "link5_to_gripper_link"]
qadr = [model.joint(n).qposadr[0] for n in JOINTS]
link5 = model.body("link5").id
grip = model.body("gripper_link").id
site = model.site("hand_tcp").id

rng = np.random.default_rng(0)
lims = np.array(K.JOINT_LIMITS)

# 1. Forward kinematics vs MuJoCo (tool point at the same gripper angle)
worst = 0.0
for _ in range(500):
    q = rng.uniform(lims[:, 0], lims[:, 1])
    for a, v in zip(qadr, q):
        data.qpos[a] = v
    mujoco.mj_forward(model, data)
    mj_tcp = data.site_xpos[site]
    mine, _ = K.forward_kinematics(q[:5], tool_gripper=q[5])
    worst = max(worst, float(np.linalg.norm(mj_tcp - mine)))
print(f"FK vs MuJoCo, 500 random poses: worst error = {worst:.2e} m")

# 2. Inverse kinematics: reachable targets generated from random poses
ok, errs, times = 0, [], []
n = 100
for _ in range(n):
    q = rng.uniform(lims[:5, 0], lims[:5, 1])
    q[1] = rng.uniform(-1.2, 1.2)
    q[3] = rng.uniform(-1.2, 1.2)
    target, pitch = K.forward_kinematics(q)
    if target[2] < 0.03 or abs(pitch) > 1.45:
        continue
    t0 = time.perf_counter()
    sol = K.inverse_kinematics(*target, pitch=pitch)
    times.append(time.perf_counter() - t0)
    if sol is None:
        continue
    reached, p2 = K.forward_kinematics(sol)
    errs.append(float(np.linalg.norm(reached - target)))
    ok += 1
tested = len(times)
print(f"IK on {tested} reachable targets: solved {ok}, max position error "
      f"{max(errs) * 1000:.4f} mm, mean solve time {np.mean(times) * 1000:.0f} ms")

# 3. Unreachable target must fail cleanly
print("IK for (5, 5, 5):", K.inverse_kinematics(5, 5, 5))
