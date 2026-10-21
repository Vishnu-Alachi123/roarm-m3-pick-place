#!/usr/bin/env python3
"""
Builds simulation/roarm_m3_pick.xml, a physics version of roarm_m3_scene.xml:

  - the URDF arm meshes stay as visuals (no contact), joints keep their limits
  - simplified parallel-jaw fingers (two sliding boxes) carry the contact physics,
    because MuJoCo collides meshes as convex hulls and the URDF jaw hulls overlap
  - table (top at z = 0), a 2.5 cm cube with a free joint
  - position actuators with damping and armature so the servos behave

Run:  python3 build_pick_scene.py
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "workspace"))
import roarm_kinematics as K  # noqa: E402

SRC = HERE / "roarm_m3_scene.xml"
OUT = HERE / "roarm_m3_pick.xml"

GRIP_FIXED = 0.0                         # the URDF jaw is frozen at this angle (visual only)
CUBE_HALF = 0.0125
FINGER_HALF_GAP = 0.030                  # finger centre offset from tool point when open
FINGER_THICK = 0.003
FINGER_TRAVEL = 0.025


def mat_to_quat(R: np.ndarray):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def main():
    xml = SRC.read_text()

    # 1. meshes become visual-only
    xml = xml.replace('type="mesh"', 'type="mesh" contype="0" conaffinity="0"')

    # 2. freeze the URDF gripper jaw at GRIP_FIXED (visual), drop its joint and actuator
    xml, n = re.subn(r'\s*<joint name="link5_to_gripper_link"[^>]*/>', "", xml)
    assert n == 1
    xml, n = re.subn(r'\s*<position name="gripper_act"[^>]*/>', "", xml)
    assert n == 1
    q_old = K._GRIPPER_BODY[1]
    R_new = K._GRIP_T[:3, :3] @ K._rz(GRIP_FIXED)
    q_new = mat_to_quat(R_new)
    old_body = '<body name="gripper_link" pos="0 0.018821 0.052035" quat="0.499998 -0.5 -0.5 -0.500002">'
    assert old_body in xml
    xml = xml.replace(old_body, f'<body name="gripper_link" pos="0 0.018821 0.052035" quat="{fmt(q_new)}">')

    # 3. fingers on link5, placed around the tool point
    P = K.tool_point_in_link5(None)                  # pinch centre on the roll axis
    # tool axes (columns) in the link5 frame: z = approach (roll axis), y = jaw spread (link5 x)
    R_t = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    s_axis = R_t[:, 1]
    q_t = fmt(mat_to_quat(R_t))
    left = P + s_axis * FINGER_HALF_GAP
    right = P - s_axis * FINGER_HALF_GAP
    fingers = f"""
              <site name="pinch" pos="{fmt(P)}" quat="{q_t}" size="0.004" rgba="0 1 0 0.6"/>
              <body name="finger_l" pos="{fmt(left)}" quat="{q_t}">
                <joint name="finger_l_slide" type="slide" axis="0 -1 0" range="0 {FINGER_TRAVEL}" damping="2"/>
                <geom name="finger_l_geom" type="box" size="0.008 {FINGER_THICK} 0.018" friction="1.5 0.02 0.001" rgba="0.15 0.15 0.18 1" mass="0.003"/>
              </body>
              <body name="finger_r" pos="{fmt(right)}" quat="{q_t}">
                <joint name="finger_r_slide" type="slide" axis="0 1 0" range="0 {FINGER_TRAVEL}" damping="2"/>
                <geom name="finger_r_geom" type="box" size="0.008 {FINGER_THICK} 0.018" friction="1.5 0.02 0.001" rgba="0.15 0.15 0.18 1" mass="0.003"/>
              </body>
"""
    marker = '<body name="gripper_link"'
    idx = xml.index(marker)
    xml = xml[:idx] + fingers.strip("\n") + "\n              " + xml[idx:]

    # 4. world: simulator options, table, cube; the old floor plane moves below the table
    xml = xml.replace('<option timestep="0.002" gravity="0 0 -9.81"/>',
                      '<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>'
                      '\n  <default><joint damping="0.05" armature="0.002"/></default>')
    xml = xml.replace('<geom name="floor" type="plane" size="1 1 0.01" material="grid"/>',
                      '<geom name="floor" type="plane" size="1 1 0.01" pos="0 0 -0.05" material="grid"/>\n'
                      '    <geom name="table" type="box" size="0.30 0.30 0.025" pos="0.18 0 -0.025" '
                      'rgba="0.55 0.45 0.35 1" friction="1.0 0.02 0.001"/>\n'
                      f'    <body name="cube" pos="0.22 0 {CUBE_HALF}">\n'
                      '      <freejoint name="cube_free"/>\n'
                      f'      <geom name="cube_geom" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" '
                      'mass="0.015" friction="1.5 0.02 0.001" rgba="0.85 0.15 0.12 1"/>\n'
                      '    </body>\n'
                      '    <body name="goal" mocap="true" pos="0.33 0 0.1">\n'
                      '      <geom type="sphere" size="0.012" contype="0" conaffinity="0" rgba="0.1 0.9 0.3 0.45"/>\n'
                      '    </body>')

    # 5. actuators: damped position servos for the arm, one per finger
    act_new = f"""  <actuator>
    <position name="base_act" joint="base_link_to_link1" kp="30" kv="1.0" ctrlrange="-3.1416 3.1416"/>
    <position name="shoulder_act" joint="link1_to_link2" kp="60" kv="2.0" ctrlrange="-1.5708 1.5708"/>
    <position name="elbow_act" joint="link2_to_link3" kp="40" kv="1.5" ctrlrange="-1 2.95"/>
    <position name="wrist_act" joint="link3_to_link4" kp="20" kv="0.8" ctrlrange="-1.5708 1.5708"/>
    <position name="roll_act" joint="link4_to_link5" kp="10" kv="0.4" ctrlrange="-3.1416 3.1416"/>
    <position name="finger_l_act" joint="finger_l_slide" kp="200" kv="4" ctrlrange="0 {FINGER_TRAVEL}"/>
    <position name="finger_r_act" joint="finger_r_slide" kp="200" kv="4" ctrlrange="0 {FINGER_TRAVEL}"/>
  </actuator>"""
    xml, n = re.subn(r"  <actuator>.*?</actuator>", act_new, xml, flags=re.S)
    assert n == 1

    OUT.write_text(xml)
    print(f"wrote {OUT}")
    print(f"tool point in link5: {np.round(P, 4)}, jaw spread axis in link5: {np.round(s_axis, 3)}")


if __name__ == "__main__":
    main()
