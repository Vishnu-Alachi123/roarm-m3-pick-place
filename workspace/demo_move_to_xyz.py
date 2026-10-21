#!/usr/bin/env python3
"""
Type a target point, see the joint angles the IK computes, then (on the real arm)
confirm before anything moves. With no arm connected it moves the simulation instead.

  python3 demo_move_to_xyz.py          # real arm if a serial port is found, else simulation
  mjpython demo_move_to_xyz.py         # macOS: also opens the simulation viewer

Coordinates are metres: origin on the table under the base axis, x forward, y left, z up.
Commands:  x y z   |   x y z pitch_degrees   |   home   |   q
"""
import math

import armfile
from armfile import POSITIONS, RoArm3Controller

X_MIN, R_MAX = 0.05, 0.40
Z_MIN, Z_MAX = 0.03, 0.30


def main():
    arm = RoArm3Controller(port=armfile.PORT)
    mode = "SIMULATION" if arm.simulated else "REAL ARM"
    print(f"\n=== {mode} ===")
    print("Enter x y z in metres (example: 0.20 0.00 0.10), or 'home', or 'q'.")
    if not arm.simulated:
        print("Keep a hand near the power switch. The first moves use the new IK, unverified on this arm.")
    try:
        while True:
            line = input("target> ").strip().lower()
            if line in ("q", "quit", "exit"):
                break
            if line == "home":
                if arm.simulated or input("Send arm to HOME? [y/N] ").strip().lower() == "y":
                    arm.move_to_position(POSITIONS["home"])
                continue
            try:
                vals = [float(v) for v in line.split()]
                assert len(vals) in (3, 4)
            except (ValueError, AssertionError):
                print("Type three numbers, e.g.  0.20 0.00 0.10")
                continue
            x, y, z = vals[:3]
            pitch = math.radians(vals[3]) if len(vals) == 4 else math.pi / 2
            r = math.hypot(x, y)
            if not (x > X_MIN and r < R_MAX and Z_MIN <= z <= Z_MAX):
                print(f"Outside the safe demo region (x > {X_MIN}, distance < {R_MAX} m, "
                      f"{Z_MIN} <= z <= {Z_MAX} m).")
                continue
            sol, used = armfile.find_ik_auto_pitch(x, y, z, preferred_pitch=pitch,
                                                    q_current=arm._last_angles[:4])
            if sol is None:
                print("No solution: that point is out of reach.")
                continue
            names = ("base", "shoulder", "elbow", "wrist", "roll")
            print("IK solution:", ", ".join(f"{n} {math.degrees(a):.1f} deg" for n, a in zip(names, sol)),
                  f"(gripper pitch {math.degrees(used):.0f} deg)")
            if not arm.simulated and input("Send to the REAL arm? [y/N] ").strip().lower() != "y":
                print("Not sent.")
                continue
            arm.move_to_xyz(x, y, z, wrist_pitch=used)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
