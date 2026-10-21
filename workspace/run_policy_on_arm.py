#!/usr/bin/env python3
"""
Run ANY policy trained with simulation/pick_env.py on the RoArm-M3 (or the simulation if no arm is found).

  # reach policy: move the gripper to a point
  mjpython run_policy_on_arm.py --task reach --goal 0.26 0.05 0.08

  # pick policy: pick the cube you placed at x y (metres), lift it to the goal
  mjpython run_policy_on_arm.py --task lift --cube 0.22 0.05 --goal 0.22 0.0 0.10

  # a model you trained yourself
  mjpython run_policy_on_arm.py --task lift --model /path/to/my_new_model.zip --cube 0.22 0.05

  add --sim to force the simulation even if an arm is connected.

How it works
  Every step the policy gets an observation, outputs a small move (dx, dy, dz) and a gripper
  command, the IK turns the move into joint angles, and armfile sends them to the arm.

  The arm reports nothing back, so a StateEstimator stands in for the sensors:
    * gripper position  = forward kinematics of the commanded joint angles
    * finger closure    = what the gripper command should have produced
    * cube position     = the x y you typed, then following the gripper once it is gripped
  There is no camera, so you must place the cube where you said. When a camera is added, replace
  StateEstimator.cube with the detected position and nothing else changes.

Using a new model
  Train it with simulation/train_sac_her.py or simulation/imitation_pick.py, then pass --model.
  The model must use the observation layout in pick_env.build_observation. The runner checks
  this and refuses a model with a different layout. If you change that function, retrain.

Safety: asks before moving, stays inside a small region, limits each step, stops if a joint
would jump, and Ctrl+C stops it. Keep a hand near the power switch. Nothing here has been run
on the real arm yet.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import armfile
import roarm_kinematics as K
from armfile import GRIPPER_CLOSED, GRIPPER_OPEN, RoArm3Controller

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "simulation"))
from pick_env import (ACTION_SCALE, CUBE_HALF, START_POINT, WORKSPACE_HIGH,  # noqa: E402
                      WORKSPACE_LOW, RoArmPickEnv, build_observation)

DEFAULT_MODELS = {"reach": "reach_sac_her.zip", "lift": "lift_imitation.zip", "pick": "lift_imitation.zip"}
CUBE_PARK = np.array([0.28, 0.15, CUBE_HALF])   # where the reach task parks the cube; reach policies ignore it
PITCH = math.pi / 2                             # gripper points straight down, as in training
GRASP_Z = 0.021                                 # pinch height at which the cube sits between the fingers
CUBE_BELOW_PINCH_TABLE = 0.0098                 # measured in simulation: gripped cube centre sits this far below the pinch point...
CUBE_BELOW_PINCH_AIR = 0.0177                   # ...and this far once lifted (it slides down a little)
GRIPPED_CLOSURE = 0.72                          # measured finger closure when blocked by the cube
STEP_SECONDS = 0.04                             # time per step during training; sets the velocity scale


class StateEstimator:
    """Stands in for the sensors the real arm does not have."""

    def __init__(self, task, goal, cube_xy=None):
        self.task, self.goal = task, np.asarray(goal, dtype=float)
        self.cube = np.array([cube_xy[0], cube_xy[1], CUBE_HALF]) if cube_xy is not None else CUBE_PARK.copy()
        self.closure, self.attached = 0.0, False
        self.prev_pinch, self.prev_cube = None, self.cube.copy()

    def observation(self, pinch):
        prev = pinch if self.prev_pinch is None else self.prev_pinch
        return build_observation(pinch, (pinch - prev) / STEP_SECONDS, self.closure, self.cube,
                                 (self.cube - self.prev_cube) / STEP_SECONDS, self.goal, self.task)

    def update(self, action, pinch):
        self.prev_pinch, self.prev_cube = pinch.copy(), self.cube.copy()
        if self.task == "reach":
            return
        command = float(np.clip((action[3] + 1.0) / 2.0, 0.0, 1.0))   # 0 = open, 1 = closed, like the simulator
        in_zone = np.linalg.norm(pinch[:2] - self.cube[:2]) < 0.02 and abs(pinch[2] - GRASP_Z) < 0.02
        limit = GRIPPED_CLOSURE if (in_zone or self.attached) else 1.0   # the cube blocks the fingers
        self.closure += float(np.clip(min(command, limit) - self.closure, -0.3, 0.3))
        if self.closure >= 0.5 and (in_zone or self.attached):
            self.attached = True
        if command < 0.5 and self.attached:                   # released: the cube drops to the table
            self.attached = False
            self.cube = np.array([pinch[0], pinch[1], CUBE_HALF])
        if self.attached:
            below = CUBE_BELOW_PINCH_AIR if pinch[2] > 0.035 else CUBE_BELOW_PINCH_TABLE
            self.cube = pinch + np.array([0.0, 0.0, -below])

    def achieved(self, pinch):
        return pinch if self.task == "reach" else self.cube


def load_policy(path, task):
    from stable_baselines3 import SAC
    env = RoArmPickEnv(task=task)             # needed by Stable-Baselines3 to load saved HER models
    model = SAC.load(path, env=env, device="cpu")
    expected = build_observation(np.zeros(3), np.zeros(3), 0.0, np.zeros(3), np.zeros(3), np.zeros(3), task)
    for key, value in expected.items():
        got = model.observation_space[key].shape
        if got != value.shape:
            sys.exit(f"This model expects '{key}' of shape {got} but pick_env.build_observation now "
                     f"produces {value.shape}. It was trained with a different observation layout; retrain it.")
    if model.action_space.shape != (4,):
        sys.exit(f"This model outputs {model.action_space.shape} actions; the runner needs 4 (dx, dy, dz, gripper).")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach", "lift", "pick"], default="reach")
    ap.add_argument("--model", default=None, help="path to a trained .zip (default depends on --task)")
    ap.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--cube", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="where you placed the cube (lift and pick tasks)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--hz", type=float, default=4.0, help="policy steps per second (slow is safer)")
    ap.add_argument("--min-z", type=float, default=None, help="lowest allowed gripper height (m)")
    ap.add_argument("--max-jump", type=float, default=0.25, help="abort if a joint would jump more than this (rad)")
    ap.add_argument("--grip-closed", type=float, default=GRIPPER_CLOSED, help="gripper angle sent when closing")
    ap.add_argument("--sim", action="store_true", help="force the simulation even if an arm is connected")
    args = ap.parse_args()

    if args.task != "reach" and args.cube is None:
        sys.exit("Tell it where you put the cube, e.g.  --cube 0.22 0.05  (metres, 10 to 29 cm in front of the base)")
    goal = args.goal if args.goal is not None else ([0.26, 0.05, 0.08] if args.task == "reach" else [0.22, 0.0, 0.10])
    steps = args.steps or (60 if args.task == "reach" else 100)
    model_path = args.model or str(ROOT / "simulation" / "models" / DEFAULT_MODELS[args.task])
    model = load_policy(model_path, args.task)

    low = WORKSPACE_LOW.copy()
    low[2] = args.min_z if args.min_z is not None else (0.06 if args.task == "reach" else 0.02)
    goal = np.clip(np.array(goal), low, WORKSPACE_HIGH)
    if not np.allclose(goal, args.goal if args.goal is not None else goal):
        print(f"Goal moved into the safe region: {np.round(goal, 3)}")

    arm = RoArm3Controller(port=None if args.sim else armfile.PORT)
    arm.sim_move_seconds = 1.0 / args.hz
    real = not arm.simulated
    print(f"\n=== {'REAL ARM' if real else 'SIMULATION'} ===  task {args.task}, model {Path(model_path).name}")
    print(f"goal {np.round(goal, 3)}" + (f", cube at {args.cube}" if args.cube else "") + f", {steps} steps at {args.hz:g}/s")

    start = np.clip(START_POINT, low, WORKSPACE_HIGH)
    if real:
        if args.task != "reach":
            print(f"The gripper will go down to z = {low[2]:.2f} m, measured from the table plane under the base.")
            if input("Is the real table at that level, and is the cube exactly where you said? [y/N] ").strip().lower() != "y":
                print("Stopped.")
                return
        print("The code cannot read the arm's position, so it assumes the arm is at HOME (all joints zero).")
        if input("Is the arm at HOME and the area clear? [y/N] ").strip().lower() != "y":
            print("Stopped.")
            return
    print(f"Moving to the start point {np.round(start, 3)} ...")
    if not arm.move_linear_path(*start, steps=15, wrist_pitch=PITCH):
        print("Could not reach the start point. Stopped.")
        return
    if real and input("Run the policy now? [y/N] ").strip().lower() != "y":
        print("Stopped at the start point.")
        return

    est = StateEstimator(args.task, goal, args.cube)
    pinch, _ = K.forward_kinematics(arm._last_angles[:5])
    tool_target, hits = pinch.copy(), 0
    try:
        for step in range(1, steps + 1):
            t0 = time.perf_counter()
            action, _ = model.predict(est.observation(pinch), deterministic=True)
            tool_target = np.clip(tool_target + action[:3] * ACTION_SCALE, low, WORKSPACE_HIGH)
            sol = K.inverse_kinematics(*tool_target, pitch=PITCH, q_current=arm._last_angles[:4])
            if sol is None:
                print(f"Step {step}: no IK solution for {np.round(tool_target, 3)}. Stopped.")
                return
            jump = float(np.max(np.abs(np.array(sol[:4]) - np.array(arm._last_angles[:4]))))
            if jump > args.max_jump:
                print(f"Step {step}: a joint would jump {jump:.2f} rad. Stopped for safety.")
                return
            command = float(np.clip((action[3] + 1.0) / 2.0, 0.0, 1.0))
            arm.send_arm_command(*sol, GRIPPER_OPEN + command * (args.grip_closed - GRIPPER_OPEN))
            pinch, _ = K.forward_kinematics(arm._last_angles[:5])
            est.update(action, pinch)
            dist = float(np.linalg.norm(est.achieved(pinch) - goal))
            extra = "" if args.task == "reach" else f"  cube {np.round(est.cube, 3)}{' (held)' if est.attached else ''}"
            print(f"step {step:3d}  gripper {np.round(pinch, 3)}{extra}  distance to goal {dist * 100:.1f} cm")
            hits = hits + 1 if dist < 0.02 else 0
            if hits >= 3:
                print("Reached the goal.")
                return
            time.sleep(max(0.0, 1.0 / args.hz - (time.perf_counter() - t0)))
        print("Step limit reached.")
    except KeyboardInterrupt:
        print("\nStopped by you.")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
