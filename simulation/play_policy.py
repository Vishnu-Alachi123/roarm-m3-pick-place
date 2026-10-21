#!/usr/bin/env python3
"""
Run a trained policy and watch it in the MuJoCo viewer (macOS needs mjpython).

  mjpython play_policy.py --task reach
  mjpython play_policy.py --task lift      # the imitation model: picks the cube up and lifts it
  mjpython play_policy.py --task pick --model models/some_other_model.zip
  mjpython play_policy.py --task reach --follow     # drag the green target; the policy chases it

In the viewer: double-click the green sphere (or the cube) to select it, then hold
Ctrl and drag with the right mouse button to move it.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from stable_baselines3 import SAC

from pick_env import RoArmPickEnv

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach", "pick", "lift"], default="reach")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--follow", action="store_true", help="chase the green target as you drag it")
    args = ap.parse_args()

    default = "reach_sac_her.zip" if args.task == "reach" else "lift_imitation.zip"
    path = args.model or str(HERE / "models" / default)
    env = RoArmPickEnv(task=args.task, render_mode="human")
    model = SAC.load(path, env=env, device="cpu")   # HER models need the env passed in
    if args.follow:
        obs, _ = env.reset()
        while env._viewer is not None and env._viewer.is_running():
            env.goal = np.clip(env.data.mocap_pos[0].copy(), [0.12, -0.18, 0.03], [0.32, 0.18, 0.20])
            obs = env._obs()
            action, _ = model.predict(obs, deterministic=True)
            env.step(action)
            env._steps = 0                       # never time out while following
            time.sleep(0.04)
        env.close()
        return
    wins = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.04)
            if truncated or terminated:
                break
        wins += int(info["is_success"])
        print(f"episode {ep + 1}: {'success' if info['is_success'] else 'miss'}")
    print(f"{wins}/{args.episodes} successful")
    env.close()


if __name__ == "__main__":
    main()
