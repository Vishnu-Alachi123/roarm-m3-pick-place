#!/usr/bin/env python3
"""
Train the RoArm-M3 in MuJoCo with SAC + HER (Stable-Baselines3).

  python3 train_sac_her.py --task reach --steps 60000     # minutes: learns to move the gripper to a goal
  python3 train_sac_her.py --task pick  --steps 1500000   # hours: learns to grasp and place the cube
  python3 train_sac_her.py --task lift  --steps 500000 --envs 8 --grad-steps 2   # goals only in the air

Output: models/<task>_sac_her.zip (the trained policy) and runs/<task>/ (TensorBoard logs).
Watch learning:  tensorboard --logdir runs
"""
import argparse
from pathlib import Path

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pick_env import RoArmPickEnv

HERE = Path(__file__).resolve().parent


def make_env(task):
    return Monitor(RoArmPickEnv(task=task), info_keywords=("is_success",))


def make_train_env(task, n_envs):
    if n_envs == 1:
        return make_env(task)
    return SubprocVecEnv([lambda: make_env(task) for _ in range(n_envs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach", "pick", "lift"], default="reach")
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envs", type=int, default=1, help="parallel simulators (faster)")
    ap.add_argument("--grad-steps", type=int, default=1, help="network updates per collection step (fewer = faster)")
    args = ap.parse_args()

    models = HERE / "models"
    models.mkdir(exist_ok=True)
    env = make_train_env(args.task, args.envs)
    eval_env = DummyVecEnv([lambda: make_env(args.task)]) if args.envs > 1 else make_env(args.task)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        learning_rate=1e-3,
        buffer_size=500_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        learning_starts=1_000,
        train_freq=1,
        gradient_steps=args.grad_steps,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        tensorboard_log=str(HERE / "runs" / args.task),
        seed=args.seed,
        verbose=1,
    )
    eval_cb = EvalCallback(eval_env, eval_freq=5_000, n_eval_episodes=20, deterministic=True,
                           best_model_save_path=str(models / f"{args.task}_best"), verbose=1)
    model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=False)
    out = models / f"{args.task}_sac_her"
    model.save(str(out))
    print(f"saved {out}.zip")


if __name__ == "__main__":
    main()
