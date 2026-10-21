#!/usr/bin/env python3
"""
Teach the policy to pick and lift the cube by imitation (behaviour cloning + DAgger).

Reinforcement learning from scratch never discovered lifting, so a rule-based expert
does the task and the network learns to copy it:

  1. The expert (a function of the observation only) picks the cube in simulation.
  2. The network is trained to output the same action as the expert for the same observation.
  3. DAgger: the network drives, the expert labels every state it visits, the new
     examples are added to the data, and the network is retrained. This fixes the
     mistakes a copied policy makes when it drifts away from the expert's path.

  ../.venv/bin/python imitation_pick.py --test-expert     # does the expert itself succeed?
  ../.venv/bin/python imitation_pick.py                   # train, writes models/lift_imitation.zip
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC, HerReplayBuffer

from pick_env import ACTION_SCALE, RoArmPickEnv

HERE = Path(__file__).resolve().parent
GRASP_Z = 0.021          # pinch height that puts the cube between the fingers
HOVER_Z = 0.10
CARRY_OFFSET = 0.009     # pinch sits this far above the cube centre while carrying


def expert_action(obs, goal):
    """Rule-based pick and lift. Uses only quantities the policy also observes."""
    pinch, closure, cube = obs[0:3], obs[6], obs[7:10]
    grip = -1.0
    if closure < 0.2:                                    # fingers open
        dxy = cube[:2] - pinch[:2]
        if np.linalg.norm(dxy) > 0.03 and pinch[2] < 0.07:
            target = np.array([pinch[0], pinch[1], HOVER_Z])     # rise before moving sideways
        elif np.linalg.norm(dxy) > 0.010:
            target = np.array([cube[0], cube[1], HOVER_Z])       # move over the cube
        else:
            target = np.array([cube[0], cube[1], GRASP_Z])       # descend
            if pinch[2] < GRASP_Z + 0.006:
                grip = 1.0                                        # close
    else:                                                # closing or closed
        grip = 1.0
        target = pinch if closure < 0.5 else goal + np.array([0.0, 0.0, CARRY_OFFSET])
    move = np.clip((target - pinch) / ACTION_SCALE, -1.0, 1.0)
    return np.concatenate([move, [grip]]).astype(np.float32)


def run_episode(env, seed, policy=None, beta=1.0, noise=0.0, rng=None):
    obs, _ = env.reset(seed=seed)
    data, max_cube_z = [], 0.0
    for _ in range(env.max_steps):
        flat, goal = obs["observation"], obs["desired_goal"]
        a_exp = expert_action(flat, goal)
        data.append(({k: v.copy() for k, v in obs.items()}, a_exp))
        if policy is not None and (beta <= 0 or rng.random() >= beta):
            a, _ = policy.predict(obs, deterministic=True)
        else:
            a = a_exp.copy()
        if noise > 0:
            a[:3] = np.clip(a[:3] + rng.normal(0, noise, 3), -1, 1)
        obs, _, _, truncated, info = env.step(a)
        max_cube_z = max(max_cube_z, float(obs["achieved_goal"][2]))
        if truncated:
            break
    final = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
    return data, final < 0.02, max_cube_z > 0.04


def evaluate(env, policy, n, seed0):
    wins = lifts = 0
    for i in range(n):
        _, ok, lifted = run_episode(env, seed0 + i, policy=policy, beta=0.0)
        wins += ok
        lifts += lifted
    return wins, lifts


def stack(samples):
    keys = samples[0][0].keys()
    obs = {k: torch.as_tensor(np.stack([s[0][k] for s in samples])) for k in keys}
    act = torch.as_tensor(np.stack([s[1] for s in samples]))
    return obs, act


def fit(model, samples, epochs, lr=1e-3, batch=512):
    obs, act = stack(samples)
    n = act.shape[0]
    opt = torch.optim.Adam(model.policy.actor.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = model.policy.actor({k: v[idx] for k, v in obs.items()}, deterministic=True)
            loss = F.mse_loss(pred, act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-expert", action="store_true")
    ap.add_argument("--task", default="lift", choices=["lift", "pick"])
    ap.add_argument("--demos", type=int, default=150)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--per-iter", type=int, default=40)
    args = ap.parse_args()

    env = RoArmPickEnv(task=args.task)
    rng = np.random.default_rng(0)

    if args.test_expert:
        res = [run_episode(env, 100 + i)[1:] for i in range(50)]
        print(f"expert alone: reached goal {sum(r[0] for r in res)}/50, lifted {sum(r[1] for r in res)}/50")
        return

    model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
                policy_kwargs=dict(net_arch=[256, 256, 256]), device="cpu", verbose=0)

    samples = []
    ok_count = 0
    for i in range(args.demos):
        d, ok, _ = run_episode(env, 1000 + i, noise=0.3 if i % 2 else 0.0, rng=rng)
        samples += d
        ok_count += ok
    print(f"collected {len(samples)} expert steps from {args.demos} episodes ({ok_count} successful)")
    loss = fit(model, samples, epochs=40)
    wins, lifts = evaluate(env, model, 30, 9000)
    print(f"round 0 | loss {loss:.4f} | reached goal {wins}/30 | lifted {lifts}/30", flush=True)

    best = (wins, lifts)
    model.save(str(HERE / "models" / "lift_imitation"))
    for it in range(1, args.iterations + 1):
        for j in range(args.per_iter):
            d, _, _ = run_episode(env, 20000 + it * 1000 + j, policy=model, beta=0.3, noise=0.1, rng=rng)
            samples += d
        loss = fit(model, samples, epochs=25)
        wins, lifts = evaluate(env, model, 30, 9000)
        print(f"round {it} | {len(samples)} samples | loss {loss:.4f} | reached goal {wins}/30 | lifted {lifts}/30", flush=True)
        if (wins, lifts) >= best:
            best = (wins, lifts)
            model.save(str(HERE / "models" / "lift_imitation"))
    print(f"saved best model: reached goal {best[0]}/30, lifted {best[1]}/30 -> models/lift_imitation.zip")


if __name__ == "__main__":
    main()
