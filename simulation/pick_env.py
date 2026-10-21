#!/usr/bin/env python3
"""
RoArmPickEnv: a Gymnasium goal-conditioned environment for the RoArm-M3 in MuJoCo.

Tasks
  reach  move the gripper's pinch point to a random 3D goal (easy; trains in minutes)
  pick   grasp the cube and bring it to a random goal on or above the table (hard)

Action  (4 floats in [-1, 1])
  [dx, dy, dz, gripper]
  dx/dy/dz  Cartesian motion of the pinch point this step, up to ACTION_SCALE metres.
            A differential-IK controller turns it into joint targets and keeps the
            gripper pointing straight down.
  gripper   -1 = open ... +1 = closed

Observation (Dict, the format Hugging Face / Stable-Baselines3 HER expects)
  observation    16 floats: pinch pos (3), pinch velocity (3), finger opening (1),
                 cube pos (3), cube relative to pinch (3), cube velocity (3)
  achieved_goal  3 floats: cube position (pick) or pinch position (reach)
  desired_goal   3 floats: where it should be

Reward: sparse, -1 each step until the goal is within SUCCESS_RADIUS, then 0.
"""
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent

ARM_JOINTS = ["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"]
ARM_ACT = ["base_act", "shoulder_act", "elbow_act", "wrist_act", "roll_act"]

ACTION_SCALE = 0.015           # metres of pinch-point motion per step at full action
FRAME_SKIP = 20                # 20 x 2 ms = 0.04 s per environment step
SUCCESS_RADIUS = 0.02          # metres
FINGER_TRAVEL = 0.025
WORKSPACE_LOW = np.array([0.10, -0.18, 0.020])
WORKSPACE_HIGH = np.array([0.34, 0.18, 0.20])
CUBE_HALF = 0.0125
START_POINT = np.array([0.22, 0.0, 0.12])


def build_observation(pinch, vel, closure, cube, cube_vel, goal, task):
    """The one place the observation layout is defined. The simulator and the real-arm
    runner both call this, so a model trained here gets identical inputs on the arm.
    closure: 0 = fingers open, 1 = fully closed."""
    pinch, vel, cube, cube_vel = (np.asarray(v, dtype=np.float64) for v in (pinch, vel, cube, cube_vel))
    obs = np.concatenate([pinch, vel, [closure], cube, cube - pinch, cube_vel]).astype(np.float32)
    achieved = cube if task in ("pick", "lift") else pinch
    return {"observation": obs,
            "achieved_goal": np.asarray(achieved, dtype=np.float32),
            "desired_goal": np.asarray(goal, dtype=np.float32)}


class RoArmPickEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, task: str = "reach", render_mode: Optional[str] = None,
                 max_steps: int = 100, reward_type: str = "sparse"):
        assert task in ("reach", "pick", "lift")
        assert reward_type in ("sparse", "dense")
        self.task, self.render_mode = task, render_mode
        self.max_steps, self.reward_type = max_steps, reward_type

        self.model = mujoco.MjModel.from_xml_path(str(HERE / "roarm_m3_pick.xml"))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.act_ids = [m.actuator(n).id for n in ARM_ACT]
        self.fl_act, self.fr_act = m.actuator("finger_l_act").id, m.actuator("finger_r_act").id
        self.qadr = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.dofadr = [m.joint(j).dofadr[0] for j in ARM_JOINTS]
        self.fl_q = m.joint("finger_l_slide").qposadr[0]
        self.fr_q = m.joint("finger_r_slide").qposadr[0]
        self.cube_q = m.joint("cube_free").qposadr[0]
        self.cube_v = m.joint("cube_free").dofadr[0]
        self.pinch = m.site("pinch").id
        self.limits = np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        big = np.full(3, np.inf, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "achieved_goal": spaces.Box(-big, big, dtype=np.float32),
            "desired_goal": spaces.Box(-big, big, dtype=np.float32),
        })

        self._viewer = None
        self._renderer = None
        self._steps = 0
        self.goal = START_POINT.copy()
        self.tool_target = START_POINT.copy()
        self.q_cmd = np.zeros(5)
        self.grip_cmd = 0.0
        self._prev_pinch = START_POINT.copy()
        self._q_start = self._solve_start_pose()

    # ---- kinematics helpers -------------------------------------------------
    def _jacobians(self):
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.pinch)
        approach = self.data.site_xmat[self.pinch].reshape(3, 3)[:, 2]
        Jp = jp[:, self.dofadr]
        Ja = np.cross(jr[:, self.dofadr].T, approach).T      # d(approach)/dq
        return Jp, Ja, approach

    def _dls_step(self, target: np.ndarray) -> np.ndarray:
        Jp, Ja, approach = self._jacobians()
        e_pos = target - self.data.site_xpos[self.pinch]
        e_dir = np.array([0.0, 0.0, -1.0]) - approach        # keep the gripper pointing down
        J = np.vstack([Jp, 0.4 * Ja])
        e = np.concatenate([e_pos, 0.4 * e_dir])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        dq[4] = 0.0                                          # roll does not move the pinch point
        return np.clip(dq, -0.25, 0.25)

    def _solve_start_pose(self):
        d = self.data
        d.qpos[:] = 0
        d.qpos[self.qadr] = [0.0, 0.5, 1.4, -0.3, 0.0]
        for _ in range(300):
            mujoco.mj_forward(self.model, d)
            q = np.array([d.qpos[a] for a in self.qadr]) + self._dls_step(START_POINT)
            q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
            d.qpos[self.qadr] = q
        return q

    # ---- gym API ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = self._q_start
        self.q_cmd = self._q_start.copy()
        self.grip_cmd = 0.0
        self._write_ctrl()

        if self.task in ("pick", "lift"):
            cx = rng.uniform(0.16, 0.28)
            cy = rng.uniform(-0.10, 0.10)
            d.qpos[self.cube_q:self.cube_q + 7] = [cx, cy, CUBE_HALF, 1, 0, 0, 0]
            if self.task == "lift" or rng.random() >= 0.5:
                gz = rng.uniform(0.05, 0.12)          # in the air: pushing cannot reach it
            else:
                gz = CUBE_HALF
            self.goal = np.array([rng.uniform(0.16, 0.28), rng.uniform(-0.10, 0.10), gz])
        else:
            d.qpos[self.cube_q:self.cube_q + 7] = [0.28, 0.15, CUBE_HALF, 1, 0, 0, 0]
            self.goal = rng.uniform([0.14, -0.12, 0.04], [0.30, 0.12, 0.16])

        for _ in range(150):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        self.tool_target = d.site_xpos[self.pinch].copy()
        self._prev_pinch = self.tool_target.copy()
        self._steps = 0
        d.mocap_pos[0] = self.goal
        self._sync_viewer()
        return self._obs(), {"is_success": False}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        m, d = self.model, self.data
        self.tool_target = np.clip(self.tool_target + action[:3] * ACTION_SCALE, WORKSPACE_LOW, WORKSPACE_HIGH)
        self.grip_cmd = (action[3] + 1.0) / 2.0 * FINGER_TRAVEL

        self._prev_pinch = d.site_xpos[self.pinch].copy()
        q_now = np.array([d.qpos[a] for a in self.qadr])
        self.q_cmd = np.clip(q_now + self._dls_step(self.tool_target),
                             self.limits[:, 0], self.limits[:, 1])
        self._write_ctrl()
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        self._steps += 1
        self._sync_viewer()

        obs = self._obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))
        success = bool(self._distance(obs["achieved_goal"], obs["desired_goal"]) < SUCCESS_RADIUS)
        truncated = self._steps >= self.max_steps
        return obs, reward, False, truncated, {"is_success": success}

    # ---- goal-env interface (HER calls compute_reward with batches) -----------
    @staticmethod
    def _distance(a, b):
        return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = self._distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            reward = -(dist > SUCCESS_RADIUS).astype(np.float32)
        else:
            reward = -dist.astype(np.float32)
        if self.task == "lift":
            # small bonus for raising the cube; depends only on achieved_goal so HER relabelling stays valid
            height = np.clip((np.asarray(achieved_goal)[..., 2] - CUBE_HALF) / 0.04, 0.0, 1.0)
            reward = reward + 0.5 * height.astype(np.float32)
        return reward

    # ---- internals ----------------------------------------------------------
    def _write_ctrl(self):
        for a, v in zip(self.act_ids, self.q_cmd):
            self.data.ctrl[a] = v
        self.data.ctrl[self.fl_act] = self.grip_cmd
        self.data.ctrl[self.fr_act] = self.grip_cmd

    def _obs(self):
        d = self.data
        pinch = d.site_xpos[self.pinch].copy()
        vel = (pinch - self._prev_pinch) / (FRAME_SKIP * self.model.opt.timestep)
        cube = d.qpos[self.cube_q:self.cube_q + 3].copy()
        cube_vel = d.qvel[self.cube_v:self.cube_v + 3].copy()
        opening = (d.qpos[self.fl_q] + d.qpos[self.fr_q]) / (2 * FINGER_TRAVEL)
        return build_observation(pinch, vel, opening, cube, cube_vel, self.goal, self.task)

    def _sync_viewer(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
