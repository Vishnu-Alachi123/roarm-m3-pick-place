# RoArm-M3 Autonomous Pick-and-Place

Reinforcement-learning pipeline for the **Waveshare RoArm-M3** 5-DOF desktop robot arm.  
Trains a pick-and-place policy in MuJoCo simulation, then deploys it to the real arm.

---

## Resource Links

| Resource | Link |
|----------|------|
| **RoArm-M3 product page** | https://www.waveshare.com/roarm-m3.htm |
| **RoArm-M3 official GitHub** | https://github.com/waveshare/RoArm-M3 |
| **RoArm-M3 wiki / docs** | https://www.waveshare.com/wiki/RoArm-M3 |
| **MuJoCo physics engine** | https://github.com/google-deepmind/mujoco |
| **MuJoCo docs** | https://mujoco.readthedocs.io/ |
| **Stable-Baselines3 (SAC + HER)** | https://stable-baselines3.readthedocs.io/ |
| **YOLOv8 (object detection)** | https://docs.ultralytics.com/ |
| **CadQuery (gripper CAD)** | https://cadquery.readthedocs.io/ |

---

## Project Status

| Area | Status |
|------|--------|
| MuJoCo simulation environment | ✅ Complete |
| SAC+HER training (reach / lift / pick) | ✅ Trained models saved |
| Simulation demo — policy viewer | ✅ Runnable |
| Simulation demo — interactive follow mode | ✅ Runnable |
| Custom TPU gripper attachment (STLs) | ✅ Ready to print |
| Run trained policy on real arm | 🔧 Code written — needs gripper printed + calibration |
| Intel RealSense depth-camera integration | 🔲 Not started |
| CV pipeline (YOLO + depth → XYZ) | 🔲 Stub exists (`cv_detector.py`) |

---

## Quick Demo

### One-command launcher

```bash
cd /path/to/roarm_docker
./demo.sh sim          # watch pick policy in simulation
./demo.sh follow       # drag sphere → arm follows interactively (sim)
./demo.sh arm          # run policy on real arm (requires hardware setup below)
```

### Manual — Simulation demo

**Requirements:** Python 3.9+, `pip install mujoco stable-baselines3 gymnasium`

```bash
cd simulation

# macOS: must use mjpython (bundled with the mujoco pip package)
mjpython play_policy.py --task pick \
    --model models/pick_best/best_model.zip \
    --episodes 5

# Linux / headless:
python play_policy.py --task pick \
    --model models/pick_best/best_model.zip \
    --episodes 5
```

The arm will reach for the red cube and lift it. Console prints success/miss counts.

### Manual — Interactive follow mode (simulation)

```bash
cd simulation
mjpython play_policy.py --task reach --follow
```

Drag the **green sphere** in the MuJoCo viewer — the arm chases it live.  
Press **Escape** or close the viewer to exit.

### Manual — Policy on real arm

See **Hardware Setup** below for prerequisites, then:

```bash
# Reach to a fixed point (simplest test, no camera needed)
python workspace/run_policy_on_arm.py --task reach --goal 0.25 0.0 0.10

# Pick a cube placed at (0.25 m, 0.05 m) on the table
python workspace/run_policy_on_arm.py --task pick \
    --model simulation/models/pick_best/best_model.zip \
    --goal 0.25 0.05 0.12 \
    --cube 0.25 0.05
```

> ⚠️ The script asks for confirmation **twice** before the arm moves.  
> Always clear the workspace and have a finger on the power switch.

---

## Hardware Setup (physical arm)

Complete these steps **in order** before running `run_policy_on_arm.py`.

### 1 · Print and fit the gripper

| Part | File | Material | Slicer settings |
|------|------|----------|-----------------|
| Flexible claw (moving jaw) | `simulation/roarm_claw_tpu.stl` | TPU 95A | 0 % infill · 2–3 walls · print upright |
| Rigid pad (fixed jaw) | `simulation/roarm_fixed_pad_pla.stl` | PLA | 20 % infill · standard |

Clip each part onto the arm's jaws. Bond a thin rubber/silicone sheet to the flat face of the PLA pad for grip.  
Regenerate STLs after caliperting the real finger widths:

```bash
# Edit FINGER_W / FINGER_D in simulation/roarm_gripper_attachment.py, then:
python simulation/roarm_gripper_attachment.py
```

### 2 · Connect the arm

```bash
# Plug the arm in via USB, then:
docker-compose up -d          # start ROS 2 container
./bridge-arm-serial.sh        # bridge USB serial into the container
```

Verify the arm is visible:

```bash
docker exec roarm ros2 topic list | grep joint
```

### 3 · Calibrate the workspace

```bash
python workspace/arm_calibration.py
```

This maps simulation coordinates (metres) to real joint angles. Run once per setup.  
The calibration is saved to `workspace/calibration.json`.

### 4 · Run the demo

```bash
python workspace/run_policy_on_arm.py --task reach --goal 0.25 0.0 0.10
```

If reach succeeds (arm smoothly moves to the target XYZ), proceed to the pick task.

---

## Repository Layout

```
roarm_docker/
├── demo.sh                     ← one-command demo launcher
├── docker-compose.yml          ← ROS 2 container
├── bridge-arm-serial.sh        ← USB serial bridge: Docker ↔ arm
│
├── simulation/
│   ├── pick_env.py             ← Gymnasium env (single source of truth for obs/reward)
│   ├── train_sac_her.py        ← SAC+HER training
│   ├── play_policy.py          ← run / visualise a saved policy; --follow for interactive
│   ├── sim.py                  ← MuJoCo wrapper (mirrors real-arm interface)
│   ├── roarm_m3_pick.xml       ← MuJoCo scene for training
│   ├── roarm_m3_scene.xml      ← MuJoCo scene for demo
│   ├── roarm_m3.urdf           ← robot description
│   ├── meshes/                 ← link STL meshes
│   ├── models/                 ← saved trained policies (zip)
│   │   ├── reach_best/
│   │   ├── pick_best/
│   │   └── lift_best/
│   ├── runs/                   ← TensorBoard event files
│   └── roarm_gripper_attachment.py  ← CadQuery gripper CAD → STL
│
├── workspace/
│   ├── arm_controller.py       ← ROS 2 node: move_to / get_position / set_gripper
│   ├── run_policy_on_arm.py    ← deploy trained policy to real arm
│   ├── arm_calibration.py      ← workspace calibration helpers
│   ├── cv_detector.py          ← YOLOv8 + pixel→arm-coord (stub, needs depth cam)
│   ├── roarm_kinematics.py     ← forward / inverse kinematics
│   ├── demo_move_to_xyz.py     ← minimal XYZ movement demo
│   └── arm_live_control.py     ← live keyboard/joystick control
│
└── simulation/                 ← STL outputs (also used as print files)
    ├── roarm_claw_tpu.stl
    ├── roarm_fixed_pad_pla.stl
    └── roarm_gripper_assembly.stl
```

---

## Architecture

```
Training (sim, offline)               Deploy (real arm)
──────────────────────                ──────────────────────────────────────
pick_env.py                           cv_detector.py (Phase 2)
  build_observation() ◄───────────────────────────── run_policy_on_arm.py
    [ee_pos  (3)                        same obs layout,        │
     ee_vel  (3)                        StateEstimator          │ SAC policy
     gripper (1)                        fills missing           │
     obj_pos (3)                        real-arm sensors        ▼
     obj_vel (3)                                         arm_controller.py
     goal    (3)]                                          ROS 2 services
  ▼
train_sac_her.py
  SAC + HER → models/<task>_best/best_model.zip
```

**Key invariant:** `build_observation()` in `pick_env.py` is the shared contract between
training and deployment. If you change the observation layout you must update
`StateEstimator` in `run_policy_on_arm.py` and retrain.

---

## Training a Policy

```bash
cd simulation

# Reach task (~5 min on CPU — good first run)
python train_sac_her.py --task reach --steps 60000

# Pick task (~30–60 min)
python train_sac_her.py --task pick --steps 200000 --envs 4

# Monitor in TensorBoard
tensorboard --logdir runs/
```

Checkpoints are saved every 5,000 steps to `models/<task>_best/`.

---

## Next Steps

### Immediate
- [ ] **Print gripper** — STLs in `simulation/`, need TPU 95A + PLA printer
- [ ] **First hardware run** — attach gripper → `arm_calibration.py` → `run_policy_on_arm.py --task reach`
- [ ] **Validate pick** — once reach is confirmed, test the pick task

### Phase 2 — Depth Camera
- [ ] Mount Intel RealSense (end-effector vs. fixed overhead — TBD by test)
- [ ] Extend `cv_detector.py` to use depth frames (replace top-down pixel mapping with real XYZ)
- [ ] Wire `cv_detector.py` output → `run_policy_on_arm.py` goal input

### Phase 3 — Closed-Loop
- [ ] Feed real object position into observation (replace `StateEstimator`)
- [ ] Retrain with domain randomisation + real-arm rollout data

---

## Tech Stack

| Component | Library / Tool | Version |
|-----------|----------------|---------|
| Physics simulation | [MuJoCo](https://github.com/google-deepmind/mujoco) | ≥ 3.x |
| RL training | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) | ≥ 2.x |
| Object detection | [YOLOv8](https://docs.ultralytics.com/) (Ultralytics) | ≥ 8.x |
| Robot middleware | ROS 2 (Dockerized) | Humble |
| Gripper CAD | [CadQuery](https://cadquery.readthedocs.io/) | ≥ 2.x |
| Arm communication | Serial via ROS 2 services | — |
| Depth camera (Phase 2) | [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) | — |
