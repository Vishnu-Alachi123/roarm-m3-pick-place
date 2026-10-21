# RoArm3 — Codebase Documentation

This document covers every file in the project: what it does, how it works, and how to use it. There is also a dedicated section explaining the inverse kinematics (IK) math used in `armfile.py`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [armfile.py — Low-Level Arm Control + IK](#2-armfilepy)
   - [Configuration Constants](#21-configuration-constants)
   - [ArmPosition](#22-armposition)
   - [RoArm3Controller](#23-roarm3controller)
   - [Inverse Kinematics — move_to_xyz](#24-inverse-kinematics--move_to_xyz)
   - [POSITIONS Dictionary](#25-positions-dictionary)
   - [PickAndPlaceController](#26-pickandplacecontroller)
   - [main() and the Interactive Menu](#27-main-and-the-interactive-menu)
3. [arm_controller.py — ROS2 High-Level Controller](#3-arm_controllerpy)
4. [arm_calibration.py — Movement Calibration Tool](#4-arm_calibrationpy)
5. [arm_live_control.py — Live Keyboard & Mouse Control](#5-arm_live_controlpy)
6. [cv_detector.py — Webcam Object Detection](#6-cv_detectorpy)
7. [How the Files Connect](#7-how-the-files-connect)

---

## 1. Project Overview

The project controls a **Waveshare RoArm-M3**, a 5+1 DOF (5 joints + gripper) serial robotic arm. There are two separate communication stacks:

| Stack | Files | How it talks to the arm |
|---|---|---|
| **Serial / JSON** | `armfile.py` | USB serial at 115200 baud, JSON commands |
| **ROS2** | `arm_controller.py`, `arm_calibration.py`, `arm_live_control.py` | ROS2 services and topics via MoveIt2 |

The serial stack (`armfile.py`) gives you direct, low-level control over every joint angle. The ROS2 stack talks to the arm through MoveIt2, which handles motion planning and lets you command the arm in Cartesian (x, y, z) space.

`cv_detector.py` is a standalone vision tool that runs on your laptop and detects objects using a webcam — no arm connection needed.

---

## 2. armfile.py

**File:** `armfile.py`  
**Run with:** `python3 armfile.py`  
**Dependencies:** `pyserial` (`pip install pyserial`)

This is the core low-level control file. It talks directly to the arm's ESP32 microcontroller over USB serial using a JSON command protocol. Every joint is controlled by sending a JSON string like:

```json
{"T": 102, "base": 0.0, "shoulder": 0.5, "elbow": 1.2, "wrist": 0.0, "roll": 0.0, "hand": 1.08, "spd": 0, "acc": 10}
```

All joint angles are in **radians**.

---

### 2.1 Configuration Constants

At the top of the file, before any classes:

```python
PORT      = "/dev/ttyUSB0"   # serial port the arm is connected to
BAUDRATE  = 115200            # must match the arm's firmware setting

SPEED_DEFAULT = 0             # 0 = maximum speed
ACC_DEFAULT   = 10            # acceleration (0–50)

GRIPPER_OPEN   = 1.08         # ~62°  — fully open
GRIPPER_CLOSED = 3.14         # ~180° — fully closed
```

**To change the serial port** (e.g. on Windows it would be `"COM3"`), edit `PORT`. On Linux/Mac, run `ls /dev/tty*` before and after plugging in the arm to find the right port.

---

### 2.2 ArmPosition

A simple data class that stores a complete arm pose — all 6 joint angles plus a description string.

```python
pos = ArmPosition(
    base=0.0, shoulder=0.5, elbow=1.2,
    wrist=0.0, roll=0.0, gripper=1.08,
    desc="My custom position"
)
pos.to_list()   # → [0.0, 0.5, 1.2, 0.0, 0.0, 1.08]
```

| Field | Joint | Range |
|---|---|---|
| `base` | Rotates the whole arm left/right | −π to +π (−180° to 180°) |
| `shoulder` | Raises/lowers the upper arm | −π/2 to +π/2 |
| `elbow` | Bends the forearm | 0 to π |
| `wrist` | Pitches the wrist up/down | −π/2 to +π/2 |
| `roll` | Rotates the wrist | −π to +π |
| `gripper` | Opens/closes the gripper | 1.08 (open) to 3.14 (closed) |

---

### 2.3 RoArm3Controller

The main class. Opens the serial connection and provides methods to send commands.

**Constructor:**
```python
arm = RoArm3Controller(port="/dev/ttyUSB0", baudrate=115200)
```
Opens the serial port and waits 0.5 seconds for the arm to be ready.

**Methods:**

#### `send_command(cmd_dict)`
The lowest-level method. Serialises a Python dict to JSON and writes it to the serial port with a newline. All other methods call this internally.

#### `send_arm_command(base, shoulder, elbow, wrist, roll, hand, speed, acc)`
Sends a **Type 102** command — the full arm pose in one message. All 6 joints move simultaneously to the given angles.

```python
arm.send_arm_command(0.0, 0.5, 1.2, 0.0, 0.0, 1.08)
```

#### `send_gripper_command(angle_rad, speed, acc)`
Sends a **Type 106** command — controls only the gripper independently, without moving any other joint.

```python
arm.send_gripper_command(1.08)   # open
arm.send_gripper_command(3.14)   # closed
```

#### `move_to_position(position, speed, acc)`
Takes an `ArmPosition` object and sends it as a Type 102 command. Prints the joint angles to the console before moving.

```python
arm.move_to_position(POSITIONS["home"])
```

#### `set_gripper(angle_rad, speed, acc)`
Convenience wrapper around `send_gripper_command`. Prints whether the gripper is OPEN, CLOSED, or INTERMEDIATE.

#### `move_to_xyz(x, y, z, wrist_pitch, wrist_roll, gripper, speed, acc)`
The inverse kinematics method. Takes a target position in metres and computes the joint angles needed to put the arm tip there. See the full explanation in [Section 2.4](#24-inverse-kinematics--move_to_xyz).

```python
arm.move_to_xyz(0.25, 0.0, 0.15)                    # tip 25cm forward, 15cm up
arm.move_to_xyz(0.20, 0.10, 0.05, wrist_pitch=1.57) # tip pointing straight down
```

---

### 2.4 Inverse Kinematics — move_to_xyz

#### What is Inverse Kinematics?

**Forward kinematics** is easy: given joint angles, calculate where the tip ends up.  
**Inverse kinematics (IK)** is the reverse: given where you *want* the tip to be, calculate what joint angles to use.

IK is harder because there are usually multiple solutions (or none), and the maths is not straightforward to invert. For this arm, a **geometric / analytical** approach is used — it solves the geometry directly using trigonometry rather than iterating numerically.

#### The Arm's Structure

The RoArm-M3 is a serial chain of links:

```
[Base] → [Shoulder] → [Upper Arm L1] → [Elbow] → [Forearm L2] → [Wrist] → [Tip L3]
```

The key insight is that the arm can be decomposed into two independent problems:

1. **Base rotation** — a simple 2D problem in the horizontal plane
2. **Shoulder + Elbow + Wrist** — a planar 3-link chain in the vertical plane

#### Link Lengths

These are the physical dimensions of the arm (in metres), taken from the RoArm-M3 URDF model:

| Variable | Value | Description |
|---|---|---|
| `L0` | 0.054 m | Vertical height from the base origin up to the shoulder joint |
| `L1` | 0.185 m | Upper arm — shoulder joint to elbow joint |
| `L2` | 0.185 m | Forearm — elbow joint to wrist joint |
| `L3` | 0.070 m | Wrist to tip (the hand_tcp offset) |

#### Coordinate Frame

```
        Z (up)
        │
        │
        └──── X (forward, away from arm)
       /
      Y (left)
```

The arm base sits at the origin. `x` is forward, `y` is left, `z` is up. All values in metres.

#### Step-by-Step IK Solution

**Step 1 — Base angle**

The base rotates around the Z axis. To face the target point (x, y), the base angle is simply:

```
base = atan2(y, x)
```

This is exact and always has a unique solution (assuming the target is not directly above the base).

**Step 2 — Remove the wrist-to-tip offset**

The wrist joint is not at the tip — there is a `L3 = 0.070 m` offset from wrist to tip. The `wrist_pitch` parameter controls the angle of the tip (0 = horizontal, π/2 = pointing straight down).

Before solving for shoulder and elbow, we back-calculate where the *wrist joint* needs to be, by subtracting the tip offset from the target:

```
wrist_r = r_total − L3 × cos(wrist_pitch)    ← horizontal distance to wrist
wrist_z = (z − L0) − L3 × sin(−wrist_pitch)  ← vertical height of wrist
```

where `r_total = sqrt(x² + y²)` is the horizontal distance from the base axis to the target.

**Step 3 — Two-link IK for shoulder and elbow**

Now we have a classic two-link planar IK problem: find shoulder angle θ₁ and elbow angle θ₂ such that the chain of length L1 + L2 reaches the wrist point (wrist_r, wrist_z).

First, check reachability. The distance to the wrist is:

```
D = sqrt(wrist_r² + wrist_z²)
```

- If `D > L1 + L2` — the target is too far away (out of reach)
- If `D < |L1 − L2|` — the target is too close (arm can't fold back that far)

If reachable, the **elbow angle** is found using the **cosine rule**:

```
cos(elbow) = (D² − L1² − L2²) / (2 × L1 × L2)
elbow = acos(cos_elbow)
```

This gives the "elbow-up" solution (elbow bends upward). There is also an elbow-down solution (`elbow = -acos(...)`), but elbow-up is used here as it's the natural picking posture.

The **shoulder angle** is then:

```
alpha = atan2(wrist_z, wrist_r)              ← angle from horizontal to the wrist point
beta  = atan2(L2 × sin(elbow), L1 + L2 × cos(elbow))  ← elbow offset angle
shoulder = alpha − beta
```

**Step 4 — Wrist compensation**

The wrist joint compensates to keep the tip at the requested `wrist_pitch` angle, regardless of how the shoulder and elbow are positioned:

```
wrist = wrist_pitch − shoulder − elbow
```

This ensures the tip orientation is maintained throughout the motion.

**Step 5 — Joint limit check**

Before sending anything, every computed angle is checked against the physical joint limits:

| Joint | Min | Max |
|---|---|---|
| base | −180° | +180° |
| shoulder | −90° | +90° |
| elbow | 0° | +180° |
| wrist | −90° | +90° |
| roll | −180° | +180° |

If any joint is out of range, the function prints a clear error and returns `False` without moving the arm.

**Step 6 — Send**

If all checks pass, `send_arm_command()` is called with the computed angles.

#### Tuning the Link Lengths

If the arm doesn't quite reach where you expect, measure `L1` and `L2` on your physical arm with a ruler and update the constants inside `move_to_xyz`. The geometry is correct — small manufacturing variations between units are the most common cause of offset.

---

### 2.5 POSITIONS Dictionary

A set of named `ArmPosition` objects for common poses used in pick-and-place:

| Key | Description |
|---|---|
| `"home"` | All joints at zero, gripper open — safe resting position |
| `"above_object"` | Positioned above the pick target, gripper open |
| `"pick"` | Same as above_object but gripper closed — the grab pose |
| `"lift"` | Raised position with gripper closed — holds the object |
| `"place"` | Rotated to the drop location, gripper open |

These are hardcoded joint angles tuned for a specific physical setup. You will need to adjust them for your workspace using `arm_calibration.py` or the calibration menu option.

---

### 2.6 PickAndPlaceController

A high-level controller that wraps `RoArm3Controller` and executes multi-step sequences.

```python
arm = RoArm3Controller()
pnp = PickAndPlaceController(arm)
pnp.pick_and_place_cycle()
```

#### `pick_and_place_cycle(approach_delay, grip_delay, lift_delay)`

Executes the full 7-step pick-and-place sequence:

1. Move to **home**
2. Move **above object** (gripper open)
3. **Descend** and close gripper
4. **Lift** the object
5. Move to **place** location
6. **Open gripper** (release)
7. Return to **home**

If any step fails, it catches the exception and attempts to return to home as a recovery action.

The delay parameters control how long the arm waits at each stage — useful for letting the gripper settle or the arm stabilise before the next move.

#### `calibrate_positions()`

Cycles through every position in the `POSITIONS` dictionary in order, pausing 2 seconds at each one. Use this to visually verify that all your positions are correct before running a full cycle.

#### `demo_gripper(cycles)`

Opens and closes the gripper `cycles` times (default 3). Useful for testing the gripper in isolation.

---

### 2.7 main() and the Interactive Menu

Running `python3 armfile.py` starts an interactive menu:

```
│  1 = Pick-and-place cycle      │
│  2 = Calibrate all positions   │
│  3 = Demo gripper              │
│  4 = Move to specific position │
│  5 = Move home                 │
│  6 = Move to XYZ coordinate    │
│  q = Quit                      │
```

Option **6** is the IK entry point. It prompts you for x, y, z in metres and an optional wrist pitch angle in degrees, then calls `move_to_xyz`.

---

## 3. arm_controller.py

**File:** `workspace/arm_controller.py`  
**Run with:** `python3 workspace/arm_controller.py` (requires ROS2 running)  
**Dependencies:** ROS2 Humble, `roarm_msgs`, MoveIt2

This is the ROS2 version of the arm controller. Instead of talking to the arm directly over serial, it communicates through ROS2 services and topics. MoveIt2 handles the motion planning and IK internally — you just give it Cartesian coordinates.

### How it connects

The node creates two **service clients** and one **publisher**:

| ROS2 interface | Type | Purpose |
|---|---|---|
| `/move_joint_cmd` | Service (`MoveJointCmd`) | Move the arm tip to an (x, y, z) position |
| `/get_pose_cmd` | Service (`GetPoseCmd`) | Query the arm's current tip position |
| `/gripper_cmd` | Topic (`Float32`) | Set the gripper angle |

On startup it waits up to 5 seconds for both services to become available. If the arm driver node isn't running, it will time out here.

### ArmController class

#### `move_to(x, y, z, label="")`
Sends a `MoveJointCmd` service request with the target coordinates in metres. Blocks until the arm confirms it has reached the position. The `label` string is just for logging.

```python
arm.move_to(0.2, 0.0, 0.2, "UP")
```

#### `get_position()`
Calls the `GetPoseCmd` service and returns the current tip position as a response object with `.x`, `.y`, `.z` fields.

#### `set_gripper(value, label="")`
Publishes a `Float32` to `/gripper_cmd`. Range is `0.0` (fully open) to `1.5` (fully closed). Sleeps for 2 seconds after publishing to give the gripper time to move.

### main() sequence

The default `main()` runs a hardcoded demo:
1. Query current position
2. Move up to z=0.2
3. Move down to z=0.05
4. Move up again
5. Open gripper → close → open
6. Return to home at z=0.1

This is a good first test to confirm the ROS2 connection is working.

---

## 4. arm_calibration.py

**File:** `workspace/arm_calibration.py`  
**Run with:** `python3 workspace/arm_calibration.py` (requires ROS2 running)  
**Dependencies:** ROS2 Humble, `roarm_msgs`, MoveIt2

An interactive tool for figuring out how far the arm actually moves per command unit, and for setting safe movement limits. The problem it solves: when you tell the arm to move 0.05 m, does it actually move 5 cm? This tool lets you measure and tune that.

### Configuration constants

At the top of the file — **these are the values you tune:**

```python
STEP_X = 0.05       # metres moved per xf/xb command
STEP_Y = 0.05       # metres moved per yl/yr command
STEP_Z = 0.05       # metres moved per zu/zd command
STEP_GRIPPER = 0.25 # gripper units per go/gc command

HOME_X = 0.2        # home position
HOME_Y = 0.0
HOME_Z = 0.15

LIMIT_X = (0.05, 0.40)    # arm won't go outside these bounds
LIMIT_Y = (-0.30, 0.30)
LIMIT_Z = (0.02, 0.40)
LIMIT_GRIPPER = (0.0, 1.5)
```

### CalibrationController class

Extends the same ROS2 node pattern as `arm_controller.py`. Tracks the current position locally in `cur_x`, `cur_y`, `cur_z` so it can apply incremental steps without querying the arm every time.

The `clamp()` helper ensures every move stays within the configured limits — the arm will silently stop at the boundary rather than trying to go further.

### Commands

| Command | Action |
|---|---|
| `xf` | Move forward (+X) by STEP_X |
| `xb` | Move backward (−X) by STEP_X |
| `yl` | Move left (+Y) by STEP_Y |
| `yr` | Move right (−Y) by STEP_Y |
| `zu` | Move up (+Z) by STEP_Z |
| `zd` | Move down (−Z) by STEP_Z |
| `go` | Open gripper by STEP_GRIPPER |
| `gc` | Close gripper by STEP_GRIPPER |
| `pos` | Query and print the arm's reported position |
| `home` | Return to HOME_X/Y/Z |
| `status` | Print current local position and step sizes |
| `q` | Quit (returns to home first) |

### Calibration workflow

1. Run the script — arm moves to home
2. Type `xf` — arm moves forward one step
3. Measure the actual physical distance moved with a ruler
4. If it moved 3 cm but you expected 5 cm, change `STEP_X = 0.03`
5. Repeat for Y and Z axes
6. Once calibrated, copy the step values into `arm_live_control.py`

---

## 5. arm_live_control.py

**File:** `workspace/arm_live_control.py`  
**Run with:** `python3 workspace/arm_live_control.py` (requires ROS2 running)  
**Dependencies:** ROS2 Humble, `roarm_msgs`, MoveIt2, optionally `pynput` (`pip install pynput`)

Real-time control of the arm using your keyboard and optionally your mouse. The script automatically detects whether `pynput` is installed and uses the best available input mode.

### Configuration constants

```python
KEY_STEP_X = 0.03          # metres per keypress / per loop tick
KEY_STEP_Y = 0.03
KEY_STEP_Z = 0.03
KEY_STEP_GRIPPER = 0.25

MOUSE_PIXELS_PER_STEP = 30  # pixels of mouse movement = one arm step
KEY_REPEAT_DELAY = 0.15     # seconds between moves when holding a key
```

### LiveController class

Same ROS2 node pattern as the other files. Adds two important features:

- **Rate limiting** — `_last_move_time` prevents flooding the ROS2 service with too many requests when a key is held down. Only one move is sent per `KEY_REPEAT_DELAY` seconds.
- **Thread lock** — `_move_lock` prevents two threads from sending move commands simultaneously (important in pynput mode where the movement loop runs on a background thread).

### Two input modes

#### Keyboard-only fallback (no pynput)

Uses Python's `tty` and `termios` modules to put the terminal into raw mode, so each keypress is read immediately without needing Enter. One keypress = one step. Works on Linux and macOS without any extra packages.

#### pynput mode (recommended)

Uses the `pynput` library for non-blocking input. Key advantages:
- **Held keys** — holding W continuously moves the arm forward until you release
- **Mouse control** — moving the mouse controls the Y and Z axes

A background thread (`movement_loop`) runs every `KEY_REPEAT_DELAY` seconds and checks which keys are currently held, then sends the appropriate move command.

### Controls

| Input | Action |
|---|---|
| W | Forward (+X) |
| S | Backward (−X) |
| A | Left (+Y) |
| D | Right (−Y) |
| Q | Up (+Z) |
| E | Down (−Z) |
| O | Open gripper |
| C | Close gripper |
| H | Go to home position |
| M | Toggle mouse mode on/off |
| ESC or X | Quit |

**Mouse mode (M to toggle):**

| Input | Action |
|---|---|
| Move mouse left/right | Y axis |
| Move mouse up/down | Z axis |
| Left click | Close gripper |
| Right click | Open gripper |

Mouse movement is accumulated — the arm only moves when the cursor has moved at least `MOUSE_PIXELS_PER_STEP` pixels from the last registered position. This prevents jitter from tiny mouse movements.

---

## 6. cv_detector.py

**File:** `workspace/cv_detector.py`  
**Run with:** `python3 workspace/cv_detector.py`  
**Dependencies:** `pip install opencv-python ultralytics`  
**No arm or ROS2 connection required** — runs entirely standalone on your laptop.

A real-time object detection tool using your webcam and YOLOv8. The purpose in this project is to be the vision layer that will eventually tell the arm where objects are. Right now it runs independently so you can test and tune detection before connecting it to the arm.

### Configuration constants

```python
CAMERA_INDEX = 0             # 0 = built-in webcam, 1 = first external camera
MODEL_NAME   = "yolov8n.pt"  # model size (n/s/m/l/x — nano is fastest)
CONFIDENCE_THRESHOLD = 0.45  # detections below this are ignored
NMS_IOU_THRESHOLD    = 0.45  # overlap threshold for duplicate suppression
FRAME_WIDTH  = 640           # capture resolution
FRAME_HEIGHT = 480
SNAPSHOT_DIR = Path("snapshots")  # where S-key saves images
```

### YOLOv8 model sizes

| Model | Speed | Accuracy | Best for |
|---|---|---|---|
| `yolov8n.pt` | Fastest | Lowest | Laptop, real-time |
| `yolov8s.pt` | Fast | Decent | Good default |
| `yolov8m.pt` | Medium | Medium | Balanced |
| `yolov8l.pt` | Slow | High | GPU recommended |
| `yolov8x.pt` | Slowest | Highest | GPU required |

The model file is downloaded automatically on first run from the Ultralytics servers.

### Detection dataclass

Each detected object is returned as a `Detection` object:

```python
@dataclass
class Detection:
    label:      str    # class name, e.g. "bottle"
    confidence: float  # 0.0–1.0
    x1, y1:    int     # top-left corner of bounding box (pixels)
    x2, y2:    int     # bottom-right corner
    class_id:  int     # YOLO class index (0–79 for COCO)
```

Computed properties: `.cx`, `.cy` (centre pixel), `.width`, `.height`, `.area`.

### ObjectDetector class

#### Constructor
```python
detector = ObjectDetector(
    camera_index=0,
    model_name="yolov8n.pt",
    confidence=0.45,
    iou=0.45,
)
```
Loads the YOLO model immediately. The model download happens here on first run.

#### `open_camera()` / `close_camera()`
Opens and releases the webcam. Sets the capture resolution to `FRAME_WIDTH × FRAME_HEIGHT`.

#### `detect(frame)`
Runs YOLOv8 on a single BGR frame (numpy array). Returns a list of `Detection` objects sorted by confidence, highest first.

```python
frame, detections = detector.detect_once()
for d in detections:
    print(f"{d.label} at ({d.cx}, {d.cy}) — {d.confidence:.0%}")
```

#### `detect_once()`
Convenience method: reads one frame from the camera and runs detection on it. Returns `(frame, detections)`.

#### `draw(frame, detections, highlight=None)`
Draws bounding boxes, confidence labels, and centre crosshairs onto a copy of the frame. If `highlight` is set to a class name (e.g. `"bottle"`), that class is drawn in red instead of the default colour. Returns the annotated frame — the original is not modified.

Also draws a HUD in the corner showing object count, current confidence threshold, active filter, and key hints.

#### `run(target_label=None)`
Opens a live window and runs the full detection loop until you press Q or ESC. This is the main entry point for interactive use.

```python
detector.run()                    # detect everything
detector.run(target_label="cup")  # highlight cups in red
```

#### `print_classes()`
Prints all 80 COCO class names the model can detect, formatted in columns.

### Live window controls

| Key | Action |
|---|---|
| Q or ESC | Quit |
| S | Save a snapshot to `snapshots/` |
| P | Pause / resume detection |
| + or = | Raise confidence threshold by 5% |
| − | Lower confidence threshold by 5% |
| 0 | Show all classes |
| 1–9 | Filter to show only class index 0–8 |

### Command-line usage

```bash
# Basic — detect everything
python3 cv_detector.py

# Highlight a specific object in red
python3 cv_detector.py --target bottle
python3 cv_detector.py --target "cell phone"

# Use a more accurate model
python3 cv_detector.py --model yolov8s.pt

# Raise confidence threshold
python3 cv_detector.py --conf 0.6

# Use a different camera
python3 cv_detector.py --camera 1

# Print all detectable class names
python3 cv_detector.py --list-classes
```

---

## 7. How the Files Connect

```
cv_detector.py          arm_calibration.py
     │                         │
     │  (future: pixel coords) │  (tune step sizes)
     │                         ↓
     │                  arm_live_control.py
     │                         │
     └──────────────────────── ↓
                         arm_controller.py   ←── ROS2 / MoveIt2 ──→  Physical Arm
                                                                            ↑
                         armfile.py  ──────── Serial JSON ────────────────┘
```

- **`arm_controller.py`** is the base ROS2 node pattern. `arm_calibration.py` and `arm_live_control.py` both follow the same structure.
- **`armfile.py`** is a completely separate stack — it bypasses ROS2 entirely and talks to the arm directly over serial. Use this when you don't have a ROS2 environment set up, or when you need precise joint-angle control.
- **`cv_detector.py`** is currently standalone. The next step is to feed its `Detection.cx` / `Detection.cy` pixel coordinates into a coordinate mapper (like the one in `arm_vision_pick.py`) to get world coordinates, then pass those to `arm.move_to_xyz()` or `arm.move_to()`.
