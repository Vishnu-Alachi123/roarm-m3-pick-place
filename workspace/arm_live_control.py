"""
arm_live_control.py

Live keyboard + mouse control for the robotic arm.

KEYBOARD CONTROLS:
  W / S       — move arm forward / backward  (X axis)
  A / D       — move arm left / right        (Y axis)
  Q / E       — move arm up / down           (Z axis)
  O           — open gripper
  C           — close gripper
  H           — return to home position
  ESC or X    — quit

MOUSE CONTROLS (when mouse mode is enabled with M key):
  Move mouse left/right  → Y axis (left/right)
  Move mouse up/down     → Z axis (up/down)
  Left click             → close gripper
  Right click            → open gripper
  M                      — toggle mouse mode on/off

CONFIGURE THESE VALUES below to tune movement speed and sensitivity.
"""

import rclpy
from rclpy.node import Node
from roarm_msgs.srv import MoveJointCmd, GetPoseCmd
from std_msgs.msg import Float32
import time
import threading
import sys

# ─────────────────────────────────────────────
#  CONTROL SETTINGS — tune these values
# ─────────────────────────────────────────────

# How far the arm moves per keypress (meters)
KEY_STEP_X = 0.03
KEY_STEP_Y = 0.03
KEY_STEP_Z = 0.03
KEY_STEP_GRIPPER = 0.25

# Mouse sensitivity — how many pixels of movement = one arm step
MOUSE_PIXELS_PER_STEP = 30

# Delay between moves when holding a key (seconds)
# Lower = faster continuous movement
KEY_REPEAT_DELAY = 0.15

# Safe home position
HOME_X = 0.2
HOME_Y = 0.0
HOME_Z = 0.15

# Movement limits
LIMIT_X = (0.05, 0.40)
LIMIT_Y = (-0.30, 0.30)
LIMIT_Z = (0.02, 0.40)
LIMIT_GRIPPER = (0.0, 1.5)

# ─────────────────────────────────────────────


def clamp(value, low, high):
    return max(low, min(high, value))


# ── Try to import pynput for keyboard/mouse ──────────────────────────────────
try:
    from pynput import keyboard as kb, mouse as ms
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class LiveController(Node):
    def __init__(self):
        super().__init__('arm_live_control')

        self.move_client = self.create_client(MoveJointCmd, '/move_joint_cmd')
        self.pose_client = self.create_client(GetPoseCmd, '/get_pose_cmd')
        self.gripper_pub = self.create_publisher(Float32, '/gripper_cmd', 10)

        self.get_logger().info('Waiting for services...')
        self.move_client.wait_for_service(timeout_sec=5.0)
        self.pose_client.wait_for_service(timeout_sec=5.0)
        self.get_logger().info('Ready!')

        self.cur_x = HOME_X
        self.cur_y = HOME_Y
        self.cur_z = HOME_Z
        self.cur_gripper = 0.0

        self._move_lock = threading.Lock()
        self._last_move_time = 0.0

        # Mouse tracking state
        self.mouse_mode = False
        self._mouse_last_x = None
        self._mouse_last_y = None

    # ── Movement ──────────────────────────────────────────────────────────────

    def move_to(self, x, y, z, silent=False):
        x = clamp(x, *LIMIT_X)
        y = clamp(y, *LIMIT_Y)
        z = clamp(z, *LIMIT_Z)

        # Rate-limit to avoid flooding the service
        now = time.time()
        if now - self._last_move_time < KEY_REPEAT_DELAY:
            return
        self._last_move_time = now

        with self._move_lock:
            req = MoveJointCmd.Request()
            req.x = float(x)
            req.y = float(y)
            req.z = float(z)
            future = self.move_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            self.cur_x, self.cur_y, self.cur_z = x, y, z

        if not silent:
            self._print_status()

    def set_gripper(self, value):
        value = clamp(value, *LIMIT_GRIPPER)
        msg = Float32()
        msg.data = float(value)
        self.gripper_pub.publish(msg)
        self.cur_gripper = value
        self._print_status()

    def go_home(self):
        print("\n  → Going home...")
        self.move_to(HOME_X, HOME_Y, HOME_Z, silent=True)
        self._print_status()

    def _print_status(self):
        mouse_str = "ON " if self.mouse_mode else "OFF"
        print(
            f"\r  x={self.cur_x:.3f}  y={self.cur_y:.3f}  z={self.cur_z:.3f}"
            f"  gripper={self.cur_gripper:.2f}  mouse={mouse_str}    ",
            end='', flush=True
        )


# ── Keyboard-only fallback (uses tty/termios, no pynput) ─────────────────────

def run_keyboard_fallback(arm: LiveController):
    """
    Simple blocking keyboard loop using raw terminal input.
    Works without pynput. Each keypress triggers one move.
    """
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print("\n  Keyboard mode active (raw input). Press keys to move.")
    print("  W/S=X  A/D=Y  Q/E=Z  O=open  C=close  H=home  X=quit\n")
    arm._print_status()

    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1).lower()

            if ch in ('x', '\x1b'):   # X or ESC
                break
            elif ch == 'w':
                arm.move_to(arm.cur_x + KEY_STEP_X, arm.cur_y, arm.cur_z)
            elif ch == 's':
                arm.move_to(arm.cur_x - KEY_STEP_X, arm.cur_y, arm.cur_z)
            elif ch == 'a':
                arm.move_to(arm.cur_x, arm.cur_y + KEY_STEP_Y, arm.cur_z)
            elif ch == 'd':
                arm.move_to(arm.cur_x, arm.cur_y - KEY_STEP_Y, arm.cur_z)
            elif ch == 'q':
                arm.move_to(arm.cur_x, arm.cur_y, arm.cur_z + KEY_STEP_Z)
            elif ch == 'e':
                arm.move_to(arm.cur_x, arm.cur_y, arm.cur_z - KEY_STEP_Z)
            elif ch == 'o':
                arm.set_gripper(arm.cur_gripper - KEY_STEP_GRIPPER)
            elif ch == 'c':
                arm.set_gripper(arm.cur_gripper + KEY_STEP_GRIPPER)
            elif ch == 'h':
                arm.go_home()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()  # newline after raw mode


# ── Full pynput mode (keyboard + mouse) ──────────────────────────────────────

def run_pynput_mode(arm: LiveController):
    """
    Non-blocking keyboard + mouse control using pynput.
    Supports held keys for continuous movement.
    """
    held_keys = set()
    stop_event = threading.Event()

    # ── Key handlers ──────────────────────────────────────────────────────────

    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            k = None

        if k:
            held_keys.add(k)

        # Instant actions (not held)
        if k == 'o':
            arm.set_gripper(arm.cur_gripper - KEY_STEP_GRIPPER)
        elif k == 'c':
            arm.set_gripper(arm.cur_gripper + KEY_STEP_GRIPPER)
        elif k == 'h':
            arm.go_home()
        elif k == 'm':
            arm.mouse_mode = not arm.mouse_mode
            arm._mouse_last_x = None
            arm._mouse_last_y = None
            arm._print_status()
        elif k == 'x':
            stop_event.set()
        elif key == kb.Key.esc:
            stop_event.set()

    def on_release(key):
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            k = None
        if k and k in held_keys:
            held_keys.discard(k)

    # ── Mouse handlers ────────────────────────────────────────────────────────

    def on_move(x, y):
        if not arm.mouse_mode:
            return
        if arm._mouse_last_x is None:
            arm._mouse_last_x = x
            arm._mouse_last_y = y
            return

        dx = x - arm._mouse_last_x
        dy = y - arm._mouse_last_y

        moved = False
        if abs(dx) >= MOUSE_PIXELS_PER_STEP:
            steps = int(dx / MOUSE_PIXELS_PER_STEP)
            arm.move_to(arm.cur_x, arm.cur_y - steps * KEY_STEP_Y, arm.cur_z, silent=True)
            arm._mouse_last_x = x
            moved = True
        if abs(dy) >= MOUSE_PIXELS_PER_STEP:
            steps = int(dy / MOUSE_PIXELS_PER_STEP)
            arm.move_to(arm.cur_x, arm.cur_y, arm.cur_z - steps * KEY_STEP_Z, silent=True)
            arm._mouse_last_y = y
            moved = True
        if moved:
            arm._print_status()

    def on_click(x, y, button, pressed):
        if not arm.mouse_mode or not pressed:
            return
        if button == ms.Button.left:
            arm.set_gripper(arm.cur_gripper + KEY_STEP_GRIPPER)
        elif button == ms.Button.right:
            arm.set_gripper(arm.cur_gripper - KEY_STEP_GRIPPER)

    # ── Continuous movement loop ───────────────────────────────────────────────

    def movement_loop():
        while not stop_event.is_set():
            if 'w' in held_keys:
                arm.move_to(arm.cur_x + KEY_STEP_X, arm.cur_y, arm.cur_z)
            if 's' in held_keys:
                arm.move_to(arm.cur_x - KEY_STEP_X, arm.cur_y, arm.cur_z)
            if 'a' in held_keys:
                arm.move_to(arm.cur_x, arm.cur_y + KEY_STEP_Y, arm.cur_z)
            if 'd' in held_keys:
                arm.move_to(arm.cur_x, arm.cur_y - KEY_STEP_Y, arm.cur_z)
            if 'q' in held_keys:
                arm.move_to(arm.cur_x, arm.cur_y, arm.cur_z + KEY_STEP_Z)
            if 'e' in held_keys:
                arm.move_to(arm.cur_x, arm.cur_y, arm.cur_z - KEY_STEP_Z)
            time.sleep(KEY_REPEAT_DELAY)

    print("\n  pynput mode active. Controls:")
    print("  W/S=X  A/D=Y  Q/E=Z  O=open  C=close  H=home  M=mouse  ESC/X=quit\n")
    arm._print_status()

    move_thread = threading.Thread(target=movement_loop, daemon=True)
    move_thread.start()

    with kb.Listener(on_press=on_press, on_release=on_release) as kl, \
         ms.Listener(on_move=on_move, on_click=on_click) as ml:
        stop_event.wait()
        kl.stop()
        ml.stop()

    print()  # newline after status line


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    arm = LiveController()

    print("\n  Moving to home position...")
    arm.move_to(HOME_X, HOME_Y, HOME_Z, silent=True)
    time.sleep(1.5)

    if PYNPUT_AVAILABLE:
        print("  pynput detected — keyboard + mouse control available.")
        run_pynput_mode(arm)
    else:
        print("  pynput not found — falling back to raw keyboard mode.")
        print("  To enable mouse control, install pynput:  pip install pynput")
        run_keyboard_fallback(arm)

    print("\n  Returning to home before exit...")
    arm.move_to(HOME_X, HOME_Y, HOME_Z, silent=True)
    time.sleep(1.5)
    rclpy.shutdown()
    print("  Done.")


if __name__ == '__main__':
    main()
