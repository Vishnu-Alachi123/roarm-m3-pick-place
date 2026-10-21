"""
arm_calibration.py

Interactive calibration tool for the robotic arm.
Use this to figure out the real-world movement range and tune your step sizes.

HOW TO USE:
  1. Run this script: python3 arm_calibration.py
  2. Use the menu to move the arm by a configured step size
  3. Measure the actual physical movement
  4. Adjust STEP_* values below until the arm moves the expected distance/angle
  5. Save the values — use them in your other scripts

CONFIGURE THESE VALUES:
  Change the STEP_* constants to tune how much the arm moves per command.
  All values are in meters (the arm's native unit).
"""

import rclpy
from rclpy.node import Node
from roarm_msgs.srv import MoveJointCmd, GetPoseCmd
from std_msgs.msg import Float32
import time

# ─────────────────────────────────────────────
#  CALIBRATION SETTINGS — tune these values
# ─────────────────────────────────────────────

STEP_X = 0.05       # meters per X step (forward/back)
STEP_Y = 0.05       # meters per Y step (left/right)
STEP_Z = 0.05       # meters per Z step (up/down)
STEP_GRIPPER = 0.25 # gripper units per step (range: 0.0 open → 1.5 closed)

# Safe home position — arm returns here on reset
HOME_X = 0.2
HOME_Y = 0.0
HOME_Z = 0.15

# Movement limits — arm won't go outside these bounds
LIMIT_X = (0.05, 0.40)
LIMIT_Y = (-0.30, 0.30)
LIMIT_Z = (0.02, 0.40)
LIMIT_GRIPPER = (0.0, 1.5)

# ─────────────────────────────────────────────


def clamp(value, low, high):
    return max(low, min(high, value))


class CalibrationController(Node):
    def __init__(self):
        super().__init__('arm_calibration')

        self.move_client = self.create_client(MoveJointCmd, '/move_joint_cmd')
        self.pose_client = self.create_client(GetPoseCmd, '/get_pose_cmd')
        self.gripper_pub = self.create_publisher(Float32, '/gripper_cmd', 10)

        self.get_logger().info('Waiting for services...')
        self.move_client.wait_for_service(timeout_sec=5.0)
        self.pose_client.wait_for_service(timeout_sec=5.0)
        self.get_logger().info('Ready!')

        # Track current position locally
        self.cur_x = HOME_X
        self.cur_y = HOME_Y
        self.cur_z = HOME_Z
        self.cur_gripper = 0.0

    def move_to(self, x, y, z):
        x = clamp(x, *LIMIT_X)
        y = clamp(y, *LIMIT_Y)
        z = clamp(z, *LIMIT_Z)
        req = MoveJointCmd.Request()
        req.x = float(x)
        req.y = float(y)
        req.z = float(z)
        future = self.move_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.cur_x, self.cur_y, self.cur_z = x, y, z
        print(f"  → Moved to x={x:.4f}  y={y:.4f}  z={z:.4f}")

    def get_position(self):
        req = GetPoseCmd.Request()
        future = self.pose_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        pos = future.result()
        print(f"  → Reported position: x={pos.x:.4f}  y={pos.y:.4f}  z={pos.z:.4f}")
        return pos

    def set_gripper(self, value):
        value = clamp(value, *LIMIT_GRIPPER)
        msg = Float32()
        msg.data = float(value)
        self.gripper_pub.publish(msg)
        self.cur_gripper = value
        print(f"  → Gripper set to {value:.3f}  (0.0=open, 1.5=closed)")
        time.sleep(1.5)

    def print_status(self):
        print(f"\n  Current (local):  x={self.cur_x:.4f}  y={self.cur_y:.4f}  z={self.cur_z:.4f}  gripper={self.cur_gripper:.3f}")
        print(f"  Step sizes:       X={STEP_X}  Y={STEP_Y}  Z={STEP_Z}  Gripper={STEP_GRIPPER}")


MENU = """
╔══════════════════════════════════════════╗
║         ARM CALIBRATION TOOL             ║
╠══════════════════════════════════════════╣
║  X axis (forward/back)                   ║
║    xf  — move forward  (+X)              ║
║    xb  — move backward (-X)              ║
║                                          ║
║  Y axis (left/right)                     ║
║    yl  — move left     (+Y)              ║
║    yr  — move right    (-Y)              ║
║                                          ║
║  Z axis (up/down)                        ║
║    zu  — move up       (+Z)              ║
║    zd  — move down     (-Z)              ║
║                                          ║
║  Gripper                                 ║
║    go  — open gripper                    ║
║    gc  — close gripper                   ║
║                                          ║
║  Utilities                               ║
║    pos — query arm position              ║
║    home — return to home position        ║
║    status — show current state           ║
║    q   — quit                            ║
╚══════════════════════════════════════════╝
"""


def main():
    rclpy.init()
    arm = CalibrationController()

    print(MENU)
    print("  Moving to home position first...")
    arm.move_to(HOME_X, HOME_Y, HOME_Z)
    time.sleep(1.5)

    while True:
        try:
            cmd = input("\n  Command: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd == 'q':
            break
        elif cmd == 'xf':
            arm.move_to(arm.cur_x + STEP_X, arm.cur_y, arm.cur_z)
        elif cmd == 'xb':
            arm.move_to(arm.cur_x - STEP_X, arm.cur_y, arm.cur_z)
        elif cmd == 'yl':
            arm.move_to(arm.cur_x, arm.cur_y + STEP_Y, arm.cur_z)
        elif cmd == 'yr':
            arm.move_to(arm.cur_x, arm.cur_y - STEP_Y, arm.cur_z)
        elif cmd == 'zu':
            arm.move_to(arm.cur_x, arm.cur_y, arm.cur_z + STEP_Z)
        elif cmd == 'zd':
            arm.move_to(arm.cur_x, arm.cur_y, arm.cur_z - STEP_Z)
        elif cmd == 'go':
            arm.set_gripper(arm.cur_gripper - STEP_GRIPPER)
        elif cmd == 'gc':
            arm.set_gripper(arm.cur_gripper + STEP_GRIPPER)
        elif cmd == 'pos':
            arm.get_position()
        elif cmd == 'home':
            arm.move_to(HOME_X, HOME_Y, HOME_Z)
            time.sleep(1.5)
        elif cmd == 'status':
            arm.print_status()
        else:
            print("  Unknown command. Type 'q' to quit or check the menu above.")

    print("\n  Returning to home before exit...")
    arm.move_to(HOME_X, HOME_Y, HOME_Z)
    time.sleep(1.5)
    rclpy.shutdown()
    print("  Done.")


if __name__ == '__main__':
    main()
