import rclpy
from rclpy.node import Node
from roarm_msgs.srv import MoveJointCmd, GetPoseCmd
from std_msgs.msg import Float32
import time

class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')
        
        # Service clients
        self.move_client = self.create_client(MoveJointCmd, '/move_joint_cmd')
        self.pose_client = self.create_client(GetPoseCmd, '/get_pose_cmd')
        
        # Gripper publisher
        self.gripper_pub = self.create_publisher(Float32, '/gripper_cmd', 10)
        
        # Wait for services
        self.get_logger().info('Waiting for services...')
        self.move_client.wait_for_service(timeout_sec=5.0)
        self.pose_client.wait_for_service(timeout_sec=5.0)
        self.get_logger().info('Ready!')

    def move_to(self, x, y, z, label=""):
        self.get_logger().info(f'Moving to {label} → x={x}, y={y}, z={z}')
        req = MoveJointCmd.Request()
        req.x = float(x)
        req.y = float(y)
        req.z = float(z)
        future = self.move_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info(f'Reached {label}')

    def get_position(self):
        req = GetPoseCmd.Request()
        future = self.pose_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        pos = future.result()
        self.get_logger().info(f'Current position → x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}')
        return pos

    def set_gripper(self, value, label=""):
        # 0.0 = fully open, 1.5 = fully closed
        self.get_logger().info(f'Gripper → {label} ({value})')
        msg = Float32()
        msg.data = float(value)
        self.gripper_pub.publish(msg)
        time.sleep(2.0)  # wait for gripper to move

def main():
    rclpy.init()
    arm = ArmController()

    print("\n=== Starting arm sequence ===\n")

    # Get starting position
    arm.get_position()
    time.sleep(1.0)

    # Move UP
    arm.move_to(0.2, 0.0, 0.2, "UP")
    time.sleep(2.0)

    # Move DOWN
    arm.move_to(0.2, 0.0, 0.05, "DOWN")
    time.sleep(2.0)

    # Move UP again
    arm.move_to(0.2, 0.0, 0.2, "UP again")
    time.sleep(2.0)

    # Open gripper fully
    arm.set_gripper(0.0, "OPEN")

    # Close gripper
    arm.set_gripper(1.5, "CLOSED")

    # Open gripper again
    arm.set_gripper(0.0, "OPEN")

    # Return to home
    arm.move_to(0.2, 0.0, 0.1, "HOME")
    time.sleep(2.0)

    print("\n=== Sequence complete ===\n")
    rclpy.shutdown()

if __name__ == '__main__':
    main()