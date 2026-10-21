#!/usr/bin/env python3
"""
RoArm3 Pick-and-Place Implementation
Low-level JSON protocol implementation for 6-DOF arm with gripper control.
 
Protocol:
- Type 102: Full arm command (all 6 joints + gripper)
- Type 106: Direct gripper control
 
Joint order: [base, shoulder, elbow, wrist, roll, hand/gripper]
All angles in RADIANS.
"""
 
import glob
import json
import math
import time
import serial
from typing import List, Tuple, Dict, Optional

# ── Configuration ─────────────────────────────────────────
def _detect_port() -> Optional[str]:
    """Find the arm's serial port. /dev/ttyUSB0 only exists on Linux
    (native or in the Docker container with the socat bridge running);
    macOS exposes USB-serial adapters as /dev/tty.usbserial-*, /dev/cu.usbserial-*,
    /dev/tty.wchusbserial*, or /dev/tty.SLAB_USBtoUART* depending on the chip.
    Returns None if nothing matches, so the caller can fall back to the
    simulator instead of blindly trying a port that isn't there."""
    candidates = (
        glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*") +
        glob.glob("/dev/tty.usbserial*") + glob.glob("/dev/cu.usbserial*") +
        glob.glob("/dev/tty.wchusbserial*") + glob.glob("/dev/cu.wchusbserial*") +
        glob.glob("/dev/tty.SLAB_USBtoUART*") + glob.glob("/dev/cu.SLAB_USBtoUART*")
    )
    return candidates[0] if candidates else None


def _launch_sim(launch_viewer: bool = True):
    """Import and start the MuJoCo simulator from ../simulation/sim.py."""
    import sys
    from pathlib import Path
    sim_dir = Path(__file__).resolve().parent.parent / "simulation"
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    from sim import RoArmSim, JOINT_LIMITS

    try:
        sim = RoArmSim(launch_viewer=launch_viewer)
    except RuntimeError as e:
        if "mjpython" not in str(e):
            raise
        print("[SIM] Running headless (no viewer window) because this isn't "
              "`mjpython`. Run `mjpython armfile.py` on macOS to see the arm.")
        sim = RoArmSim(launch_viewer=False)
    return sim, JOINT_LIMITS


def _clamp_to_limits(angles, limits):
    """Clamp each joint angle to the simulator's [lo, hi] range instead of
    letting an out-of-range value (e.g. the real gripper's wider travel)
    raise — mirrors a servo just hitting its mechanical stop."""
    return [min(max(a, lo), hi) for a, (lo, hi) in zip(angles, limits)]


PORT = _detect_port()
BAUDRATE = 115200
 
SPEED_DEFAULT = 0
ACC_DEFAULT = 10
 
# Gripper angle limits (radians)
GRIPPER_OPEN = 1.08      # ~62° = open
GRIPPER_CLOSED = 3.14    # ~180° = closed

# ── Kinematics ───────────────────────────────────────────────────────────
# URDF-accurate FK/IK lives in roarm_kinematics.py. Coordinates are metres in
# the scene frame: origin on the table below the base axis, x forward, y left,
# z up. Pitch is the gripper's approach angle below horizontal in the arm's
# plane: 0 = pointing straight out, 90 deg = straight down, above 90 = tilted
# back toward the base. NOTE: joint angles use the URDF convention. Whether the
# firmware's T:102 angles match it (the wiki lists elbow "middle" = 1.57) still
# has to be confirmed on the real arm.
import roarm_kinematics as _K


def solve_ik(x: float, y: float, z: float,
             wrist_pitch: float = 0.0, wrist_roll: float = 0.0,
             q_current=None, verbose: bool = False
             ) -> Optional[Tuple[float, float, float, float, float]]:
    """(x, y, z, pitch) -> (base, shoulder, elbow, wrist, roll) in radians, or
    None if unreachable or outside joint limits at that pitch."""
    sol = _K.inverse_kinematics(x, y, z, pitch=wrist_pitch, roll=wrist_roll,
                                 q_current=q_current)
    if sol is None and verbose:
        print(f"[IK] No solution for ({x:.3f}, {y:.3f}, {z:.3f}) at pitch "
              f"{math.degrees(wrist_pitch):.0f}\u00b0 (out of reach or joint limit).")
    return sol


def find_ik_auto_pitch(x: float, y: float, z: float,
                        wrist_roll: float = 0.0,
                        preferred_pitch: float = 0.0,
                        q_current=None,
                        step_deg: float = 10.0
                        ) -> Tuple[Optional[Tuple[float, float, float, float, float]], Optional[float]]:
    """Try preferred_pitch first, then every other approach pitch from 0 to
    150 deg (closest to the preferred one first) until one has a solution.
    Returns (solution_or_None, pitch_used_or_None)."""
    candidates = [preferred_pitch] + sorted(
        (math.radians(d) for d in range(0, 151, int(step_deg))),
        key=lambda p: abs(p - preferred_pitch))
    for pitch in candidates:
        sol = solve_ik(x, y, z, wrist_pitch=pitch, wrist_roll=wrist_roll, q_current=q_current)
        if sol is not None:
            return sol, pitch
    return None, None


def forward_kinematics(base: float, shoulder: float, elbow: float,
                        wrist: float, roll: float = 0.0) -> Tuple[float, float, float, float]:
    """Joint angles -> (x, y, z, pitch) of the tool point."""
    pos, pitch = _K.forward_kinematics([base, shoulder, elbow, wrist, roll])
    return float(pos[0]), float(pos[1]), float(pos[2]), float(pitch)

# ───────────────────────────────────────────────────────────


class ArmPosition:
    """Represents a predefined arm position with all 6 DOF."""
    
    def __init__(self, base: float, shoulder: float, elbow: float,
                 wrist: float, roll: float, gripper: float, desc: str = ""):
        self.base = base
        self.shoulder = shoulder
        self.elbow = elbow
        self.wrist = wrist
        self.roll = roll
        self.gripper = gripper
        self.desc = desc
    
    def to_list(self) -> List[float]:
        """Return joint angles as [base, shoulder, elbow, wrist, roll, gripper]"""
        return [self.base, self.shoulder, self.elbow, self.wrist, self.roll, self.gripper]
    
    def __repr__(self) -> str:
        return (f"ArmPosition(base={self.base:.3f}, shoulder={self.shoulder:.3f}, "
                f"elbow={self.elbow:.3f}, wrist={self.wrist:.3f}, roll={self.roll:.3f}, "
                f"gripper={self.gripper:.3f})")
 
 
class RoArm3Controller:
    """Low-level RoArm3 controller using JSON serial protocol."""
    
    def __init__(self, port: Optional[str] = PORT, baudrate: int = BAUDRATE):
        """Connect to the real arm over serial if one is plugged in;
        otherwise fall back to the MuJoCo simulator (simulation/sim.py)
        so the same script/menu works away from the lab."""
        self.simulated = False
        self.sim = None
        self._sim_limits = None
        self.port = None
        # [base, shoulder, elbow, wrist, roll, gripper] last commanded —
        # tracked so move_linear_path() can find the current tip position
        # via forward_kinematics() without needing hardware feedback.
        self._last_angles = [0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN]
        self.sim_move_seconds = 1.0     # how long each simulated move takes

        if port is not None:
            try:
                self.port = serial.Serial(port, baudrate, timeout=1.0)
                time.sleep(0.5)
                print(f"[INFO] Connected to {port} @ {baudrate} baud")
                return
            except serial.SerialException as e:
                print(f"[WARN] Could not open {port} ({e}); falling back to simulation.")
        else:
            print("[WARN] No RoArm-M3 serial port detected; falling back to simulation.")

        self.simulated = True
        self.sim, self._sim_limits = _launch_sim()

    def close(self):
        """Close the serial connection or the simulator."""
        if self.simulated:
            if self.sim:
                self.sim.close()
            print("[INFO] Simulation closed")
        elif self.port and self.port.is_open:
            self.port.close()
            print("[INFO] Serial connection closed")
    
    def send_command(self, cmd_dict: Dict) -> bool:
        """
        Send JSON command to RoArm3.
        
        Args:
            cmd_dict: Dictionary representing the command
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd_json = json.dumps(cmd_dict)
            self.port.write((cmd_json + "\n").encode())
            print(f"[CMD] {cmd_json}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send command: {e}")
            return False
    
    def send_arm_command(self, base: float, shoulder: float, elbow: float,
                        wrist: float, roll: float, hand: float,
                        speed: int = SPEED_DEFAULT, acc: int = ACC_DEFAULT) -> bool:
        """
        Send full arm command (Type 102).
        Controls all 6 joints including gripper.
        
        Args:
            base, shoulder, elbow, wrist, roll, hand: Joint angles in radians
            speed: Movement speed (0-255)
            acc: Acceleration (0-50)
        
        Returns:
            True if successful, False otherwise
        """
        if self.simulated:
            angles = _clamp_to_limits(
                [base, shoulder, elbow, wrist, roll, hand], self._sim_limits
            )
            print(f"[SIM] base={angles[0]:.3f} shoulder={angles[1]:.3f} "
                  f"elbow={angles[2]:.3f} wrist={angles[3]:.3f} "
                  f"roll={angles[4]:.3f} hand={angles[5]:.3f}")
            self.sim.move_joints(angles, duration=self.sim_move_seconds)
            self._last_angles = list(angles)
            return True

        cmd = {
            "T": 102,
            "base": round(base, 4),
            "shoulder": round(shoulder, 4),
            "elbow": round(elbow, 4),
            "wrist": round(wrist, 4),
            "roll": round(roll, 4),
            "hand": round(hand, 4),
            "spd": speed,
            "acc": acc
        }
        success = self.send_command(cmd)
        if success:
            self._last_angles = [base, shoulder, elbow, wrist, roll, hand]
        return success
    
    def send_gripper_command(self, angle_rad: float, 
                            speed: int = 100, acc: int = 30) -> bool:
        """
        Send direct gripper command (Type 106).
        Control gripper angle independently.
        
        Args:
            angle_rad: Gripper angle in radians (1.08 = open, 3.14 = closed)
            speed: Gripper speed
            acc: Gripper acceleration
        
        Returns:
            True if successful, False otherwise
        """
        if self.simulated:
            angles = self.sim.get_joint_angles()
            angles[5] = angle_rad
            angles = _clamp_to_limits(angles, self._sim_limits)
            print(f"[SIM] gripper={angles[5]:.3f}")
            self.sim.move_joints(angles, duration=0.6)
            self._last_angles = list(angles)
            return True

        cmd = {
            "T": 106,
            "cmd": round(angle_rad, 4),
            "spd": speed,
            "acc": acc
        }
        success = self.send_command(cmd)
        if success:
            self._last_angles[5] = angle_rad
        return success
    
    def move_to_position(self, position: ArmPosition, 
                        speed: int = SPEED_DEFAULT, acc: int = ACC_DEFAULT) -> bool:
        """
        Move arm to predefined position.
        
        Args:
            position: ArmPosition object
            speed: Movement speed
            acc: Acceleration
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\n[MOVE] → {position.desc}")
        print(f"       Base={position.base:.3f}  Shoulder={position.shoulder:.3f}  "
              f"Elbow={position.elbow:.3f}")
        print(f"       Wrist={position.wrist:.3f}  Roll={position.roll:.3f}  "
              f"Gripper={position.gripper:.3f}")
        
        success = self.send_arm_command(
            position.base, position.shoulder, position.elbow,
            position.wrist, position.roll, position.gripper,
            speed, acc
        )
        return success
    
    def set_gripper(self, angle_rad: float, speed: int = 100, acc: int = 15) -> bool:
        """
        Set gripper position.
        
        Args:
            angle_rad: Target angle in radians
            speed: Gripper speed
            acc: Gripper acceleration
        
        Returns:
            True if successful, False otherwise
        """
        state = "CLOSED" if angle_rad >= GRIPPER_CLOSED - 0.1 else \
                "OPEN" if angle_rad <= GRIPPER_OPEN + 0.1 else \
                "INTERMEDIATE"
        
        print(f"[GRIP] Setting gripper to {angle_rad:.3f} rad ({state})")
        success = self.send_gripper_command(angle_rad, speed, acc)
        return success

    def move_to_xyz(self,
                    x: float, y: float, z: float,
                    wrist_pitch: float = 0.0,
                    wrist_roll: float = 0.0,
                    gripper: float = GRIPPER_OPEN,
                    speed: int = SPEED_DEFAULT,
                    acc: int = ACC_DEFAULT,
                    auto_pitch: bool = True) -> bool:
        """
        Move the tip of the arm to a target (x, y, z) position in meters
        using geometric inverse kinematics (solve_ik / module level).

        The arm is modelled as:
          - Base  : rotates around the vertical (Z) axis
          - Shoulder + Elbow + Wrist : planar 3-link chain in the vertical plane
            that passes through the target point

        Coordinate frame (matches RoArm-M3 / ROS2 convention):
          x — forward, y — left, z — up (metres, from the base joint)

        Args:
            x, y, z     : target tip position in metres
            wrist_pitch : preferred wrist pitch angle (rad). 0 = horizontal
                          tip, positive = tip points down.
            wrist_roll  : wrist roll angle (rad), passed straight through
            gripper     : gripper angle (rad)
            speed, acc  : motion speed / acceleration
            auto_pitch  : if the preferred wrist_pitch has no valid solution,
                          search nearby pitches (find_ik_auto_pitch) instead
                          of failing outright.

        Returns:
            True if a valid IK solution was found and the command was sent,
            False if the target is unreachable at any pitch.
        """
        if auto_pitch:
            solution, pitch_used = find_ik_auto_pitch(
                x, y, z, wrist_roll=wrist_roll, preferred_pitch=wrist_pitch,
                q_current=self._last_angles[:4]
            )
        else:
            solution = solve_ik(x, y, z, wrist_pitch=wrist_pitch, wrist_roll=wrist_roll,
                                 q_current=self._last_angles[:4], verbose=True)
            pitch_used = wrist_pitch if solution is not None else None

        if solution is None:
            print(f"[IK] No solution for ({x:.3f}, {y:.3f}, {z:.3f}) "
                  f"at any approach pitch (out of reach or joint limit).")
            return False

        base, shoulder, elbow, wrist, roll = solution
        pitch_note = "" if abs(pitch_used - wrist_pitch) < 1e-6 else \
            f"  (requested pitch {math.degrees(wrist_pitch):.0f}° didn't fit; "\
            f"using {math.degrees(pitch_used):.0f}° instead)"
        print(f"\n[IK] Target  x={x:.3f}  y={y:.3f}  z={z:.3f}{pitch_note}")
        print(f"     Base={math.degrees(base):.1f}°  "
              f"Shoulder={math.degrees(shoulder):.1f}°  "
              f"Elbow={math.degrees(elbow):.1f}°  "
              f"Wrist={math.degrees(wrist):.1f}°  "
              f"Roll={math.degrees(roll):.1f}°")

        return self.send_arm_command(
            base, shoulder, elbow, wrist, roll, gripper, speed, acc
        )

    def move_linear_path(self,
                        x: float, y: float, z: float,
                        steps: int = 10,
                        wrist_pitch: float = 0.0,
                        wrist_roll: float = 0.0,
                        gripper: Optional[float] = None,
                        speed: int = SPEED_DEFAULT,
                        acc: int = ACC_DEFAULT) -> bool:
        """
        Move the tip in a straight Cartesian line from its current position
        (from forward_kinematics on the last commanded joints) to (x, y, z),
        solving IK at each waypoint — instead of jumping directly from the
        current joint angles to the target's. Falls back to move_to_xyz's
        auto-pitch search per waypoint if the interpolated pitch doesn't fit.

        Args:
            x, y, z     : target tip position in metres
            steps       : number of intermediate waypoints (>=1)
            wrist_pitch : desired pitch at the target (interpolated from
                          the current pitch)
            wrist_roll  : wrist roll angle (rad), held constant
            gripper     : gripper angle (rad); defaults to the last-commanded
                          gripper angle (i.e. left unchanged)
            speed, acc  : motion speed / acceleration per waypoint

        Returns:
            True if every waypoint found a valid IK solution, False if any
            waypoint failed (the arm stops at the last successful waypoint).
        """
        start_x, start_y, start_z, start_pitch = forward_kinematics(
            *self._last_angles[:4]
        )
        if gripper is None:
            gripper = self._last_angles[5]

        print(f"\n[PATH] Linear move: ({start_x:.3f}, {start_y:.3f}, {start_z:.3f}) "
              f"-> ({x:.3f}, {y:.3f}, {z:.3f}) in {steps} steps")

        steps = max(1, steps)
        for i in range(1, steps + 1):
            t = i / steps
            wx = start_x + (x - start_x) * t
            wy = start_y + (y - start_y) * t
            wz = start_z + (z - start_z) * t
            wpitch = start_pitch + (wrist_pitch - start_pitch) * t

            ok = self.move_to_xyz(wx, wy, wz, wrist_pitch=wpitch,
                                   wrist_roll=wrist_roll, gripper=gripper,
                                   speed=speed, acc=acc)
            if not ok:
                print(f"[PATH] Stopped at waypoint {i}/{steps} — no IK solution.")
                return False
        return True
 
 
# ── Predefined Positions ───────────────────────────────────
 
POSITIONS = {
    "home": ArmPosition(
        base=0.0, shoulder=0.0, elbow=0.0,
        wrist=0.0, roll=0.0, gripper=GRIPPER_OPEN,
        desc="Home (arm retracted, gripper open)"
    ),
    "above_object": ArmPosition(
        base=-0.1866, shoulder=0.65, elbow=2.6,
        wrist=0.0, roll=0.0, gripper=GRIPPER_OPEN,
        desc="Above object (ready to descend)"
    ),
    "pick": ArmPosition(
        base=-0.1866, shoulder=0.65, elbow=2.6,
        wrist=0.0, roll=0.0, gripper=GRIPPER_CLOSED,
        desc="At object (gripper closed)"
    ),
    "lift": ArmPosition(
        base=-0.1866, shoulder=0.4, elbow=1.5,
        wrist=0.0, roll=0.0, gripper=GRIPPER_CLOSED,
        desc="Lifted (gripper closed, ready to move)"
    ),
    "place": ArmPosition(
        base=-0.7, shoulder=0.4, elbow=1,
        wrist=0.0, roll=0.0, gripper=GRIPPER_OPEN,
        desc="Place location (gripper open)"
    ),
}
 
# ───────────────────────────────────────────────────────────
 
class PickAndPlaceController:
    """High-level pick-and-place automation."""
    
    def __init__(self, arm: RoArm3Controller):
        """
        Initialize pick-and-place controller.
        
        Args:
            arm: RoArm3Controller instance
        """
        self.arm = arm
    
    def pick_and_place_cycle(self,
                            approach_delay: float = .5,
                            grip_delay: float = 1.5,
                            lift_delay: float = 3) -> bool:
        """
        Execute complete pick-and-place cycle.
        
        Motion sequence:
        1. Move to home (reset)
        2. Move above object
        3. Descend and close gripper
        4. Lift with object secured
        5. Move to place location
        6. Open gripper (release)
        7. Return to home
        
        Args:
            approach_delay: Wait time after approaching (seconds)
            grip_delay: Wait time after gripper closes (seconds)
            lift_delay: Wait time after lifting (seconds)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n" + "╔" + "═" * 38 + "╗")
            print("║   PICK-AND-PLACE CYCLE STARTING        ║")
            print("╚" + "═" * 38 + "╝")
            
            # Step 1: Home
            print("\n[Step 1/7] Moving to HOME...")
            if not self.arm.move_to_position(POSITIONS["home"]):
                raise Exception("Failed to move to home")
            time.sleep(1.0)
            
            # Step 2: Approach
            print("\n[Step 2/7] APPROACHING object...")
            if not self.arm.move_to_position(POSITIONS["above_object"]):
                raise Exception("Failed to approach")
            time.sleep(approach_delay)
            
            # Step 3: Descend & Grip
            print("\n[Step 3/7] DESCENDING to object...")
            if not self.arm.move_to_position(POSITIONS["pick"]):
                raise Exception("Failed to descend")
            print("           → Gripper closed, waiting for grip to settle...")
            time.sleep(grip_delay)
            
            # Step 4: Lift
            print("\n[Step 4/7] LIFTING object...")
            if not self.arm.move_to_position(POSITIONS["lift"]):
                raise Exception("Failed to lift")
            time.sleep(lift_delay)
            
            # Step 5: Move to Place
            print("\n[Step 5/7] MOVING to PLACE location...")
            if not self.arm.move_to_position(POSITIONS["place"]):
                raise Exception("Failed to move to place")
            time.sleep(2.0)
            
            # Step 6: Release
            print("\n[Step 6/7] RELEASING object...")
            if not self.arm.set_gripper(GRIPPER_OPEN):
                raise Exception("Failed to release gripper")
            print("           → Object released")
            time.sleep(2.0)
            
            # Step 7: Return Home
            print("\n[Step 7/7] RETURNING to HOME...")
            if not self.arm.move_to_position(POSITIONS["home"]):
                raise Exception("Failed to return home")
            time.sleep(1.0)
            
            print("\n" + "╔" + "═" * 38 + "╗")
            print("║   ✓ PICK-AND-PLACE CYCLE COMPLETE     ║")
            print("╚" + "═" * 38 + "╝\n")
            return True
        
        except Exception as e:
            print(f"\n[ERROR] Cycle failed: {e}")
            print("[RECOVERY] Attempting to return to HOME...")
            try:
                self.arm.move_to_position(POSITIONS["home"])
                time.sleep(1.0)
            except:
                pass
            return False
    
    def calibrate_positions(self) -> bool:
        """
        Cycle through all predefined positions for calibration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n" + "╔" + "═" * 38 + "╗")
            print("║   CALIBRATION MODE: All Positions     ║")
            print("╚" + "═" * 38 + "╝")
            
            for pos_name, position in POSITIONS.items():
                print(f"\n{'-' * 40}")
                if not self.arm.move_to_position(position):
                    print(f"[ERROR] Failed at position '{pos_name}'")
                    return False
                time.sleep(2.0)
            
            print(f"\n{'-' * 40}")
            print("[CALIB] Complete. Returning to HOME.\n")
            self.arm.move_to_position(POSITIONS["home"])
            return True
        
        except Exception as e:
            print(f"[ERROR] Calibration failed: {e}")
            return False
    
    def demo_gripper(self, cycles: int = 3) -> bool:
        """
        Demonstrate gripper open/close motion.
        
        Args:
            cycles: Number of open/close cycles
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n[DEMO] Gripper demonstration ({cycles} cycles)...\n")
            
            for i in range(cycles):
                print(f"Cycle {i + 1}/{cycles}:")
                
                print("  → Opening gripper...")
                if not self.arm.set_gripper(GRIPPER_OPEN):
                    raise Exception("Failed to open gripper")
                time.sleep(0.5)
                
                print("  → Closing gripper...")
                if not self.arm.set_gripper(GRIPPER_CLOSED):
                    raise Exception("Failed to close gripper")
                time.sleep(0.5)
            
            print("\n[DEMO] Complete.\n")
            return True
        
        except Exception as e:
            print(f"[ERROR] Gripper demo failed: {e}")
            return False
 
 
# ───────────────────────────────────────────────────────────
 
 
def print_menu():
    """Display interactive menu."""
    print("\n┌" + "─" * 36 + "┐")
    print("│  RoArm3 Pick-and-Place Control  │")
    print("├" + "─" * 36 + "┤")
    print("│  1 = Pick-and-place cycle      │")
    print("│  2 = Calibrate all positions   │")
    print("│  3 = Demo gripper              │")
    print("│  4 = Move to specific position │")
    print("│  5 = Move home                 │")
    print("│  6 = Move to XYZ coordinate    │")
    print("│  7 = Linear path to XYZ        │")
    print("│  q = Quit                      │")
    print("└" + "─" * 36 + "┘")
 
 
def main():
    """Main interactive control loop."""
    print("[INFO] Initializing RoArm3 Pick-and-Place Controller\n")
    
    try:
        # Initialize controller
        arm = RoArm3Controller(port=PORT, baudrate=BAUDRATE)
        pnp = PickAndPlaceController(arm)
        
        time.sleep(1.0)
        print("[INFO] Ready.\n")
        
        while True:
            print_menu()
            choice = input("Enter choice: ").strip().lower()
            
            if choice == "1":
                pnp.pick_and_place_cycle()
            
            elif choice == "2":
                pnp.calibrate_positions()
            
            elif choice == "3":
                pnp.demo_gripper()
            
            elif choice == "4":
                print("\nAvailable positions:")
                for i, pos_name in enumerate(POSITIONS.keys(), 1):
                    pos = POSITIONS[pos_name]
                    print(f"  {i} = {pos_name}: {pos.desc}")
                
                pos_choice = input("\nSelect position (number or name): ").strip().lower()
                
                try:
                    idx = int(pos_choice) - 1
                    pos_name = list(POSITIONS.keys())[idx]
                except (ValueError, IndexError):
                    pos_name = pos_choice
                
                if pos_name in POSITIONS:
                    arm.move_to_position(POSITIONS[pos_name])
                    time.sleep(1.0)
                else:
                    print(f"[ERROR] Unknown position: {pos_name}")
            
            elif choice == "5":
                print("\n[HOME] Moving to home position...")
                arm.move_to_position(POSITIONS["home"])
                time.sleep(1.0)
            
            elif choice == "6":
                print("\n[XYZ] Enter target coordinates in metres.")
                print("      Example: 0.25 0.0 0.15  (x forward, y left, z up)")
                try:
                    raw = input("  x y z: ").strip().split()
                    tx, ty, tz = float(raw[0]), float(raw[1]), float(raw[2])
                    
                    pitch_raw = input("  Wrist pitch in degrees (0 = horizontal tip, "
                                      "press Enter for 0): ").strip()
                    import math
                    wrist_pitch = math.radians(float(pitch_raw)) if pitch_raw else 0.0
                    
                    arm.move_to_xyz(tx, ty, tz, wrist_pitch=wrist_pitch)
                    time.sleep(1.0)
                except (ValueError, IndexError):
                    print("[ERROR] Invalid input. Enter three numbers, e.g.: 0.25 0.0 0.15")

            elif choice == "7":
                print("\n[PATH] Straight-line move from the current tip position.")
                print("       Example: 0.25 0.0 0.15  (x forward, y left, z up)")
                try:
                    raw = input("  target x y z: ").strip().split()
                    tx, ty, tz = float(raw[0]), float(raw[1]), float(raw[2])

                    steps_raw = input("  Number of waypoints (press Enter for 10): ").strip()
                    steps = int(steps_raw) if steps_raw else 10

                    arm.move_linear_path(tx, ty, tz, steps=steps)
                    time.sleep(0.5)
                except (ValueError, IndexError):
                    print("[ERROR] Invalid input. Enter three numbers, e.g.: 0.25 0.0 0.15")

            elif choice in ("q", "quit", "exit"):
                print("\n[INFO] Closing connection...")
                arm.close()
                print("[INFO] Goodbye.")
                break
            
            else:
                print("[ERROR] Invalid choice. Try again.")
    
    except Exception as e:
        print(f"[FATAL] {e}")
        return 1
    
    return 0
 
 
if __name__ == "__main__":
    exit(main())