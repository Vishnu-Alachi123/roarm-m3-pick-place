"""
cv_detector.py

Webcam-based object detection using OpenCV + YOLOv8.
Detects objects in real-time and can feed target coordinates
to the RoArm3 pick-and-place controller.

REQUIREMENTS:
  pip install opencv-python ultralytics numpy

USAGE:
  Standalone detection:
    python3 cv_detector.py

  Integrated with arm (pick detected object):
    from cv_detector import ObjectDetector
    detector = ObjectDetector()
    target = detector.get_best_target(label="cup")
    if target:
        arm_x, arm_y = detector.pixel_to_arm_coords(target.cx, target.cy)

CONTROLS (when running standalone):
  Q or ESC  — quit
  S         — save current frame as snapshot
  P         — pause/resume detection
  +/-       — increase/decrease confidence threshold
  1-9       — filter by class index (0 = show all)
"""

import cv2
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────

# Webcam index (0 = default laptop webcam, 1 = external)
CAMERA_INDEX = 0

# YOLOv8 model — options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
# 'n' (nano) is fastest; 'x' is most accurate
MODEL_NAME = "yolov8n.pt"

# Detection confidence threshold (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.45

# Non-maximum suppression IoU threshold
NMS_IOU_THRESHOLD = 0.45

# Display window title
WINDOW_TITLE = "RoArm3 — Object Detection"

# Frame resolution (None = use camera default)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Arm workspace mapping — tune these to match your physical setup
# These define the real-world area (in meters) that the camera can see
# relative to the arm's coordinate system
ARM_WORKSPACE_X = (0.10, 0.35)   # arm X range visible in frame (near → far)
ARM_WORKSPACE_Y = (-0.20, 0.20)  # arm Y range visible in frame (left → right)
ARM_PICK_Z = 0.02                 # Z height to descend to for picking

# ───────────────────────────────────────────────────────────


@dataclass
class Detection:
    """A single detected object."""
    label: str          # class name (e.g. "cup", "bottle")
    confidence: float   # detection confidence 0.0–1.0
    x1: int             # bounding box top-left x
    y1: int             # bounding box top-left y
    x2: int             # bounding box bottom-right x
    y2: int             # bounding box bottom-right y
    class_id: int       # YOLO class index

    @property
    def cx(self) -> int:
        """Bounding box center X (pixels)."""
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        """Bounding box center Y (pixels)."""
        return (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def __repr__(self) -> str:
        return (f"Detection(label='{self.label}', conf={self.confidence:.2f}, "
                f"center=({self.cx}, {self.cy}), size={self.width}x{self.height})")


# Colour palette for bounding boxes (BGR)
_PALETTE = [
    (0, 255, 0),    (0, 128, 255),  (255, 0, 128),  (0, 255, 255),
    (255, 128, 0),  (128, 0, 255),  (255, 255, 0),  (0, 200, 100),
    (200, 0, 200),  (100, 200, 0),
]


def _class_colour(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


class ObjectDetector:
    """
    Real-time object detector using YOLOv8 and a laptop webcam.

    Example usage:
        detector = ObjectDetector()
        detector.run()                          # standalone live view

        # Or use programmatically:
        frame, detections = detector.detect_once()
        target = detector.get_best_target(detections, label="bottle")
        if target:
            arm_x, arm_y = detector.pixel_to_arm_coords(target.cx, target.cy,
                                                         frame.shape)
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        model_name: str = MODEL_NAME,
        confidence: float = CONFIDENCE_THRESHOLD,
        iou: float = NMS_IOU_THRESHOLD,
    ):
        self.camera_index = camera_index
        self.confidence = confidence
        self.iou = iou
        self._cap: Optional[cv2.VideoCapture] = None
        self._model = None
        self._model_name = model_name
        self._paused = False
        self._filter_class: Optional[int] = None   # None = show all
        self._last_detections: List[Detection] = []

        print(f"[CV] Loading model '{model_name}'...")
        self._load_model(model_name)
        print(f"[CV] Model ready. Classes: {len(self._model.names)}")

    # ── Model ─────────────────────────────────────────────────────────────────

    def _load_model(self, model_name: str):
        """Load YOLOv8 model (downloads automatically on first run)."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_name)
        except ImportError:
            raise ImportError(
                "ultralytics is required. Install with:\n"
                "  pip install ultralytics"
            )

    # ── Camera ────────────────────────────────────────────────────────────────

    def open_camera(self) -> bool:
        """Open webcam. Returns True on success."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            print(f"[CV] ERROR: Could not open camera index {self.camera_index}")
            return False

        if FRAME_WIDTH and FRAME_HEIGHT:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[CV] Camera opened: {w}x{h} @ {fps:.0f} fps")
        return True

    def close_camera(self):
        """Release webcam."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            print("[CV] Camera released")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame from the webcam. Returns None on failure."""
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects
        """
        results = self._model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self._model.names[cls_id]
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=cls_id,
                ))

        self._last_detections = detections
        return detections

    def detect_once(self) -> Tuple[Optional[np.ndarray], List[Detection]]:
        """
        Capture one frame and run detection.

        Returns:
            (frame, detections) — frame is None if camera read failed
        """
        frame = self.read_frame()
        if frame is None:
            return None, []
        detections = self.detect(frame)
        return frame, detections

    # ── Targeting ─────────────────────────────────────────────────────────────

    def get_best_target(
        self,
        detections: Optional[List[Detection]] = None,
        label: Optional[str] = None,
    ) -> Optional[Detection]:
        """
        Return the best detection to target.

        Picks the largest bounding box (closest object) optionally filtered
        by class label.

        Args:
            detections: List of detections (uses last frame's if None)
            label: Filter to this class name (e.g. "bottle"). None = any.

        Returns:
            Best Detection, or None if nothing found
        """
        dets = detections if detections is not None else self._last_detections
        if label:
            dets = [d for d in dets if d.label.lower() == label.lower()]
        if not dets:
            return None
        return max(dets, key=lambda d: d.area)

    def pixel_to_arm_coords(
        self,
        px: int,
        py: int,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to arm workspace coordinates (X, Y in meters).

        Assumes the camera is mounted looking straight down at the workspace.
        Tune ARM_WORKSPACE_X / ARM_WORKSPACE_Y at the top of this file to
        match your physical setup.

        Args:
            px: Pixel X (horizontal, 0 = left)
            py: Pixel Y (vertical, 0 = top)
            frame_shape: (height, width, channels) from frame.shape

        Returns:
            (arm_x, arm_y) in meters
        """
        h, w = frame_shape[:2]

        # Normalise to [0, 1]
        nx = px / w   # 0 = left, 1 = right
        ny = py / h   # 0 = top,  1 = bottom

        # Map to arm workspace
        # Camera top → arm far (large X), camera bottom → arm near (small X)
        arm_x = ARM_WORKSPACE_X[1] - ny * (ARM_WORKSPACE_X[1] - ARM_WORKSPACE_X[0])
        # Camera left → arm left (negative Y), camera right → arm right (positive Y)
        arm_y = ARM_WORKSPACE_Y[0] + nx * (ARM_WORKSPACE_Y[1] - ARM_WORKSPACE_Y[0])

        return round(arm_x, 4), round(arm_y, 4)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        highlight: Optional[Detection] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a frame.

        Args:
            frame: BGR image
            detections: Detections to draw
            highlight: If set, this detection gets a special highlight colour

        Returns:
            Annotated frame (copy)
        """
        out = frame.copy()
        h, w = out.shape[:2]

        for det in detections:
            # Skip if class filter is active
            if self._filter_class is not None and det.class_id != self._filter_class:
                continue

            is_target = (highlight is not None and det is highlight)
            colour = (0, 0, 255) if is_target else _class_colour(det.class_id)
            thickness = 3 if is_target else 2

            # Bounding box
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), colour, thickness)

            # Label background
            label_text = f"{det.label} {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(det.y1 - 4, th + 4)
            cv2.rectangle(
                out,
                (det.x1, label_y - th - baseline - 2),
                (det.x1 + tw + 4, label_y + baseline),
                colour, -1
            )
            cv2.putText(
                out, label_text,
                (det.x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

            # Centre crosshair
            cv2.drawMarker(
                out, (det.cx, det.cy), colour,
                cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA
            )

            # Arm coords overlay (if target)
            if is_target:
                arm_x, arm_y = self.pixel_to_arm_coords(det.cx, det.cy, out.shape)
                coord_text = f"arm ({arm_x:.3f}, {arm_y:.3f})"
                cv2.putText(
                    out, coord_text,
                    (det.x1, det.y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA
                )

        # HUD overlay
        self._draw_hud(out, detections, w, h)
        return out

    def _draw_hud(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        w: int,
        h: int,
    ):
        """Draw heads-up display info on the frame."""
        lines = [
            f"Objects: {len(detections)}",
            f"Conf: {self.confidence:.0%}",
            f"Filter: {self._model.names[self._filter_class] if self._filter_class is not None else 'all'}",
            "PAUSED" if self._paused else "",
        ]
        y = 20
        for line in lines:
            if not line:
                continue
            colour = (0, 0, 255) if line == "PAUSED" else (200, 200, 200)
            cv2.putText(
                frame, line, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA
            )
            y += 20

        # Controls hint at bottom
        hint = "Q=quit  S=save  P=pause  +/-=conf  1-9=filter  0=all"
        cv2.putText(
            frame, hint, (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA
        )

    # ── Standalone live loop ──────────────────────────────────────────────────

    def run(self, target_label: Optional[str] = None):
        """
        Run the live detection loop with a display window.

        Args:
            target_label: If set, highlight the best detection of this class
                          (e.g. "bottle", "cup", "person")
        """
        if not self.open_camera():
            return

        print(f"\n[CV] Starting live detection. Target label: {target_label or 'any'}")
        print("[CV] Controls: Q/ESC=quit  S=save  P=pause  +/-=conf  1-9=filter  0=all\n")

        frame_count = 0
        fps_timer = time.time()
        fps_display = 0.0
        snapshot_dir = Path("snapshots")

        try:
            while True:
                # ── Read frame ────────────────────────────────────────────────
                frame = self.read_frame()
                if frame is None:
                    print("[CV] Failed to read frame — camera disconnected?")
                    break

                # ── Detect ────────────────────────────────────────────────────
                if not self._paused:
                    detections = self.detect(frame)
                else:
                    detections = self._last_detections

                # ── Pick best target ──────────────────────────────────────────
                target = self.get_best_target(detections, label=target_label)

                # ── Annotate ──────────────────────────────────────────────────
                annotated = self.draw_detections(frame, detections, highlight=target)

                # FPS counter
                frame_count += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps_display = frame_count / elapsed
                    frame_count = 0
                    fps_timer = time.time()
                cv2.putText(
                    annotated, f"FPS: {fps_display:.1f}",
                    (annotated.shape[1] - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
                )

                # ── Show ──────────────────────────────────────────────────────
                cv2.imshow(WINDOW_TITLE, annotated)

                # ── Key handling ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):   # Q or ESC
                    break

                elif key == ord('s'):       # Save snapshot
                    snapshot_dir.mkdir(exist_ok=True)
                    fname = snapshot_dir / f"snap_{int(time.time())}.jpg"
                    cv2.imwrite(str(fname), annotated)
                    print(f"[CV] Snapshot saved: {fname}")

                elif key == ord('p'):       # Pause/resume
                    self._paused = not self._paused
                    print(f"[CV] {'Paused' if self._paused else 'Resumed'}")

                elif key == ord('+') or key == ord('='):
                    self.confidence = min(0.95, self.confidence + 0.05)
                    print(f"[CV] Confidence threshold: {self.confidence:.0%}")

                elif key == ord('-'):
                    self.confidence = max(0.05, self.confidence - 0.05)
                    print(f"[CV] Confidence threshold: {self.confidence:.0%}")

                elif key == ord('0'):       # Show all classes
                    self._filter_class = None
                    print("[CV] Filter: all classes")

                elif ord('1') <= key <= ord('9'):
                    self._filter_class = key - ord('1')
                    name = self._model.names.get(self._filter_class, "?")
                    print(f"[CV] Filter: class {self._filter_class} ({name})")

        finally:
            self.close_camera()
            cv2.destroyAllWindows()
            print("[CV] Detection stopped.")

    # ── Integration helpers ───────────────────────────────────────────────────

    def scan_for_object(
        self,
        label: str,
        timeout: float = 10.0,
        min_confidence: float = 0.5,
    ) -> Optional[Detection]:
        """
        Scan webcam frames until a specific object is found or timeout expires.
        Useful for integration with the arm controller.

        Args:
            label: Object class to look for (e.g. "bottle", "cup")
            timeout: Max seconds to wait
            min_confidence: Minimum confidence to accept

        Returns:
            Best Detection found, or None if not found within timeout
        """
        if not self.open_camera():
            return None

        print(f"[CV] Scanning for '{label}' (timeout={timeout}s)...")
        deadline = time.time() + timeout
        best: Optional[Detection] = None

        try:
            while time.time() < deadline:
                frame = self.read_frame()
                if frame is None:
                    break
                detections = self.detect(frame)
                candidates = [
                    d for d in detections
                    if d.label.lower() == label.lower()
                    and d.confidence >= min_confidence
                ]
                if candidates:
                    best = max(candidates, key=lambda d: d.area)
                    print(f"[CV] Found: {best}")
                    break
                time.sleep(0.05)
        finally:
            self.close_camera()

        if best is None:
            print(f"[CV] '{label}' not found within {timeout}s")
        return best

    def list_available_classes(self) -> Dict[int, str]:
        """Return all YOLO class names indexed by ID."""
        return dict(self._model.names)

    def print_classes(self):
        """Print all detectable class names."""
        print("\n[CV] Detectable object classes:")
        names = self._model.names
        cols = 4
        items = list(names.items())
        for i in range(0, len(items), cols):
            row = items[i:i + cols]
            print("  " + "  ".join(f"{idx:3d}: {name:<18}" for idx, name in row))
        print()


# ── Arm integration example ───────────────────────────────────────────────────

def pick_detected_object(
    arm,                        # RoArm3Controller instance from armfile.py
    pnp,                        # PickAndPlaceController instance
    target_label: str = "bottle",
    pick_z: float = ARM_PICK_Z,
):
    """
    Detect an object with the webcam and command the arm to pick it up.

    This replaces the hardcoded POSITIONS["above_object"] with a dynamically
    detected position from the camera.

    Args:
        arm: RoArm3Controller instance
        pnp: PickAndPlaceController instance
        target_label: YOLO class name to pick (e.g. "bottle", "cup", "cell phone")
        pick_z: Z height to descend to when picking

    Example:
        from armfile import RoArm3Controller, PickAndPlaceController, PORT, BAUDRATE
        from cv_detector import pick_detected_object

        arm = RoArm3Controller(port=PORT, baudrate=BAUDRATE)
        pnp = PickAndPlaceController(arm)
        pick_detected_object(arm, pnp, target_label="bottle")
    """
    import time
    from armfile import ArmPosition, GRIPPER_OPEN, GRIPPER_CLOSED, POSITIONS

    detector = ObjectDetector()
    detection = detector.scan_for_object(target_label, timeout=10.0)

    if detection is None:
        print(f"[CV] No '{target_label}' detected — aborting pick.")
        return False

    # We need one frame to get the shape for coordinate mapping
    detector.open_camera()
    frame = detector.read_frame()
    detector.close_camera()

    if frame is None:
        print("[CV] Could not read frame for coordinate mapping.")
        return False

    arm_x, arm_y = detector.pixel_to_arm_coords(
        detection.cx, detection.cy, frame.shape
    )
    print(f"[CV] Target arm coords: x={arm_x:.3f}, y={arm_y:.3f}, z={pick_z:.3f}")

    # Move to the detected position with inverse kinematics (gripper pointing
    # straight down). x, y come from the camera; z is the pick height.
    import math
    hover_z = pick_z + 0.08
    print(f"\n[CV+ARM] Picking detected '{target_label}'...")
    arm.move_to_position(POSITIONS["home"])
    time.sleep(1.0)
    if not arm.move_to_xyz(arm_x, arm_y, hover_z, wrist_pitch=math.pi / 2, gripper=GRIPPER_OPEN):
        print("[CV+ARM] Hover point unreachable, aborting.")
        return False
    time.sleep(0.5)
    if not arm.move_to_xyz(arm_x, arm_y, pick_z, wrist_pitch=math.pi / 2, gripper=GRIPPER_OPEN):
        print("[CV+ARM] Pick point unreachable, aborting.")
        return False
    time.sleep(1.0)
    arm.set_gripper(GRIPPER_CLOSED)
    time.sleep(1.5)
    arm.move_to_position(POSITIONS["lift"])
    time.sleep(2.0)
    arm.move_to_position(POSITIONS["place"])
    time.sleep(1.0)
    arm.set_gripper(GRIPPER_OPEN)
    time.sleep(1.0)
    arm.move_to_position(POSITIONS["home"])
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RoArm3 webcam object detector")
    parser.add_argument(
        "--target", "-t", type=str, default=None,
        help="Highlight detections of this class (e.g. bottle, cup, person)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=MODEL_NAME,
        help=f"YOLOv8 model to use (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"Camera index (default: {CAMERA_INDEX})"
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="Print all detectable class names and exit"
    )
    args = parser.parse_args()

    detector = ObjectDetector(
        camera_index=args.camera,
        model_name=args.model,
        confidence=args.conf,
    )

    if args.list_classes:
        detector.print_classes()
        return

    detector.run(target_label=args.target)


if __name__ == "__main__":
    main()
