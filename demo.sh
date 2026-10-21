#!/usr/bin/env bash
# ============================================================
# RoArm-M3 Demo Launcher
# Usage:
#   ./demo.sh sim       — watch pick policy in simulation (5 episodes)
#   ./demo.sh follow    — interactive: drag sphere, arm follows (sim)
#   ./demo.sh arm       — run pick policy on real arm (requires HW setup)
#   ./demo.sh reach     — run reach policy on real arm (simplest HW test)
# ============================================================
set -e

SIM_DIR="$(cd "$(dirname "$0")/simulation" && pwd)"
WS_DIR="$(cd "$(dirname "$0")/workspace" && pwd)"
PICK_MODEL="$SIM_DIR/models/pick_best/best_model.zip"
REACH_MODEL="$SIM_DIR/models/reach_best/best_model.zip"

# Detect mjpython (macOS MuJoCo requirement) vs plain python
if command -v mjpython &>/dev/null; then
    PY=mjpython
else
    PY=python3
fi

case "${1:-}" in

  sim)
    echo "==> Simulation: pick policy (5 episodes)"
    echo "    Viewer will open. Watch the arm pick the red cube."
    echo "    Press Escape or close viewer to exit."
    cd "$SIM_DIR"
    $PY play_policy.py --task pick --model "$PICK_MODEL" --episodes 5
    ;;

  follow)
    echo "==> Simulation: interactive follow mode"
    echo "    Drag the GREEN SPHERE in the viewer — the arm chases it."
    echo "    Press Escape or close viewer to exit."
    cd "$SIM_DIR"
    $PY play_policy.py --task reach --follow
    ;;

  arm)
    echo "==> Real arm: pick policy"
    echo ""
    echo "    Prerequisites:"
    echo "      1. docker-compose up -d"
    echo "      2. ./bridge-arm-serial.sh"
    echo "      3. python workspace/arm_calibration.py  (first time only)"
    echo ""
    echo "    Place a cube at ~25 cm in front of the arm, centred."
    echo "    The script will ask for TWO confirmations before moving."
    echo ""
    read -r -p "    Ready? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    cd "$(dirname "$0")"
    python3 workspace/run_policy_on_arm.py \
        --task pick \
        --model "$PICK_MODEL" \
        --goal 0.25 0.05 0.12 \
        --cube 0.25 0.05
    ;;

  reach)
    echo "==> Real arm: reach policy (simplest hardware test)"
    echo ""
    echo "    Prerequisites:"
    echo "      1. docker-compose up -d"
    echo "      2. ./bridge-arm-serial.sh"
    echo "      3. python workspace/arm_calibration.py  (first time only)"
    echo ""
    echo "    The arm will move to (0.25, 0.0, 0.10) m."
    echo "    The script will ask for TWO confirmations before moving."
    echo ""
    read -r -p "    Ready? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    cd "$(dirname "$0")"
    python3 workspace/run_policy_on_arm.py \
        --task reach \
        --model "$REACH_MODEL" \
        --goal 0.25 0.0 0.10
    ;;

  *)
    echo "Usage: $0 <sim|follow|arm|reach>"
    echo ""
    echo "  sim     — watch pick policy in simulation"
    echo "  follow  — interactive: drag sphere, sim arm follows"
    echo "  arm     — run pick policy on real arm (HW required)"
    echo "  reach   — run reach policy on real arm (simplest HW test)"
    exit 1
    ;;

esac
