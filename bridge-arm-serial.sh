#!/bin/bash
# Run this on the Mac HOST (not inside Docker) whenever the ROS2 stack
# in the container needs to talk to the physical arm.
#
# Why this exists: Docker Desktop on macOS runs Linux in a VM and does not
# pass through host USB/serial devices, so /dev/tty.usbserial-* on the Mac
# is invisible inside the container. This bridges the real serial port over
# TCP so the container can reach it via host.docker.internal.
#
# Usage: ./bridge-arm-serial.sh [tcp-port]
# Then inside the container run connect-arm-serial.sh to expose it as
# /dev/ttyUSB0 there.

set -e

TCP_PORT="${1:-9001}"

PORT=$(ls /dev/tty.usbserial* /dev/cu.usbserial* /dev/tty.wchusbserial* \
          /dev/cu.wchusbserial* /dev/tty.SLAB_USBtoUART* \
          /dev/cu.SLAB_USBtoUART* 2>/dev/null | head -1)

if [ -z "$PORT" ]; then
    echo "No USB-serial device found. Is the arm plugged in?"
    echo "Run 'ls /dev/tty.*' before/after plugging it in to find the name"
    echo "if it doesn't match the patterns this script checks."
    exit 1
fi

echo "Bridging $PORT -> TCP port $TCP_PORT (Ctrl+C to stop)"
socat TCP-LISTEN:"$TCP_PORT",reuseaddr,fork FILE:"$PORT",raw,echo=0,b115200
