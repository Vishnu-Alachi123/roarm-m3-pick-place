#!/bin/bash
# Run this INSIDE the container after bridge-arm-serial.sh is running on
# the Mac host. It creates /dev/ttyUSB0 as a pseudo-terminal that
# transparently proxies to the real serial port over TCP, so armfile.py's
# port auto-detection (or ROS2 nodes expecting /dev/ttyUSB0) work unmodified.
#
# Usage: ./connect-arm-serial.sh [tcp-port]

TCP_PORT="${1:-9001}"
rm -f /dev/ttyUSB0

echo "Connecting /dev/ttyUSB0 -> host.docker.internal:$TCP_PORT"

# socat's PTY session ends the moment the client (e.g. pyserial) closes the
# port, which deletes /dev/ttyUSB0. Respawn it in a loop so the device comes
# back immediately for the next open — needed because Python scripts open
# and close the serial connection repeatedly, not just once.
( while true; do
    socat PTY,link=/dev/ttyUSB0,raw,waitslave TCP:host.docker.internal:"$TCP_PORT" 2>/tmp/connect-arm-serial.log
    sleep 0.5
done ) &

sleep 1
if [ -e /dev/ttyUSB0 ]; then
    echo "/dev/ttyUSB0 ready (auto-restarts on close, see /tmp/connect-arm-serial.log)."
else
    echo "Failed to create /dev/ttyUSB0 — is bridge-arm-serial.sh running on the host?"
    exit 1
fi
