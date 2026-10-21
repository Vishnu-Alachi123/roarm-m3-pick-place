#!/bin/bash
set -e

Xvfb :99 -screen 0 1280x800x24 &
sleep 1

DISPLAY=:99 fluxbox &
sleep 1

x11vnc -display :99 -forever -shared -nopw -quiet -bg -o /tmp/x11vnc.log

source /opt/ros/humble/setup.bash
source /home/ws/roarm_ws/install/setup.bash 2>/dev/null || true

exec /bin/bash
