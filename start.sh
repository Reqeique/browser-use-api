#!/bin/bash

# Start Xvfb
Xvfb :99 -screen 0 1280x720x16 &
export DISPLAY=:99

# Start x11vnc
x11vnc -display :99 -nopw -forever &

# Start noVNC
/usr/share/novnc/utils/launch.sh --vnc localhost:5900 &

# Start the application
uvicorn main:app --host 0.0.0.0 --port 5000
