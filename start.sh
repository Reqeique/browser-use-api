#!/bin/bash

# Start Xvfb on display :99
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp -nolisten unix &
export DISPLAY=:99

# Wait for Xvfb to start
sleep 2

# Start x11vnc on port 5900
x11vnc -display :99 -nopw -listen localhost -xkb -ncache 10 -forever -shared &

# Start noVNC on port 6080
websockify --web=/usr/share/novnc 6080 localhost:5900 &

# Start the application
uvicorn main:app --host 0.0.0.0 --port 