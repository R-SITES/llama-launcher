#!/bin/bash
# llama.cpp Launcher — Command Builder UI
set -e

cd "$(dirname "$0")" || exit 1
PORT=9876
PIDFILE="/tmp/llama-launcher.pid"
LOGFILE="/tmp/llama-launcher.log"

# If already running, just open browser
if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    xdg-open "http://localhost:$PORT" &>/dev/null &
    exit 0
fi

# Clean stale PID
rm -f "$PIDFILE"

# Start server
python3 server.py &> "$LOGFILE" &
PID=$!
echo $PID > "$PIDFILE"

# Wait for it
sleep 1.5
if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: Server failed to start. Check $LOGFILE" >&2
    cat "$LOGFILE" >&2
    rm -f "$PIDFILE"
    exit 1
fi

# Open browser
xdg-open "http://localhost:$PORT" &>/dev/null &
