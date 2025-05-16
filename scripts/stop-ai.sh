#!/bin/bash

APP_MODULE="app.main:app"
PID=$(pgrep -f "uvicorn.*$APP_MODULE" || true)

if [ -n "$PID" ]; then
  echo "🛑 [ai-server] Stopping PID $PID"
  kill "$PID"
  sleep 2
else
  echo "ℹ️ No running process found."
fi
