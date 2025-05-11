#!/bin/bash
set -e

PROJECT_DIR="/home/ubuntu/ai-server"
APP_DIR="$PROJECT_DIR/app"
VENV_DIR="$PROJECT_DIR/venv"
APP_MODULE="main:app"
PORT=8000
LOG_FILE="$HOME/ai.log"

echo "▶️ [ai-server] Starting FastAPI (from app dir)"
cd "$APP_DIR" || exit 1

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
# pip install --upgrade pip
# pip install -r "$PROJECT_DIR/requirements.txt"

exec uvicorn "$APP_MODULE" --host 0.0.0.0 --port "$PORT" >> "$LOG_FILE" 2>&1
