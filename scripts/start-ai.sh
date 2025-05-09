#!/bin/bash
set -e

PROJECT_DIR="/home/ubuntu/ai-server"
VENV_DIR="$PROJECT_DIR/venv"
APP_MODULE="app.main:app"
PORT=8000
LOG_FILE="$HOME/ai.log"

echo "▶️ [ai-server] Starting FastAPI (systemd)"
cd "$PROJECT_DIR" || exit 1

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

nohup uvicorn "$APP_MODULE" --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &
