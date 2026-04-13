#!/bin/bash

# Watcher — start FastAPI + React (Vite) UI

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

BACKEND_PID=""
FRONTEND_PID=""

echo "Starting Watcher Application..."
echo ""

if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with required configuration."
    exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

# Trakt OAuth return URL — must match Vite dev port and your Trakt app settings.
export UI_BASE_URL="${UI_BASE_URL:-http://localhost:8501}"

cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "${BACKEND_PID}" ] && kill -9 "${BACKEND_PID}" 2>/dev/null || true
    [ -n "${FRONTEND_PID}" ] && kill -9 "${FRONTEND_PID}" 2>/dev/null || true
    # npm/vite may leave children; free dev ports
    for port in 8080 8501; do
        pids=$(lsof -t -i:"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null || true
        fi
    done
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting FastAPI backend on http://localhost:8080..."
pids=$(lsof -t -i:8080 2>/dev/null || true)
if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
fi
uvicorn app.main:app --reload --port 8080 >"$REPO_ROOT/backend.log" 2>&1 &
BACKEND_PID=$!

sleep 3

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Failed to start backend. Check backend.log for details."
    exit 1
fi

echo "Backend started (PID: $BACKEND_PID)"
echo ""

if [ ! -f frontend/package.json ]; then
    echo "frontend/package.json not found."
    exit 1
fi

if [ ! -d frontend/node_modules ]; then
    echo "Installing frontend dependencies (first run)..."
    (cd frontend && npm install)
fi

echo "Starting React UI on http://localhost:8501..."
pids=$(lsof -t -i:8501 2>/dev/null || true)
if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
fi
(cd frontend && npm run dev) >"$REPO_ROOT/frontend.log" 2>&1 &
FRONTEND_PID=$!

sleep 3

if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "Failed to start frontend. Check frontend.log for details."
    kill "$BACKEND_PID" 2>/dev/null || true
    exit 1
fi

echo "Frontend started (PID: $FRONTEND_PID)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Watcher is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "React UI:     http://localhost:8501"
echo "Backend API:  http://localhost:8080"
echo "API Docs:     http://localhost:8080/docs"
echo ""
echo "UI_BASE_URL=$UI_BASE_URL (set in .env to override)"
echo ""
echo "Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop both services..."
echo ""

wait "$BACKEND_PID" "$FRONTEND_PID"
