#!/bin/bash

# Watcher - Start Both Backend and Frontend

echo "🎬 Starting Watcher Application..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with required configuration."
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Start the FastAPI backend
echo "🚀 Starting FastAPI backend on http://localhost:8080..."
uvicorn app.main:app --reload --port 8080 > backend.log 2>&1 &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Failed to start backend. Check backend.log for details."
    exit 1
fi

echo "✅ Backend started (PID: $BACKEND_PID)"
echo ""

# Start the Streamlit frontend
echo "🎨 Starting Streamlit frontend on http://localhost:8501..."
export API_BASE_URL=${API_BASE_URL:-"http://localhost:8080"}
streamlit run streamlit_app.py --server.port 8501 --server.address localhost > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait a bit for frontend to start
sleep 3

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Failed to start frontend. Check frontend.log for details."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎬 Watcher is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Frontend UI:  http://localhost:8501"
echo "🔧 Backend API:  http://localhost:8080"
echo "📚 API Docs:     http://localhost:8080/docs"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "Press Ctrl+C to stop both services..."
echo ""

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID

