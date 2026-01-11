#!/bin/bash

# Run the Watcher Streamlit App

echo "🎬 Starting Watcher Streamlit App..."
echo ""
echo "Make sure the FastAPI backend is running on http://localhost:8080"
echo "If not, start it with: uvicorn app.main:app --reload"
echo ""

# Set environment variable for API base URL (can be overridden)
export API_BASE_URL=${API_BASE_URL:-"http://localhost:8080"}
export IMAGES_DIR=ui/static/images

# Run streamlit
streamlit run ui/streamlit_app.py --server.port 8501 --server.address localhost

