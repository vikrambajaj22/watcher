#!/bin/bash

# Run the Watcher Streamlit App

echo "🎬 Starting Watcher Streamlit App..."
echo ""
echo "Make sure the FastAPI backend is running on http://localhost:8080"
echo "If not, start it with: uvicorn app.main:app --reload"
echo ""

# Load repo-root .env so ADMIN_API_KEY and API_BASE_URL stay aligned with the API
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Set environment variable for API base URL (can be overridden)
export API_BASE_URL=${API_BASE_URL:-"http://localhost:8080"}
export IMAGES_DIR=ui/static/images

# Run streamlit
streamlit run ui/streamlit_app.py --server.port 8501 --server.address localhost

