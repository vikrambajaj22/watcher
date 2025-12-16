# watcher

> watchu lookin at?

watcher is a personal movie / TV show recommendation app powered by Trakt, TMDB, and OpenAI LLMs.

## Features
- 🔐 Trakt OAuth authentication
- 📺 Browse and sync watch history
- ✨ AI-powered recommendations (movies/TV/all)
- 🔍 Find similar items via KNN search
- ⚙️ Admin panel for embeddings and FAISS management

## Quick Start

### Start Everything
```bash
./start.sh
```

Access:
- **Streamlit Web App:** http://localhost:8501
- **Backend API:** http://localhost:8080/docs

### Or Start Separately
```bash
# Backend
uvicorn app.main:app --reload --port 8080

# Frontend
./run_streamlit.sh
```

## Streamlit Features

### 🔐 Authentication
- Trakt OAuth login/logout

### 📺 Watch History
- View all watched content
- Filter, sort, and search
- One-click sync from Trakt
- **Click "Find Similar" on any item!**

### ✨ Recommendations
- Personalized suggestions (movies/TV/all)
- AI reasoning for each recommendation
- Adjustable count (1-20)
- **Click "Find Similar" on any item!**

### 🔍 Similar Items
- Search by TMDB ID
- Search by text description
- Click "Find Similar" from watch history or recommendations

### ⚙️ Admin Panel
- System status monitoring
- Sync operations
- Embeddings generation
- FAISS index management

## Environment (minimal)
Create a `.env` file in the repo root (values below are examples):
```env
TRAKT_CLIENT_ID=your-client-id
## Environment Setup

Create a `.env` file in the repo root:
```env
TRAKT_CLIENT_ID=your-client-id
TRAKT_CLIENT_SECRET=your-client-secret
TRAKT_REDIRECT_URI=http://127.0.0.1:8080/auth/trakt/callback
TMDB_API_KEY=your-tmdb-key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=watcher
OPENAI_API_KEY=your-openai-key
```

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Sync Watch History
```bash
python sync_worker.py
```

### 3. Generate Embeddings
```bash
python -c "from app.embeddings import index_all_items; index_all_items(batch_size=128)"
```

### 4. Build FAISS Index
```bash
python -c "from app.vector_store import rebuild_index; rebuild_index(dim=768, factory='IDMap,IVF100,Flat')"
```

### 5. Start the App
```bash
./start.sh
```

## API Endpoints

### Core
- `GET /history` - Fetch watch history
- `POST /recommend/{movie|tv|all}` - Get recommendations
- `POST /mcp/knn` - Find similar items

### Authentication
- `GET /auth/trakt/start` - Start OAuth
- `GET /auth/trakt/callback` - OAuth callback

### Admin
- `POST /admin/sync/trakt` - Sync Trakt history
- `POST /admin/embed/item` - Embed single item
- `POST /admin/embed/full` - Embed all items
- `POST /admin/faiss/rebuild` - Rebuild FAISS index

## Deployment

### GCP Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/watcher

gcloud run deploy watcher \
  --image gcr.io/PROJECT_ID/watcher \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars TRAKT_CLIENT_ID=xxx,TRAKT_CLIENT_SECRET=xxx,TRAKT_REDIRECT_URI=xxx,TMDB_API_KEY=xxx,MONGODB_URI=xxx,MONGODB_DB_NAME=watcher,OPENAI_API_KEY=xxx
```

## License
MIT
