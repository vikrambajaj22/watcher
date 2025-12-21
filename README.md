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
Before generating embeddings, check the section on TMDB sync behavior below to understand how initial sync works.
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

## TMDB sync behavior
### Export-first initial sync, then incremental changes

The sync flow prefers TMDB's official daily export files (complete ID dumps) for the initial full ingest and falls back to the `/discover/{movie|tv}` endpoint if an export isn't available. After the initial ingest, the service uses the `/movie/changes` and `/tv/changes` endpoints for incremental updates.

Why exports?
- TMDB provides daily gzipped ID exports for movies and TV series which contain every TMDB ID (millions of titles) and are updated daily. Using the exports:
  - avoids the 500-page limit on the `/discover` API
  - provides full coverage of IDs
  - is faster and more reliable for initial population

Export URLs (example):
```
http://files.tmdb.org/p/exports/movie_ids_MM_DD_YYYY.json.gz
http://files.tmdb.org/p/exports/tv_series_ids_MM_DD_YYYY.json.gz
```
Each line in the gz file is JSON, e.g.:
```
{"id":550,"original_title":"Fight Club","popularity":61.4}
```

How the sync works:
1. On the first run (no stored `last_sync`), the app attempts an export-based initial ingest:
   - Try today's export; if not present, try previous days (configurable lookback, default 7 days).
   - Stream the gz file; extract IDs; batch IDs into CHUNK_SIZE; fetch full metadata per ID via `/movie/{id}` or `/tv/{id}` (append credits, keywords) and upsert.
   - Optionally queue embeddings for each item (recommended to disable for the initial bulk ingest).
   - Set the `tmdb_<media_type>_last_sync` meta to the max `updated_at` observed so later runs can use `/changes`.
2. If no suitable export is found within the lookback window, fall back to the `/discover/{movie|tv}` path (with a safety cap on pages to avoid TMDB page errors).
   - Note: `/discover` is limited by TMDB and may not return the full universe of IDs; TMDB's API allows up to 500 pages — the app caps discover pages by default to 500 pages. For 100% coverage prefer the export files described above.
   - You can override the cap with the environment variable `TMDB_DISCOVER_MAX_PAGES` (e.g., export TMDB_DISCOVER_MAX_PAGES=500) but this may still be limited by TMDB.
3. On subsequent runs, use `/movie/changes` and `/tv/changes` to fetch incremental updates only.

Operational notes & recommendations
- Initial ingest can be very large. For safety/cost control: run the initial ingest with `embed_updated=False` (metadata-only), then run embeddings separately with throttling.
- Exports avoid the discover 500-page limit, but large runs may still hit TMDB rate limits when fetching per-ID details — consider adding rate-limiting/backoff.
- The export streaming handles gz files and processes IDs in batches to bound memory usage.

Extra tips
- To force use of TMDB exports (recommended for full coverage) make sure your environment can reach `http://files.tmdb.org/p/exports/` and increase the `days_back_limit` if the latest export isn't available for any reason.
- To control discover behavior (if you must use discover), set `TMDB_DISCOVER_MAX_PAGES` in your environment to the number of pages to attempt (default 500). Example:

```bash
export TMDB_DISCOVER_MAX_PAGES=500
```

Then run your sync (metadata-only recommended for initial run):

```bash
python - <<'PY'
from app.tmdb_sync import sync_tmdb
sync_tmdb('movie', full_sync=True, embed_updated=False)
PY
```

Force-refresh metadata
----------------------
If you want to re-fetch metadata from TMDB even for IDs already present in your database (for example if you suspect data drift or want to refresh all fields), use the `force_refresh` flag:

```bash
python - <<'PY'
from app.tmdb_sync import sync_tmdb
# full sync, re-fetch metadata for all IDs, but skip embedding during the initial pass
sync_tmdb('movie', full_sync=True, embed_updated=False, force_refresh=True)
PY
```

After a force-refresh you can run embedding passes separately (recommended) so your system doesn't hit TMDB and your embedding provider simultaneously.

**Location**: the sync logic is implemented in `app/tmdb_sync.py` (see `_sync_from_export`, `_fetch_all_ids_by_discover`, and `sync_tmdb_changes`).

## API Endpoints

### Core
- `GET /history` - Fetch watch history
- `POST /recommend/{movie|tv|all}` - Get recommendations
- `POST /mcp/knn` - Find similar items

Notes for `POST /mcp/knn`:
- `media_type` is a required field in the request body and must be one of: `movie`, `tv`, or `all`.
  - When providing a specific `tmdb_id`, `media_type` must be either `movie` or `tv` (not `all`) because a TMDB numeric id can correspond to distinct movie and TV records in the database; the API will validate and reject `tmdb_id`+`media_type: all` requests.

Example requests

By TMDB id (required to specify `movie` or `tv`):

```bash
curl -X POST "${API_BASE_URL:-http://localhost:8080}/mcp/knn" \
  -H "Content-Type: application/json" \
  -d '{"tmdb_id": 60803, "k": 10, "results_media_type": "movie", "input_media_type": "movie"}'
```

By free-text (you may use `all` to search across both movies and TV):

```bash
curl -X POST "${API_BASE_URL:-http://localhost:8080}/mcp/knn" \
  -H "Content-Type: application/json" \
  -d '{"text": "mind-bending thriller with a twist ending", "k": 10, "results_media_type": "all"}'
```

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
