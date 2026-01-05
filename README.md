# watcher

> Personal movie & TV recommendations built on Trakt, TMDB, FAISS, and LLMs

Watcher is a personal media discovery and recommendation application that integrates Trakt watch history, TMDB metadata, and LLM-powered reasoning. It uses FAISS for vector similarity and stores embeddings in compact sidecar files and the FAISS index for fast nearest-neighbor search.

## Features
- 🔐 Trakt OAuth authentication
- 📺 Browse and sync watch history
- 📊 Visual Explorer — clustered, interactive visualization of history
- ✨ AI-powered recommendations with human-readable reasoning
- 🔍 Similar-item search by TMDB id or free-text
- ❓Will I Like It? — personalized LLM predictions on specific titles
- ⚙️ Admin panel for sync, embedding generation, FAISS management, and cache control

## Quick Start

Start the whole project:

```bash
./start.sh
```

Access:
- Streamlit UI: http://localhost:8501
- Backend API docs: http://localhost:8080/docs

Or start components separately:

```bash
# Backend
uvicorn app.main:app --reload --port 8080

# Frontend (Streamlit)
./run_streamlit.sh
```

### Environment

Create a `.env` in the repo root with the following values (examples):

```env
TRAKT_CLIENT_ID=your-client-id
TRAKT_CLIENT_SECRET=your-client-secret
TRAKT_REDIRECT_URI=http://127.0.0.1:8080/auth/trakt/callback
TMDB_API_KEY=your-tmdb-key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=watcher
OPENAI_API_KEY=your-openai-key
FAISS_INDEX_DIR=./faiss_index
```

Install dependencies

```bash
pip install -r requirements.txt
```

### Overview: Embeddings and FAISS

- Embeddings are computed from TMDB metadata and are used only for nearest-neighbor search and candidate generation for LLM reasoning. Embedding vectors are persisted to sidecar files in the FAISS index directory (`labels.npy` and `vecs.npy`) and are also stored as labels inside the FAISS index. Embedding vectors are not written into MongoDB documents.

- FAISS index files (for example `tmdb.index`) and sidecar files form the canonical on-disk representation of computed vectors and labels. A JSON `sidecar_meta.json` lives alongside them and records the embedding model name, timestamp, dims, and number of vectors.

- The system supports two embedding workflows:
  1. Batched / full-index rebuilding: compute embeddings for metadata records and build a FAISS index. This produces the on-disk index file and sidecars. Use this for initial population or full reindex.
  2. Incremental single-item upserts: attempt to compute and insert/update a single vector in the sidecars and (if supported) update the FAISS index in-place. If the local FAISS build does not support in-place updates, a full rebuild is scheduled instead.

#### TMDB Sync Behavior (export-first, then incremental)

The sync logic prioritizes high-coverage export files published by TMDB for the initial ingest and uses the changes endpoints for incremental updates.

1. Initial ingest
   - The sync attempts to fetch TMDB daily export files (gzipped newline-delimited JSON of IDs) and streams them for a complete ID list. This is the recommended method for an initial, full population because it avoids the discover-page limits and provides broad coverage.
   - IDs are batched and the app fetches full metadata per ID (details, credits, keywords) and upserts them into MongoDB.
   - Embedding computation during the initial ingest is optional and controlled by the `embed_updated` / `embed_updated` flag when calling the TMDB sync routines; it is recommended to skip embeddings during the initial large ingest (embed_updated=false) and run a separate embedding/index pass afterwards.

2. Fallback / discover
   - If no suitable export file is available within the configured lookback window, the sync falls back to the `/discover/{movie|tv}` route with a safe cap on pages.

3. Incremental updates
   - After initial population the sync uses `/movie/changes` and `/tv/changes` to fetch deltas and apply metadata updates.
   - The sync exposes parameters to control whether embeddings are computed during the incremental pass (embed_updated). For typical usage, compute embeddings separately so metadata sync and embedding provider usage are rate-limited independently.

#### Embedding Generation & Index Lifecycle (recommended workflows)

- Full rebuild path (recommended for initial indexing):
  - Compute embeddings for each metadata record.
  - Build and persist a FAISS index file (CPU index) and save sidecars: `labels.npy`, `vecs.npy`, and `sidecar_meta.json`.
  - Optionally transfer the index to GPU resources if configured.

- Incremental upsert path (fast for single items):
  - Compute an embedding for a single item and try to update sidecars and FAISS index in-place.
  - If the local FAISS implementation doesn't support in-place add/remove, the operation returns a rebuild-required response and a full rebuild can be scheduled.

- Cache & process behavior:
  - Each backend process keeps an in-memory FAISS index cached after the first load to avoid repeated disk reads.
  - The admin UI and API provide controls to clear the in-process cache when an out-of-process rebuild or external index write occurs.

#### Admin and API notes

Core endpoints used by the UI and scripts:

- `GET /admin/faiss/status` — reports whether an index/sidecars are present and returns the `sidecar_meta` and a `cached` boolean indicating whether this process has the index loaded in memory.
- `POST /admin/faiss/clear-cache` — clears the FAISS index cache in the current process.
- `POST /admin/faiss/rebuild` — spawn a detached CLI process to rebuild the FAISS index (useful for large, long-running rebuilds).
- `POST /admin/embed/full` — schedule a background full rebuild that computes embeddings and writes sidecars/index.
- `POST /admin/embed/item` — attempt an incremental upsert for a single item; if an in-place update is not possible, this endpoint schedules a background rebuild.
- `POST /admin/faiss/upsert-item` — lower-level endpoint that attempts an in-place upsert and returns its status.

Use `localhost:8080/redoc` to explore the full API.

UI: Streamlit admin pages let you:
- Trigger full rebuilds, per-item upserts, and detached rebuilds
- Inspect `sidecar_meta` (model, dims, num_vectors)
- See whether the current process has the index cached
- Clear the FAISS cache for this backend process

#### Practical Recommendations

- Initial population: run TMDB sync with `embed_updated=false` (metadata-only) to avoid concurrently fetching TMDB metadata and computing embeddings:

```bash
python - <<'PY'
from app.tmdb_sync import sync_tmdb
# full export-based ingest (recommended), but skip embeddings in this large job
sync_tmdb('movie', full_sync=True, embed_updated=False)
PY
```

- Once metadata is populated, create embeddings and build the FAISS index:

```bash
# Full rebuild from metadata (computes embeddings and produces sidecars + FAISS index)
python -m app.faiss_rebuild_cli --dim 384 --factory "IDMap,IVF100,Flat"
```

Or trigger from the API/UI (admin panel) using the "Rebuild FAISS" controls.

- Small updates: use single-item upsert flows from the UI or `POST /admin/embed/item` to avoid full rebuilds when possible.

#### Operational Notes

- Embedding vectors are not stored inside MongoDB documents. The canonical storage for embeddings is the FAISS index and the sidecar files. This keeps metadata fast and small while keeping vectors available for nearest-neighbor search and export.

- Module-level (process) cache: the FAISS index is cached per Python process. If you run multiple workers, each worker caches its own copy. Clearing the cache via the API affects only the process that receives the request.

- If an external tool updates the on-disk index, clear the in-process cache (or restart the process) so the backend reloads the fresh index on next use.

#### Examples & Troubleshooting

- Check FAISS status:

```bash
curl -sS http://localhost:8080/admin/faiss/status | jq .
```

- Clear the FAISS cache for the current process:

```bash
curl -X POST http://localhost:8080/admin/faiss/clear-cache
```

- Trigger an incremental upsert for single item (UI-friendly):

```bash
curl -X POST http://localhost:8080/admin/embed/item -H 'Content-Type: application/json' -d '{"id": 550, "media_type": "movie"}'
```

- Trigger a detached FAISS rebuild:

```bash
curl -X POST http://localhost:8080/admin/faiss/rebuild -H 'Content-Type: application/json' -d '{"dim": 384, "factory": "IDMap,IVF100,Flat"}'
```

#### Development Notes

- During a very large initial ingest avoid computing embeddings inline to reduce load on embedding providers and TMDB. After metadata is stable, run embedding/indexing passes with throttling and batching.

## Deployment
TBD
License

MIT

