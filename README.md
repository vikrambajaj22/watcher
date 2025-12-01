# watcher

> watchu lookin at?

watcher is a personal movie / TV show recommendation app powered by Trakt, TMDB, and OpenAI LLMs that power poster and plot understanding.

## Features
- Syncs your Trakt history automatically
- Recommends content based on your watch patterns
- LLM-powered analysis of scripts or posters

## To Run Locally
```bash
uvicorn app.main:app --reload --port 8080
```

## To Deploy on GCP Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/watcher

gcloud run deploy watcher \
  --image gcr.io/PROJECT_ID/watcher \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars TRAKT_CLIENT_ID=xxx,TRAKT_CLIENT_SECRET=xxx,TRAKT_REDIRECT_URI=xxx,TMDB_API_KEY=xxx,TMDB_READ_ACCESS_TOKEN=xxx,MONGODB_URI=xxx,MONGODB_DB_NAME=watcher,OPENAI_CLIENT_ID=xxx,OPENAI_API_KEY=xxx
```

## Environment (minimal)
Create a `.env` file in the repo root (values below are examples):
```env
TRAKT_CLIENT_ID=your-client-id
TRAKT_CLIENT_SECRET=your-client-secret
TRAKT_REDIRECT_URI=http://127.0.0.1:8080/auth/trakt/callback
TMDB_API_KEY=your-tmdb-key
TMDB_READ_ACCESS_TOKEN=your-tmdb-read-access-token
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=watcher
OPENAI_API_KEY=your-openai-secret-key
```

---

## Embeddings & FAISS (quick setup)
This project stores canonical metadata in Mongo and uses FAISS as the vector store for ANN retrieval. The README below contains the minimal steps to run the embedding pipeline, build the FAISS index, and run the server.

### Prerequisites
- Python 3.10+
- MongoDB reachable at `MONGODB_URI`
- For GPU acceleration (optional): CUDA-enabled GPU with matching `torch` and `faiss-gpu` installs; on Apple Silicon install PyTorch with MPS support.

### Important env vars (additional)
- `EMBED_MODEL_NAME` — SentenceTransformers model (default `multi-qa-mpnet-base-dot-v1`)
- `EMBED_DEVICE` — `cuda`, `mps`, or `cpu` (auto-detected if unset)
- `FAISS_USE_GPU` — `true`/`false` (default `false`) — transfer index to GPU if available
- `FAISS_GPU_ID` — GPU device id (default `0`)
- `FAISS_INDEX_DIR` — path to persist FAISS index files (default `./faiss_index`)

### Install dependencies
```bash
# from repo root
python -m pip install -r requirements.txt
# For GPU, install a CUDA-compatible PyTorch and `faiss-gpu` instead of faiss-cpu
```

### Quick bootstrap (commands)
1) Sync Trakt and TMDB items into Mongo (movies and TV shows):
```bash
python sync_worker.py
```

2Compute embeddings for TMDB items (bootstrap or after a sync):
```bash
python -c "from app.embeddings import index_all_items; print('processed', index_all_items(batch_size=128))"
```

3) Build (rebuild) FAISS index from Mongo embeddings (saves CPU index and attempts GPU transfer):
```bash
python -c "from app.vector_store import rebuild_index; rebuild_index(dim=768, factory='IDMAP,IVF100,Flat')"
```
- Note: use `dim=384` for MiniLM models (e.g., `all-MiniLM-L6-v2`).

4) Start the web app (FAISS index will be loaded into `app.state.faiss_index` at startup):
```bash
uvicorn app.main:app --reload --port 8080
```

### Useful admin endpoints
- Trigger background embedding of a single TMDB item:
```bash
curl -X POST http://127.0.0.1:8080/admin/reindex \
  -H 'Content-Type: application/json' \
  -d '{"id": 12345, "media_type": "movie"}'
```
- Trigger full embedding run (background):
```bash
curl -X POST http://127.0.0.1:8080/admin/reindex \
  -H 'Content-Type: application/json' \
  -d '{"full": true, "batch_size": 128}'
```
- Trigger FAISS rebuild (background):
```bash
curl -X POST http://127.0.0.1:8080/admin/reindex \
  -H 'Content-Type: application/json' \
  -d '{"build_faiss": true, "dim": 768}'
```

### MCP KNN (tooling) example
Query nearest neighbors by text, tmdb_id or embedding vector:
```bash
curl -X POST http://127.0.0.1:8080/mcp/knn \
  -H 'Content-Type: application/json' \
  -d '{"text":"dark sci-fi about memory","k":10}'
```

### Notes & troubleshooting
- If using GPU: ensure `torch` and `faiss-gpu` versions match your CUDA runtime; mismatches cause runtime errors.
- FAISS index files and metadata are saved under `FAISS_INDEX_DIR` — persist this directory across deploys.
- Hybrid search: use Mongo for rich metadata / filters and FAISS for vector retrieval.

---

## Logging
Custom logger ensures clear logs for debugging and observability.

## License
MIT
