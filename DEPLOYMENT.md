# Watcher Local Development

## Quick Start

From repo root (loads `.env`, starts API on **8080** and Vite dev server on **8501**):

```bash
./start.sh
```

Open **http://localhost:8501**

Logs: `backend.log`, `frontend.log`

## Manual Setup (Two Terminals)

```bash
# Terminal 1 — Backend API
source .env
export UI_BASE_URL=http://localhost:8501
uvicorn app.main:app --reload --port 8080

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

## Environment

Create `.env` in repo root:

```env
TRAKT_CLIENT_ID=your-client-id
TRAKT_CLIENT_SECRET=your-client-secret
TRAKT_REDIRECT_URI=http://127.0.0.1:8080/auth/trakt/callback
UI_BASE_URL=http://localhost:8501
TMDB_API_KEY=your-tmdb-key
OPENAI_API_KEY=your-openai-key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=watcher
```

**Optional**: if you set `ADMIN_API_KEY`, create `frontend/.env` or `frontend/.env.local`:

```env
VITE_ADMIN_API_KEY=same-value-as-backend
```

## MongoDB

Local MongoDB (default `mongodb://localhost:27017`):

```bash
# macOS (Homebrew)
brew services start mongodb-community

# Docker
docker run -d -p 27017:27017 --name watcher-mongo mongo:7

# Or install natively on Linux/Windows
```

## Dependencies

```bash
pip install -r requirements.txt
# Optional dev tools
pip install -r requirements-dev.txt
```

Frontend:
```bash
cd frontend
npm install
```

## API

- **Docs**: http://localhost:8080/docs (Swagger)
- **ReDoc**: http://localhost:8080/redoc
- **Health**: http://localhost:8080/health

## Frontend Dev

- **Vite proxy**: `/api-proxy` → `http://127.0.0.1:8080` (no `VITE_API_BASE_URL` needed)
- **Hot reload**: changes auto-refresh

## Admin Routes

If `ADMIN_API_KEY` is set, protected routes require the header:

```bash
curl -H "X-API-Key: $ADMIN_API_KEY" http://localhost:8080/admin/sync/status
```

The React app injects `VITE_ADMIN_API_KEY` automatically (see `frontend/src/api/client.ts`).

## Build for Production

Frontend:
```bash
cd frontend
VITE_API_BASE_URL="https://your-api.example.com" \
VITE_ADMIN_API_KEY="your-admin-key" \
npm run build
```

Output: `frontend/dist/` — deploy as static files.

Backend: Docker image or direct Python:
```bash
# Docker
docker build -t watcher-api .
docker run -p 8080:8080 --env-file .env watcher-api

# Or systemd/supervisor for persistence
```

## Troubleshooting

**Trakt OAuth fails**: Ensure `TRAKT_REDIRECT_URI` in `.env` matches your app settings exactly.

**CORS errors**: Set `WATCHER_CORS_ORIGINS` when deploying frontend elsewhere (default allows `localhost:8501`).

**MongoDB connection refused**: Check MongoDB is running on port 27017.

**API slow**: LLM calls (taste planner, picker) add 2–5 seconds per recommendation. Check OpenAI quota/rate limits.

---

For production deployment notes and legacy architecture, see:
- [README.md](README.md) — project overview
- [LEGACY.md](LEGACY.md) — old FAISS + TMDB metadata approach, GCP deployment
- [DEVELOPMENT.md](DEVELOPMENT.md) — detailed API routes and admin key reference
