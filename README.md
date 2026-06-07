# Watcher

> Personal movie & TV recommendations powered by Trakt watch history and LLMs

Watcher is a personal media discovery application that generates tailored recommendations by analyzing your Trakt watch history with LLMs, then discovering candidates from the TMDB API.

**No persistent TMDB metadata cache. No vector embeddings. No FAISS index.** Just watch history → LLM reasoning → fresh TMDB discover results.

## Features

- 🔐 **Trakt OAuth** — sign in with your Trakt account
- 📺 **Watch history sync** — full Trakt history with runtime data for watch-time stats
- ✨ **LLM-powered recommendations** — taste planner generates a thematic profile; three candidate sources (genre discover, TMDB similar/recommendations, keyword discover) merged via RRF; picker selects and reasons from the taste profile
- 🔍 **TMDB typeahead** — title search with poster previews on Similar and Will I Like pages
- 🎯 **Similar titles** — TMDB `/similar` + `/recommendations` merged via RRF; cross-type mode (movie → TV or TV → movie) via LLM keyword search
- 🤔 **Will I Like?** — LLM scores 0–100% likelihood based on your watch history
- 🎬 **Interactive UI** — React SPA with responsive design

## Quick Start

### Local Dev (React UI + FastAPI)

One command from repo root (loads `.env`, starts API on **8080** and Vite dev on **8501**):

```bash
./start.sh
```

Open **http://localhost:8501**.

**Or manually, in two terminals:**

```bash
# Terminal 1
source .env && export UI_BASE_URL=http://localhost:8501 && uvicorn app.main:app --reload --port 8080

# Terminal 2
cd frontend && npm install && npm run dev
```

### Environment

Create `.env` in repo root:

```env
TRAKT_CLIENT_ID=your-trakt-client-id
TRAKT_CLIENT_SECRET=your-trakt-client-secret
TRAKT_REDIRECT_URI=http://127.0.0.1:8080/auth/trakt/callback
UI_BASE_URL=http://localhost:8501
TMDB_API_KEY=your-tmdb-api-key
OPENAI_API_KEY=your-openai-api-key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=watcher

# Optional: protect admin routes with a secret key
# ADMIN_API_KEY=your-long-random-secret
```

Install dependencies:

```bash
pip install -r requirements.txt
# Optional: dev tools (ruff, isort)
pip install -r requirements-dev.txt
```

## Architecture

### Overview

```
┌─────────────────────────────────────────────┐
│         Watcher React SPA (8501)            │
│  - Watch history page                       │
│  - Recommendations page                     │
│  - Settings / maintenance                   │
└──────────────────┬──────────────────────────┘
                   │ HTTPS
                   ▼
┌─────────────────────────────────────────────┐
│      FastAPI Backend (8080)                 │
│  - /auth/* — Trakt OAuth flow               │
│  - /history — fetch watch history           │
│  - /recommend/* — LLM recommendations       │
│  - /admin/* — maintenance (sync, cache)     │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴─────────┬──────────────────┐
       ▼                 ▼                  ▼
    Trakt API        TMDB API          MongoDB
    (watch history)  (discover,        (watch history
                      details,          only)
                      similar)
```

### Data Flow

1. **User signs in** → Trakt OAuth → store access token
2. **Sync watch history** → fetch from Trakt `/sync/watched/{movie|tv}` → enrich with TMDB metadata (poster, overview, `runtime_minutes` / `episode_runtime_minutes`) → store in MongoDB
3. **Request recommendations** → taste planner + three-source candidate fetch + picker → return top N with reasoning
4. **User browses** → watch history queried from MongoDB (5-minute TTL cache)

**Key design choices:**
- Watch history is stored (needed for personalization, much smaller dataset)
- TMDB metadata is **not** stored — fetch fresh results on each discover call
- No embeddings, no vector search, no FAISS index
- Stateless recommendation logic — each request is independent

### LLM Recommendation Flow

The `/recommend/tmdb/{media_type}` endpoint:

1. **Taste Planner** (`taste_planner_v1.jinja2`):
   - Input: ranked watch history + media type scope
   - Output: `discover_queries` (structured TMDB params) + `taste_summary` (2–3 sentence thematic profile)
   - Model: `gpt-4.1-nano`

2. **Candidate Fetch** (three sources, all merged via RRF):
   - **Genre discover** — planner's `discover_queries` hit TMDB `/discover/{movie|tv}` with genre IDs, date ranges, vote thresholds
   - **Similar/recommendations** — TMDB `/similar` + `/recommendations` seeded from the top 3 ranked history items, merged via RRF
   - **Keyword discover** — `taste_summary` → LLM extracts 6 thematic terms → each resolved to a TMDB keyword ID via `/search/keyword` → per-keyword `/discover` results merged via RRF

3. **Picker** (`tmdb_picker_v1.jinja2`):
   - Input: `taste_summary` + candidate list (with overviews)
   - Output: picks with thematic reasoning (references taste profile themes, never specific watched titles)
   - Model: `gpt-4.1-nano`

4. **Return**: Top N recommendations (id, title, overview, poster, reasoning)

## API Endpoints

### Authentication

- `GET /auth/status` — check if user is logged in
- `GET /auth/login` — redirect to Trakt OAuth
- `GET /auth/trakt/callback` — Trakt OAuth callback (automatic)
- `POST /auth/logout` — clear session

### Watch History

- `GET /history` — fetch user's watch history (supports pagination, filtering)
  - Query params: `limit`, `offset`, `media_type`, etc.

### Recommendations

- `POST /recommend/tmdb/{media_type}` — LLM taste planner + TMDB discover
  - `media_type`: `all`, `movie`, or `tv`
  - Body: `{ "recommend_count": 10 }`
  - Response: list of `Recommendation` objects with reasoning

### Discovery Helpers

- `GET /search?q=&limit=` — **Title typeahead** — proxies TMDB `/search/multi`, returns movies and TV shows with poster thumbnails
- `POST /similar` — **Similar titles** — merges TMDB `/similar` + `/recommendations` via RRF (k=60), TTL-cached 6h. Set `cross_type: true` for opposite-type results (movie → TV or TV → movie): LLM extracts thematic keywords from the title's overview → resolved to TMDB keyword IDs → per-keyword `/discover` merged via RRF
- `POST /will-like` — **Will I Like?** — LLM (`gpt-4.1-nano`) scores 0–100% likelihood from watch history, TTL-cached 1h

### Maintenance (requires `ADMIN_API_KEY` if set)

- `GET /admin/sync/status` — check last sync timestamps
- `POST /admin/sync/trakt` — manually trigger Trakt history sync
- `POST /admin/clear-history-cache` — clear in-process cache

### Health

- `GET /health` — service health check

Full API docs: `http://localhost:8080/docs` (Swagger) or `/redoc` (ReDoc)

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for production setup.

Watcher can run on:
- **GCP Cloud Run** (stateless backend) + static UI host
- **Single VPS** (all-in-one: API + MongoDB on one box)
- **Cloudflare Pages** (free static UI) + VPS backend

## Development Notes

### Project Structure

```
app/
  main.py                       — FastAPI app entry
  api.py                        — API routes
  db.py                         — MongoDB connection & queries
  will_like.py                  — LLM-based Will I Like? logic
  
  auth/
    trakt_auth.py              — Trakt OAuth handling
  
  dao/
    history.py                 — watch history queries
  
  process/
    tmdb_recommendation.py     — LLM + TMDB discover recommender
  
  prompts/
    recommend/
      taste_planner_v1.jinja2  — LLM taste profile generation
      tmdb_picker_v1.jinja2    — LLM recommendation ranking
  
  schemas/
    recommendations/           — Pydantic models for API responses
  
  utils/
    openai_client.py           — OpenAI API wrapper
    prompt_registry.py         — prompt loading
    logger.py                  — logging setup
  
  tmdb_discover.py             — TMDB API integration (discover, similar)
  tmdb_client.py               — low-level TMDB HTTP client

frontend/
  src/
    pages/                     — React pages (history, recommendations, etc.)
    components/                — reusable React components
    api/                       — fetch wrappers for API calls
    App.tsx                    — main app entry
```

### Configuration

Key environment variables:

| Variable | Purpose |
|----------|---------|
| `TRAKT_CLIENT_ID` / `TRAKT_CLIENT_SECRET` | Trakt OAuth credentials |
| `TRAKT_REDIRECT_URI` | Where Trakt redirects after OAuth |
| `UI_BASE_URL` | URL where the SPA is hosted (for Trakt redirects and CORS) |
| `TMDB_API_KEY` | TMDB API key |
| `OPENAI_API_KEY` | OpenAI API key for LLM recommendations |
| `MONGODB_URI` | MongoDB connection string |
| `MONGODB_DB_NAME` | MongoDB database name |
| `ADMIN_API_KEY` | Optional: require this header on admin routes |

### Running Tests

```bash
pytest
```

## Troubleshooting

### OAuth redirect fails

- Check `TRAKT_REDIRECT_URI` matches exactly in both `.env` and Trakt app settings
- Check `UI_BASE_URL` matches the domain the browser sees
- Verify `WATCHER_CORS_ORIGINS` on API includes the UI origin

### Recommendations are slow

- LLM calls (taste planner, picker) can take 2–5 seconds per request
- TMDB discover queries add 1–2 seconds
- Consider caching the taste profile for a user session if patterns emerge

### MongoDB connection fails

- Ensure MongoDB is running: `mongod --dbpath /path/to/data`
- Check `MONGODB_URI` syntax (especially auth)
- Verify database and collections exist

## License

MIT
