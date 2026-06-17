# Watcher

> Personal movie & TV recommendations powered by Trakt watch history and LLMs

Watcher is a personal media discovery application that generates tailored recommendations by analyzing your Trakt watch history with LLMs, then discovering candidates from the TMDB API.

**No persistent TMDB metadata cache. No vector embeddings. No FAISS index.** Just watch history → LLM reasoning → fresh TMDB discover results.

## Features

- 🔐 **Trakt OAuth** — sign in with your Trakt account; account switch clears all cached state automatically
- 📺 **Watch history sync** — full Trakt history with runtime data, genres, and poster metadata
- 🔖 **Watchlist** — synced bidirectionally with Trakt custom lists (`TRAKT_MOVIE_LIST_ID` / `TRAKT_TV_LIST_ID`); add/remove from any results page via bookmark toggle; genre filter tags; home page counter widget; items already in history auto-cleared on sync
- ✨ **LLM recommendations** — taste planner + three-source TMDB candidate fetch (genre discover, similar/recommendations, keyword discover) merged via RRF; picker reasons from your taste profile; optional genre filter (names → TMDB genre IDs, hard-filters the candidate pool)
- 🧠 **Taste Profile** — LLM-generated snapshot: signature, summary, top genres, recurring themes, avoid list; cached 1h and shared across features
- 🗂️ **Watch History UI** — filter by genre and watch year (both dropdowns derived from history data); sort by latest/earliest watched, release year, title, or engagement; genre tags shown inline; open in TMDB links
- 🔍 **Discover by description** — natural language → LLM extracts structured filters (genres, cast, keywords, year range) → TMDB Discover; optional media-type selector (Movies/TV/Both) overrides LLM inference; excludes already-watched titles
- 👤 **Actor Search** — find every title in your history featuring a specific actor or director, with character roles
- 💬 **Chat** — conversational agent backed by a LangGraph `StateGraph` (agent ↔ ToolNode loop with a conditional self-correction `verify` node); routing lives in self-describing tool docstrings, not a prescriptive prompt; the user's taste profile is injected into context every turn (TTL-cached) so the agent personalises and picks from the watchlist itself without an extra lookup; 14 tools: recommendations (optional genre filter), similar (incl. cross-type movie↔TV), similar-in-history, will-like, description search, cast lookup, title details, taste profile, person lookup, actor history, watch history, watchlist read/add/remove (the similar tools share `compute_similar` with the `/similar` route — no duplicated logic); streams tool status + response via SSE; intermediate lookups shown as collapsible summaries, final results as compact cards; renders markdown in responses; answers follow-up questions about returned titles (e.g. "which of these have actors I'd recognise?") and re-tries once if its answer misses the question; models configurable via `CHAT_MODEL` / `CHAT_VERIFY_MODEL`; works on mobile (HTTPS-free UUID fallback, fixed sticky input)
- 🎯 **Similar titles** — TMDB `/similar` + `/recommendations` merged via RRF; cross-type mode (movie → TV or vice versa) via LLM keyword search; three modes: **By Title** (any title → TMDB similar), **From History** (pick a watched title as seed), **To History** (search any title → find matches from your history that TMDB considers similar)
- 🤔 **Will I Like?** — LLM scores 0–100% likelihood based on your taste profile
- 🎬 **Interactive UI** — React SPA with responsive design, 404 page, and graceful poster fallbacks

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
```

> `requirements.txt` includes `langgraph`, `langchain-openai`, and `langchain-core` for the Chat feature.

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
2. **Sync watch history** → fetch from Trakt `/sync/watched/{movie|tv}` → enrich with TMDB metadata (poster, overview, `runtime_minutes` / `episode_runtime_minutes`) → store in MongoDB; auto-clears watchlist items that now appear in history
3. **Request recommendations** → taste planner + three-source candidate fetch + picker → return top N with reasoning
4. **User browses** → watch history queried from MongoDB (5-minute TTL cache)
5. **Watchlist** → local MongoDB cache synced bidirectionally with two Trakt custom lists; mutations write to Trakt first, then update cache; pull sync on demand (paginated, up to 1000 items/page)

**Key design choices:**
- Watch history is stored (needed for personalization, much smaller dataset)
- TMDB metadata is **not** stored — fetch fresh results on each discover call
- No embeddings, no vector search, no FAISS index
- Stateless recommendation logic — each request is independent

### Taste Profile

`GET /taste-profile` (and `app/taste_profile.py`) generates a structured viewer profile from watch history using one LLM call, cached for 1 hour:

- **Signature** — punchy 5–10 word label
- **Summary** — 2–3 sentence taste description
- **Genres / Themes / Avoid** — up to 5 items each

This profile is shared: Will I Like? uses it as context instead of raw history, and the Recommendations picker uses it to select and reason about candidates. The Taste Profile page in the UI displays it directly.

### LLM Recommendation Flow

The `/recommend/tmdb/{media_type}` endpoint:

1. **Taste Planner** (`taste_planner_v1.jinja2`):
   - Input: ranked watch history + media type scope
   - Output: `discover_queries` (structured TMDB params)
   - Model: `gpt-4.1-nano`

2. **Candidate Fetch** (three sources, all merged via RRF):
   - **Genre discover** — planner's `discover_queries` hit TMDB `/discover/{movie|tv}` with genre IDs, date ranges, vote thresholds
   - **Similar/recommendations** — TMDB `/similar` + `/recommendations` seeded from the top 3 ranked history items, merged via RRF
   - **Keyword discover** — taste profile text → LLM extracts 6 thematic terms → each resolved to a TMDB keyword ID via `/search/keyword` → per-keyword `/discover` results merged via RRF

3. **Picker** (`tmdb_picker_v1.jinja2`):
   - Input: shared taste profile text + candidate list (with overviews)
   - Output: picks with thematic reasoning
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

### Watchlist

- `GET /watchlist` — return cached watchlist (`?media_type=movie|tv`)
- `POST /watchlist` — add item; resolves TMDB metadata, writes to Trakt, updates cache; body: `{ tmdb_id, media_type, title?, poster_path?, overview?, release_date? }`
- `DELETE /watchlist/{tmdb_id}?media_type=` — remove from Trakt and local cache
- `POST /watchlist/sync` — pull both Trakt lists, refresh cache; returns `{ added, removed }`

### Recommendations

- `POST /recommend/tmdb/{media_type}` — LLM taste planner + TMDB discover
  - `media_type`: `all`, `movie`, or `tv`
  - Body: `{ "recommend_count": 10, "genre_hint": ["comedy", "thriller"] }` (`genre_hint` optional — genre names resolved to TMDB IDs and hard-filtered)
  - Response: list of `Recommendation` objects with reasoning

### Discovery Helpers

- `GET /search?q=&limit=` — **Title typeahead** — proxies TMDB `/search/multi`, returns movies and TV shows with poster thumbnails
- `GET /search/person?q=&limit=` — **Person typeahead** — proxies TMDB `/search/person`, returns people with profile photo and known-for titles
- `POST /similar` — **Similar titles** — merges TMDB `/similar` + `/recommendations` via RRF (k=60), TTL-cached 6h. Set `cross_type: true` for opposite-type results (movie → TV or TV → movie): LLM extracts thematic keywords from the title's overview → resolved to TMDB keyword IDs → per-keyword `/discover` merged via RRF. Set `filter_to_history: true` to restrict results to titles already in your watch history
- `GET /taste-profile` — **Taste Profile** — LLM-generated viewer profile (signature, summary, genres, themes, avoid), TTL-cached 1h
- `POST /will-like` — **Will I Like?** — LLM scores 0–100% likelihood using the shared taste profile, TTL-cached 1h
- `POST /discover/describe` — **Discover by description** — natural language → LLM filter extraction → TMDB Discover; optional `media_type` body field (`movie`/`tv`/`both`) overrides LLM inference; genres joined with OR for broader results; excludes watched titles, TTL-cached 1h
- `GET /history/actor?name=` — **Actor search** — TMDB person lookup → combined credits cross-referenced with watch history
- `POST /chat` — **Chat** — LangGraph SSE stream; accepts `{ thread_id, message }`; streams `tool_start`, `tool_result`, `message`, `done` events; agent model `CHAT_MODEL` (default `gpt-4.1-mini`), self-correction judge `CHAT_VERIFY_MODEL` (default `gpt-4.1-nano`)

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
  main.py                       — FastAPI app entry, CORS config
  api.py                        — all API routes
  db.py                         — MongoDB connection
  will_like.py                  — LLM Will I Like? (Pydantic response, TTL cache)
  taste_profile.py              — shared taste profile (LLM, 1h cache, Pydantic)
  similar.py                    — shared compute_similar() (used by /similar route + chat tools)
  actor_history.py              — actor/director search against watch history
  chat.py                       — LangGraph chat agent (agent ↔ ToolNode + conditional verify node, taste context, SSE)
  trakt_sync.py                 — Trakt → TMDB enrich → MongoDB sync; auto-clears watched items from watchlist
  watchlist_sync.py             — Trakt custom list pull (paginated) + push add/remove; TMDB enrichment

  auth/
    trakt_auth.py              — Trakt OAuth flow

  config/
    settings.py                — pydantic-settings env config

  dao/
    history.py                 — watch history queries (5-min TTL cache)
    watchlist.py               — watchlist cache CRUD (MongoDB `watchlist` collection)

  process/
    tmdb_recommendation.py     — taste planner + candidate fetch + picker
    describe_discover.py       — natural language → TMDB Discover (1h cache)

  prompts/                       — Jinja2 prompt templates, loaded via PromptRegistry
    chat/
      system_v1.jinja2          — legacy prescriptive prompt (kept for rollback)
      system_v2.jinja2          — slim prompt; taste context + behaviour principles (active)
      verify_v1.jinja2          — self-correction judge prompt
    recommend/
      taste_planner_v1.jinja2
      tmdb_picker_v1.jinja2
    discover/
      extract_filters_v1.jinja2
    will_like/
      predict_v1.jinja2
    taste_profile/
      generate_v1.jinja2

  schemas/
    api.py                     — all API Pydantic models
    recommendations/           — recommendation-specific schemas

  utils/
    openai_client.py
    prompt_registry.py
    logger.py

  tmdb_discover.py             — TMDB discover, similar, recommendations, RRF merge
  tmdb_client.py               — low-level TMDB HTTP helpers

frontend/
  public/
    404.png                    — poster fallback and 404 page image
  src/
    pages/                     — History, Watchlist, Recommend, WillLike, Similar, Discover,
                                  Actor, Chat, Taste, Admin, Home, NotFound
    components/                — Layout, MediaCard, SearchTypeahead, AiBlurb, SimilarResultRow, etc.
    api/                       — typed fetch wrappers (watcher.ts, client.ts)
    lib/poster.ts              — TMDB poster URL helper with /404.png fallback
    contexts/
      AuthContext.tsx          — auth state provider
      WatchlistContext.tsx     — watchlist state, optimistic toggle, sync
    App.tsx                    — routes
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
| `ADMIN_API_KEY` | Optional: require this header on admin/recommend/chat routes |
| `CHAT_MODEL` | Optional: chat agent model (default `gpt-4.1-mini`) |
| `CHAT_VERIFY_MODEL` | Optional: chat self-correction judge model (default `gpt-4.1-nano`) |
| `TRAKT_MOVIE_LIST_ID` | Trakt custom list slug for movie watchlist (default: `watchlist-movies`) |
| `TRAKT_TV_LIST_ID` | Trakt custom list slug for TV watchlist (default: `watchlist-shows`) |
| `WATCHER_CORS_ORIGINS` | Comma-separated allowed origins (default: localhost:8501) |
| `WATCHER_CORS_ORIGIN_REGEX` | Optional regex for additional origins (e.g. LAN IP for mobile) |

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
- The taste profile is cached for 1 hour — the first Will I Like or Recommendations request after startup pays the cost; subsequent ones reuse it

### MongoDB connection fails

- Ensure MongoDB is running: `mongod --dbpath /path/to/data`
- Check `MONGODB_URI` syntax (especially auth)
- Verify database and collections exist

## License

MIT
