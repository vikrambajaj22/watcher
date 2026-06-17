# Refactor: Remove Embeddings, Use TMDB API for Similar Items

## Overview
Remove FAISS + embeddings entirely. Use TMDB's native `/similar` and `/recommendations` endpoints instead.

## Changes Required

### 1. Remove Files (Can Delete)
- `app/embeddings.py` — embedding computation logic
- `app/faiss_index.py` — FAISS index loading/caching
- `app/faiss_rebuild_cli.py` — FAISS CLI rebuild tool
- `app/process/recommendation.py` (old version) — KNN logic
- `app/tmdb_sync.py` — no longer needed (metadata not cached)
- `sync_worker.py` — periodic sync not needed
- `tools/mongo_local_dump_export.py` — TMDB metadata export (not used)
- `frontend/src/pages/RecommendComparePage.tsx` — compare recommendations UI (not needed)
- `frontend/src/pages/VisualPage.tsx` — t-SNE visual explorer (depends on embeddings)
- `frontend/src/components/ClusterChart.tsx` — cluster chart for the visual explorer

### 2. Update API Endpoints

#### `/mcp/knn` (Similar Items)
**Current**: FAISS vector similarity search  
**New**: Use TMDB `/movie/{id}/similar` or `/tv/{id}/similar`

```python
# Old: compute embedding, KNN search in FAISS, fetch metadata from MongoDB
# New: call TMDB similar endpoint directly, return results

@router.post("/mcp/knn", response_model=SimilarResponse)
def similar_items(payload: SimilarRequest):
    """Find similar movies/TV shows using TMDB's similarity API."""
    # If user provided title: resolve to TMDB ID via discover or search
    # Call TMDB /movie/{id}/similar or /tv/{id}/similar
    # Return paginated results (title, poster, overview, etc.)
```

#### `/recommend/{media_type}` (Old Endpoint)
**Current**: Uses FAISS KNN + LLM  
**Status**: **REMOVE** — keep only `/recommend/tmdb/{media_type}` (new LLM approach)  
**Note**: This endpoint is what the compare page used. Removing it also removes compare functionality.

#### `/visualize/clusters` (Visual Explorer)
**Current**: t-SNE + K-means over watch-history embeddings  
**Status**: **REMOVE** — the t-SNE visual explorer depends on embeddings, which are gone. Remove the endpoint along with the `VisualPage` and `ClusterChart` frontend.

#### `/mcp/will-like` (Will I Like?)
**Current**: Embedding similarity threshold (0.65)  
**New**: Reimplement with LLM-only logic (no embeddings)

**Approach**:
- User provides: TMDB ID, title, and media type
- Fetch item metadata from TMDB (overview, genres, cast)
- Call LLM: "Based on the user's watch history [formatted list], will they like [item details]? Respond with JSON: {will_like: bool, confidence: 0-1, reasoning: str}"
- Return structured response with prediction + reasoning

**Implementation**:
- Use same LLM call pattern as taste planner
- Can reuse taste planner prompts/logic
- No FAISS or embeddings needed
- Response format: `{will_like: bool, score: float, explanation: str, item: {...}}`

### 3. Remove Admin Endpoints
- `GET /admin/faiss/status` — remove
- `POST /admin/faiss/clear-cache` — remove
- `POST /admin/faiss/rebuild` — remove
- `POST /admin/embed/full` — remove
- `POST /admin/embed/item` — remove
- `POST /admin/faiss/upsert-item` — remove

### 4. Update Frontend

#### App.tsx
- Remove imports: `RecommendComparePage`, `VisualPage`
- Remove routes: `<Route path="/recommend-compare" element={<RecommendComparePage />} />` and `<Route path="/visual" element={<VisualPage />} />`
- Remove any navigation links to `/recommend-compare` and `/visual`

#### Maintenance Page (`frontend/src/pages/MaintenancePage.tsx` or equivalent)
- Remove "Embeddings & Index" tab
- Remove FAISS rebuild controls
- Remove FAISS status display
- Keep sync controls

#### Similar Items Component (`frontend/src/components/SimilarResultRow.tsx`)
- Update API call to new `/mcp/knn` signature
- No changes needed if response format stays similar

### 5. Database Changes

#### Collections to Drop (no longer used)
- `tmdb_metadata` — cached TMDB data (~1.3 GB)
- `tmdb_failures` — failed fetch tracking
- `sync_meta` (partial) — remove TMDB sync job records (keep Trakt-related)

#### Keep
- `watch_history` — user's watch history

**Action**: Drop immediately (after verifying backup exists)
```bash
# Remove old collections
mongosh <<EOF
use watcher
db.tmdb_metadata.drop()
db.tmdb_failures.drop()
# Clear old TMDB sync job records
db.sync_meta.deleteMany({ "key": { \$regex: "^tmdb_sync" } })
EOF
```

### 6. Environment Variables

#### Remove (no longer needed)
- `FAISS_SOURCE`
- `FAISS_INDEX_DIR`
- `FAISS_BUCKET`
- `FAISS_PREFIX`
- `FAISS_MOUNT_PATH`
- `FAISS_PERSIST_DIR`
- `FAISS_USE_GPU`
- `EMBED_DEVICE`
- `EMBEDDING_MODEL` (if exists)

#### Keep
- `MONGODB_URI`, `MONGODB_DB_NAME`
- `OPENAI_API_KEY`, `TMDB_API_KEY`
- `TRAKT_CLIENT_ID`, `TRAKT_CLIENT_SECRET`, `TRAKT_REDIRECT_URI`
- `UI_BASE_URL`, `WATCHER_CORS_ORIGINS`
- `ADMIN_API_KEY`

### 7. Update Documentation

#### README.md
- Remove embeddings/FAISS section
- Update "How It Works" to show new flow
- Remove "Overview: Embeddings and FAISS" section
- Update admin endpoints reference

#### DEPLOYMENT.md
- Remove FAISS environment variable examples
- Simplify to just API + MongoDB + Trakt

#### DEVELOPMENT.md
- Remove FAISS admin routes from API reference
- Update `/mcp/knn` route documentation
- Remove admin key requirement from FAISS endpoints

#### LEGACY.md
- Already documents this; no changes needed

### 8. Code Cleanup

#### app/api.py
- Remove imports: `embeddings`, `faiss_index`, `faiss_rebuild_cli`
- Remove FAISS-related utility functions
- Update endpoint signatures as needed

#### app/main.py
- Remove FAISS initialization from startup (if present)

#### Imports in other files
- Search for `from app.embeddings import`, `from app.faiss_import`
- Remove unused imports

#### app/schemas/
- May simplify response schemas if they have FAISS-specific fields

### 9. Tests (if any)
- Remove/update tests that test FAISS functionality
- Update tests that call removed endpoints
- Add tests for new TMDB similar API integration

### 10. Docker & Dependencies

#### requirements.txt
- Remove `faiss-cpu` or `faiss-gpu`
- Remove `sentence-transformers`
- Likely can remove: `scikit-learn` (was for t-SNE)
- Keep: `fastapi`, `pymongo`, `pydantic`, `httpx`, `openai`, `python-dotenv`, etc.

#### Dockerfile
- Simplify (no FAISS/embedding model downloads)
- Smaller base image possible
- Faster cold starts

### 11. Git Cleanup
- Delete obsolete files (step 1)
- Consider squashing commits related to old approach if not yet pushed
- Tag current state before major changes

### 12. UI Modernization

With the visual explorer and compare pages gone, the surviving surface is small enough to refresh in one pass. Goal: a cohesive, responsive, accessible SPA — no half-migrated styling.

#### Design System & Tokens
- Centralize design tokens (colors, spacing, radii, typography, shadows) as CSS custom properties in one `theme.css` (or Tailwind config). No more per-component hardcoded hex/px.
- Define **light + dark** palettes; respect `prefers-color-scheme` and add a manual toggle persisted to `localStorage`.
- Pick one type scale and one spacing scale; apply consistently.

#### Component Library
- Standardize on a single approach. Options (pick one):
  - **Tailwind CSS** + headless primitives (Radix UI / Headless UI) — utility-first, small runtime
  - **shadcn/ui** — copy-in components on Radix + Tailwind, full control
  - A styled component kit (e.g., Mantine / Chakra) — fastest, heavier bundle
- Build/adopt shared primitives: `Button`, `Card`, `Input`, `Select`, `Spinner`, `Toast`, `Modal`, `Tabs`, `EmptyState`, `Skeleton`.
- Replace ad-hoc loading text with **skeleton loaders** for history/recommendation/similar cards.

#### Layout & Navigation
- Responsive app shell: top bar + collapsible sidebar (or bottom tab bar on mobile).
- Consistent page container, max-width, and section spacing across `/`, `/history`, `/recommend`, `/will-like`, `/similar`, `/admin`.
- Persistent nav with active-route highlighting; mobile hamburger/drawer.

#### Media Cards
- Unified `MediaCard` used by recommendations, similar, and history (poster, title, year, rating, overview-on-hover/expand).
- Lazy-load posters (`loading="lazy"`), graceful poster-missing fallback, consistent aspect ratio.
- Copy-to-clipboard TMDB link action (ties into Future Enhancements).

#### Feedback & States
- Toast/snackbar system for sync results, errors, and admin actions (replace inline alerts).
- First-class **empty states** (no history yet, no results) and **error states** (API down, not logged in) with clear CTAs.
- Disabled/loading button states during in-flight requests.

#### Accessibility
- Keyboard-navigable nav, modals, and menus; visible focus rings.
- Proper labels/`aria-*` on inputs and icon-only buttons; sufficient color contrast in both themes.
- Honor `prefers-reduced-motion` for any transitions.

#### Performance & Polish
- Route-level code splitting (`React.lazy` + `Suspense`) so the heavy chart deps are no longer in the main bundle (Recharts is removed with the visual explorer — confirm it's dropped from `package.json`).
- Subtle, consistent transitions (page/route, card hover) via a single motion convention.

#### Files (indicative)
- `frontend/src/theme.css` (or `tailwind.config.ts`) — tokens
- `frontend/src/components/ui/` — shared primitives
- `frontend/src/components/MediaCard.tsx` — unified card
- `frontend/src/components/AppShell.tsx` — layout/nav
- Refactor existing pages to consume the above; delete bespoke one-off styles

#### Scope Guardrail
This is a visual/UX refresh, **not** a feature change. Keep API calls and data flow identical; only the presentation layer changes.

### 13. Route & Naming Cleanup (`/mcp/*` is not MCP)

The `/mcp/*` routes and the internal `mcp_*` naming are misleading: this code uses OpenAI **function/tool calling**, not the Model Context Protocol. `mcp_knn` is doubly wrong — `knn` is a FAISS-era name (k-nearest-neighbors) even though the endpoint now just hits TMDB `/similar`. Rename for honesty before adding the chat feature (enhancement #3), which *could* later expose a real MCP server.

#### API routes
- `POST /mcp/knn` → `POST /similar`
- `POST /mcp/will-like` → `POST /will-like`
- Single-user app, so a clean cut-over is fine; optionally keep thin deprecated aliases (308 redirect) for one release.

#### Backend internals
- `app/mcp_will_like.py` → `app/will_like.py` (`compute_will_like` is already well-named — keep it)
- `app/utils/llm_orchestrator.py`: `call_mcp_knn` → `call_similar`; `call_model_with_mcp_function` → `call_model_with_tool`; `mcp_function_schema` → `tool_schema`; update the `match` cases (`"mcp_knn"` → `"similar"`)
- `app/api.py`: update imports, route decorators, and handler names (`mcp_knn` → `similar`, `mcp_will_like` → `will_like`)
- `app/schemas/api.py`: update docstrings that reference `/mcp/*`
- Any tool-schema JSON: rename the tool `mcp_knn` → `similar`

#### Frontend
- `frontend/src/api/*`: update fetch paths `/mcp/knn` → `/similar`, `/mcp/will-like` → `/will-like`
- Rename any `mcp`-flavored component/variable names

#### Docs
- `README.md` and `DEVELOPMENT.md` already document the target names (`/similar`, `/will-like`) — the code rename above brings the implementation in line with them. Update any lingering `/mcp/*` mentions that surface during the rename.

---

## Implementation Order

1. ✅ **Step 1** — Create feature branch
2. ✅ **Step 2** — `/similar` endpoint using TMDB similar/recommendations API (TTL-cached)
3. ✅ **Step 3** — Removed all admin FAISS endpoints
4. ✅ **Step 4** — `/will-like` reimplemented with LLM-only logic (TTL-cached), no embeddings
5. ✅ **Step 5** — Frontend updated: removed visual/compare pages, updated admin, simplified similar page
6. ✅ **Step 6** — Deleted: embeddings.py, faiss_index.py, faiss_rebuild_cli.py, mcp_will_like.py, process/recommendation.py, tmdb_sync.py, llm_orchestrator.py, knn_utils.py, vector_store.py, sync_worker.py, tools/mcp_knn.json
7. ✅ **Step 7** — Cleaned imports; trakt_sync now stores poster_path/overview; history DAO no longer queries tmdb_metadata
8. ✅ **Step 8** — requirements.txt stripped of faiss-cpu, sentence-transformers, scikit-learn, torch, etc.
9. ✅ **Step 9** — `tmdb_failures` dropped; `tmdb_metadata` kept for future use (not queried by app)
10. ✅ **Step 10** — Route & naming cleanup: `/mcp/knn` → `/similar`, `/mcp/will-like` → `/will-like`
11. ✅ **Step 11** — UI modernization: Tailwind CSS v4, active nav, modern cards, dead CSS removed, recharts removed
12. ✅ **Step 12** — Documentation updated (README, DEVELOPMENT, this file)
13. **Step 13** — Test locally, commit

---

## Decisions

✅ **1. Reimplement `/mcp/will-like` with LLM logic** (no embeddings)
   - Use LLM to predict based on watch history + item details
   - Same approach as taste planner

✅ **2. Remove old `/recommend/{media_type}` endpoint**
   - Single endpoint: `/recommend/tmdb/{media_type}` (new LLM approach only)

✅ **3. Drop `tmdb_metadata` from MongoDB immediately**
   - Saves ~1.3 GB storage
   - No longer needed

---

## Future Enhancements

1. **Find Titles by Description** ✅

   A natural-language search endpoint that lets the user describe what they want to watch and returns matching titles. Goes beyond keyword search by inferring structured filters from free text.

   **Backend**
   - New endpoint: `POST /discover/describe` — body: `{ query: str, limit: int }`
   - LLM extracts structured filters from the query: `media_type`, `genres`, `cast` (actor/director names), `keywords`, `year_range`
   - Fan out to TMDB Discover API with extracted filters + keyword search
   - Optionally cross-reference with watch history to surface unwatched titles
   - Return ranked results with a short explanation of why each matches

   **Frontend**
   - New `/discover` route with a free-text input
   - "What do you want to watch tonight?" prompt style
   - Results as `MediaCard` grid with match reasoning
   - Example queries: "90s sci-fi with practical effects", "feel-good comedies like The Grand Budapest Hotel", "anything with Cate Blanchett"

   **Notes**
   - Filter inference is the core value: `media_type`, `with_genres`, `with_cast`, `with_crew`, `primary_release_year`, `with_keywords`
   - TMDB `/discover/movie` and `/discover/tv` support all of these natively
   - Cast/crew name → TMDB person ID lookup needed before Discover call
   - Free-text remainder (mood, style) maps to `with_keywords` or a secondary `/search/keyword` call

2. **Watch History UI Improvements** ✅
   - Genre/year filters (year range inputs + genre text filter, client-side)
   - Copy-to-clipboard button for TMDB URLs on each row
   - Sort by release year added; genres stored during Trakt sync and shown inline

3. **"Where Have I Seen This Actor?" Feature** ✅
   - `GET /history/actor?name=...`: TMDB person lookup → credits cross-referenced with watch history
   - `/actor` page: person profile header + watched titles grid with character roles

4. **"Chat with Watcher" — Conversational Agent** ✅

   SSE-streaming chat endpoint backed by a **LangGraph** `StateGraph` (agent node ↔ ToolNode, with a conditional self-correction `verify` node). **Routing and decisions live in the tools and graph structure, not a prescriptive prompt** — each tool's docstring states when/when-not to use it, the slim `system_v2.jinja2` carries only identity + behaviour principles, and the agent self-decides (e.g. it picks from the watchlist itself rather than delegating to a hardcoded sub-prompt).

   **Taste profile always in context** — `_agent_node` rebuilds the system message every turn via `_system_prompt()`, which injects `get_taste_text()` (signature + summary + genres + themes) into `system_v2.jinja2`. The profile is computed once and TTL-cached for 1h (`taste_profile.py`), so it is present on every turn with no extra LLM/DB round-trip, and degrades silently to no taste block when history isn't synced yet. This is what lets the agent personalise recommendations and pick from the watchlist without first calling a history/taste tool.

   **Architecture** (`app/chat.py`)
   - 14 LangChain `@tool` functions: `get_recommendations` (optional `genres` filter), `find_similar`, `find_similar_in_history`, `will_i_like`, `search_by_description`, `get_cast`, `get_title_details`, `get_taste_profile`, `lookup_person`, `actor_in_history`, `get_history`, `get_watchlist_tool` (also the basis for watchlist recommendations — the agent fetches the list, filters by `media_type`, and picks the best taste-fit itself), `add_to_watchlist_tool`, `remove_from_watchlist_tool`
   - **No duplicated logic with the API/UI**: tools call the same shared functions the routes do — `find_similar` / `find_similar_in_history` → `app/similar.py::compute_similar` (also backs `POST /similar`), `will_i_like` → `compute_will_like`, `get_taste_profile` → `compute_taste_profile`, `get_recommendations` → `TmdbRecommender`, `search_by_description` → `discover_by_description`, `actor_in_history` → `get_actor_history`, watchlist tools → the watchlist DAO/sync
   - `ChatOpenAI` (langchain-openai) bound with tools → `_agent_node`; models configurable via `settings.CHAT_MODEL` (default `gpt-4.1-mini`) and `settings.CHAT_VERIFY_MODEL` (default `gpt-4.1-nano`)
   - `ToolNode` (langgraph.prebuilt) for tool execution with `handle_tool_errors=True`
   - `verify` node: after a final answer on a **tool-using** turn, a cheap LLM judge (`verify_v1.jinja2`) checks whether the user's question was actually addressed; if not, it injects a correction nudge and loops back to the agent **once** (`State.revised` caps it). Pure-chat turns skip verification (balanced latency).
   - Graph: `START → agent → (tools → agent)* → [verify → agent?] → END`
   - `_graph.astream_events(version="v2")` drives the SSE generator; the final answer is buffered so a correction supersedes the first draft
   - SSE event types: `tool_start`, `tool_result`, `message`, `error`, `done`

   **Dependencies added**: `langgraph`, `langchain-openai`, `langchain-core`

   **Frontend** (`/chat`)
   - Transcript view; tool status indicators (spinner while running, checkmark when done)
   - Tool results rendered inline as compact `MediaCard` grids (3–4 col, no overview, no Find Similar)
   - All returned items shown — no slice cap so agent follow-ups only reference visible titles
   - `actor_history`, `history`, `cast`, and `watchlist` results shown as collapsible summaries, not card grids (so the agent's own pick — shown via `get_title_details` — is the only prominent card)
   - `will_like` result shown as poster + `VerdictBadge` + `AiBlurb`; no Find Similar link
   - `title_details` shown as poster + metadata line (year · genres · runtime · rating) + overview; `taste_profile` shown as signature + summary + genre/theme/avoid chips
   - Example prompts shown when history is empty; renders markdown in text responses
   - Mobile: HTTPS-free UUID fallback (`crypto.randomUUID` feature-detected), sticky input via `100dvh` flex layout

5. **Actor Search Auto-populate Dropdown** ✅
   - The Actor Search page uses a plain text input. Add a typeahead dropdown backed by TMDB `/search/person` (similar to the title search typeahead) so users can confirm the exact person before submitting.
   - Show profile photo + known-for titles in the suggestion row.

6. **Actor Search by Photo** *(stretch goal)*
   - Allow the user to upload or paste a photo of an actor; identify the person via a vision-capable LLM or a reverse-image/face-recognition API, then run the existing actor history lookup.
   - Possible approach: send the image to a vision LLM (e.g. `gpt-4o`) with the prompt "Who is the person in this image? Return their full name." → pass the name to the existing `GET /history/actor?name=` flow.

7. **Light / Dark Mode Toggle**
   - Currently dark-mode only. Add a light-mode palette and a manual toggle persisted to `localStorage`.
   - Respect `prefers-color-scheme` as the default; CSS custom properties in `index.css` already use `[data-theme]` / `:root` conventions via Tailwind v4 — extend with a `light` theme variant.
   - Persist the toggle to `localStorage` and expose it in the nav bar.

8. **"Will I Like?" on Discover results**
   - Surface a per-result "Will I Like?" verdict on the Discover (description search) result cards, reusing the existing `/will-like` LLM scoring.
   - Each `MediaCard` gets an inline verdict badge + confidence (lazy-loaded per card, or batched) so the user can gauge fit without leaving the page.
   - Reuse `compute_will_like` / `WillLikeResponse`; consider a batch endpoint (`POST /will-like/batch`) to score many titles in one call and avoid N round-trips.

9. **Add any title to chat — title-scoped Q&A**
   - Let the user attach/pin a specific title to a chat turn (from search typeahead, a card action, or `/chat?id=&type=`) and ask questions scoped to it.
   - Surface prior interactions with that title if it's in watch history (watch count, dates, rewatches) via the existing history DAO.
   - Answer questions about its cast/crew and "where else have I seen them" by chaining the new `get_cast` tool with `actor_in_history`.
   - Backend: a chat tool like `title_context(tmdb_id, media_type)` that bundles TMDB metadata + history match (if any) + top cast into one structured result the agent can reason over.
   - Titles across all pages would have a "chat with this title" option (a chat icon) that opens it up in the chat page to chat with it.

12. **Watchlist** ✅

   Personal watchlist synced bidirectionally with Trakt custom lists — movies and TV shows managed separately, both readable and writable from Watcher or on mobile via Trakt.

   **Sources of truth**
   - **Movies**: Trakt custom list (`TRAKT_MOVIE_LIST_ID` in `.env`)
   - **TV shows**: Trakt custom list (`TRAKT_TV_LIST_ID` in `.env`)
   - Trakt auth already in place — no additional credentials needed
   - Items added/removed in Watcher write back to Trakt immediately; items added on mobile sync in on next pull

   **Trakt API** (verified against live API):
   - `GET /users/me/lists/{slug}/items` — fetch list contents
   - `POST /users/me/lists/{slug}/items` — add item by TMDB ID (`{"movies": [{"ids": {"tmdb": id}}]}`)
   - `POST /users/me/lists/{slug}/items/remove` — remove item

   **Data model** — local MongoDB `watchlist` cache, unique key `(tmdb_id, media_type)`:
   ```
   { tmdb_id, media_type, title, poster_path, overview, release_date, synced_at }
   ```
   Local cache is write-through: mutations hit Trakt first, then update local state. Pull sync refreshes the cache on demand and on app load.

   **Backend**
   - `app/dao/watchlist.py` — local cache CRUD
   - `app/watchlist_sync.py` — Trakt pull (GET both lists → upsert cache), push add/remove (POST to Trakt → update cache)
   - New endpoints:
     - `GET /watchlist` — return cached items (`?media_type=movie|tv`)
     - `POST /watchlist` — add item: resolves TMDB metadata, writes to Trakt, updates cache
     - `DELETE /watchlist/{tmdb_id}?media_type=` — remove: deletes from Trakt, updates cache
     - `POST /watchlist/sync` — pull latest from both Trakt lists, refresh cache; returns `{ added, removed }`
   - History sync auto-clear: after each Trakt history sync, remove watchlist items that now appear in watch history; surface count as `watchlist_cleared: N` in sync response

   **Frontend**
   - New `/watchlist` page: poster card grid (using `MediaCard`), filter tabs (All / Movies / TV), per-card remove button, "Sync" button to pull latest from Trakt, inline add via `SearchTypeahead`
   - `useWatchlist` context — fetches cache on mount, exposes `watchlist`, `isOnWatchlist(id, mediaType)`, `toggle(item)`; optimistic updates so UI feels instant

   **Integrations with existing features**
   - **Recommendations** — watchlist badge + add/remove toggle on every card; optional "hide watchlisted" filter
   - **Similar Titles** — badge + toggle on By Title and From History results; not shown in To History mode (all results already watched)
   - **Will I Like?** — show "already on your watchlist" alongside prediction; add/remove button in result panel
   - **Discover** — badge + toggle on each result card
   - **Chat** — three new tools: `get_watchlist()`, `add_to_watchlist(tmdb_id, media_type)`, `remove_from_watchlist(tmdb_id, media_type)`; enables queries like "which of my watchlist shows matches my taste?"
   - **Actor Search** — watchlist badge inline on filmography results
   - **Home page** — summary widget: "12 movies · 8 shows to watch" linking to `/watchlist`

   **Build order**: Trakt sync + local cache + endpoints → WatchlistPage → `useWatchlist` context + MediaCard badges → history auto-clear → chat tools → home widget

13. **In-Progress Shows**

   TV shows that have been started but not completed, derived from Trakt watch history — no extra list or manual tracking needed.

   **Definition**: a show where at least one episode has been watched but the series is not fully watched (i.e. watched episode count < total aired episodes).

   **Implementation**
   - Backend: `GET /history/in-progress` — filters watch history to TV shows with `watch_count > 0` and `aired_episodes > watched_episodes` (data already stored during Trakt sync)
   - Frontend: "In Progress" filter tab on the History page, alongside existing filters

   **States (no overlap)**
   - Watchlist — want to watch (Trakt custom list, not yet started)
   - In Progress — started, not finished (derived from history)
   - Watched — fully complete (existing history)

15. **Code Quality**
   - Add Ruff for code formatting/linting
   - Add pre-commit hooks
   - MongoDB indexes on `watch_history`

16. **Performance**
   - Make efficiency improvements (speed, LLM cost, etc.)
   - Ensure Trakt sync doesn't block recommendation requests
   - Monitor LLM call latency

