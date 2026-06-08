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

   SSE-streaming chat endpoint backed by a **LangGraph** `StateGraph` (agent node ↔ ToolNode, conditional edge loops until no tool calls remain).

   **Architecture** (`app/chat.py`)
   - 8 LangChain `@tool` functions: `get_recommendations` (optional `genres` filter), `find_similar`, `will_i_like`, `search_by_description`, `get_cast`, `lookup_person`, `actor_in_history`, `get_history`
   - `ChatOpenAI` (langchain-openai) bound with tools → `_agent_node`; model: `gpt-4.1-mini`
   - `ToolNode` (langgraph.prebuilt) for tool execution with `handle_tool_errors=True`
   - Graph: `START → agent → (tools → agent)* → END`
   - `_graph.astream_events(version="v2")` drives the SSE generator
   - SSE event types: `tool_start`, `tool_result`, `message`, `error`, `done`

   **Dependencies added**: `langgraph`, `langchain-openai`, `langchain-core`

   **Frontend** (`/chat`)
   - Transcript view; tool status indicators (spinner while running, checkmark when done)
   - Tool results rendered inline as compact `MediaCard` grids (3–4 col, no overview, no Find Similar)
   - All returned items shown — no slice cap so agent follow-ups only reference visible titles
   - `actor_history`, `history`, and `cast` results shown as collapsible summaries, not card grids
   - `will_like` result shown as poster + `VerdictBadge` + `AiBlurb`; no Find Similar link
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

6. **Code Quality**
   - Add Ruff for code formatting/linting
   - Add pre-commit hooks
   - MongoDB indexes on `watch_history`

6. **Performance**
   - Ensure Trakt sync doesn't block recommendation requests
   - Monitor LLM call latency

