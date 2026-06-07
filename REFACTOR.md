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
- Remove import: `RecommendComparePage`
- Remove route: `<Route path="/recommend-compare" element={<RecommendComparePage />} />`
- Remove any navigation links to `/recommend-compare`

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

---

## Implementation Order

1. **Step 1** — Create feature branch
2. **Step 2** — Update `/mcp/knn` to use TMDB similar API
3. **Step 3** — Remove admin FAISS endpoints
4. **Step 4** — Remove/update `/mcp/will-like` (decide keep/remove/reimplement)
5. **Step 5** — Update frontend (maintenance page, similar component)
6. **Step 6** — Remove old files (embeddings, FAISS, sync_worker, tmdb_sync)
7. **Step 7** — Clean up imports, environment variables
8. **Step 8** — Update requirements.txt, Dockerfile
9. **Step 9** — Drop MongoDB collections
10. **Step 10** — Update all documentation
11. **Step 11** — Test locally, commit

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

## Future Enhancements (Still Relevant)

These features from the old roadmap are still worth considering post-refactor:

1. **Watch History UI Improvements**
   - Genre/year filters on `/history` endpoint and React UI
   - Copy-to-clipboard for TMDB URLs on each card
   - Sort options (recently watched, by year, by title)

2. **"Where Have I Seen This Actor?" Feature**
   - New endpoint: search watch history by actor name
   - Return titles where they appear with roles
   - Frontend: "Actor Search" tab/section
   - Note: Currently would require storing TMDB credits; alternative is to fetch on-demand

3. **LangGraph Agentic Recommendations** (learning project)
   - Conversational UI: "Chat with Watcher"
   - Parse natural language intent (recommend, similar, will-like)
   - Route to appropriate tools
   - Support refinement loop ("more like X, less like Y")
   - New endpoint: `POST /recommend/agent`

4. **Code Quality**
   - Add Ruff for code formatting/linting
   - Add pre-commit hooks
   - MongoDB indexes on `watch_history` (even though tmdb_metadata is gone)

5. **Performance**
   - Ensure Trakt sync doesn't block recommendation requests (already noted in refactor)
   - Cache watch history appropriately
   - Monitor LLM call latency

