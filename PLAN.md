# Watcher Evolution Plan

A single consolidated plan covering optimizations, TMDB sync, LangGraph agentic recommendations, MongoDB-to-Turso migration, improvements to recommendations/Will I Like/Visual Explorer/Similar Items, new features (e.g. "Where have I seen this actor?"), full UI overhaul, and multi-user support. Multi-user is last priority; single user is fine for now.

**Phases are ordered in execution order — work through them top to bottom.**

---

## Phase 0: Prevent tmdb_metadata Loss (URGENT)

**Problem:** `tmdb_metadata` keeps vanishing on watcher-mongo (e2-micro VM). Swap helped with watch_history OOMs but tmdb_metadata is still being lost. Likely triggers:
- **Trakt sync (watch history)** — Runs when you hit Recommend and there's new Trakt activity. Does `delete_many({})` + `insert_many()` on watch_history. That burst of writes can spike MongoDB memory on the VM. OOM kill → mongod dies → recovery corrupts/loses all collections including tmdb_metadata. Sync only touches watch_history, but the crash affects the whole DB.
- **GCE VM restarts** (maintenance, preemption) — abrupt shutdown, MongoDB recovery fails
- **OOM from MongoDB alone** — 1.34GB tmdb_metadata (largest collection) + 0.3GB cache on 1GB VM; tight headroom

**No app code deletes tmdb_metadata** — the app only does `update_one` upserts. The loss is from MongoDB process crashes.

### 0.1 Options (pick one or combine)

| Option | Effort | Description |
|--------|--------|-------------|
| **A) Automated backup off-VM** | Low | MongoDB data lives on the VM disk (DEPLOYMENT.md: persistent disk, no auto-delete). When mongod crashes, that disk can be corrupted — backup on the same disk doesn't help. Use `mongodump` → upload to **GCS** (or another off-VM location) so you have a copy that survives VM crashes. Cron on VM or Cloud Run Job. Add `tools/backup_to_gcs.sh` and `tools/restore_from_gcs.sh`. |
| **B) Larger VM** | Low | Upgrade watcher-mongo from e2-micro (1GB) to e2-small (2GB). ~$7/mo vs ~$3.5/mo. Reduces OOM frequency. |
| **C) MongoDB Atlas** | N/A | Atlas M0 free tier is 512MB — tmdb_metadata.bson alone is ~1.34GB, so not viable. Paid tiers would work but add cost. |
| **D) Accelerate Turso migration** | Medium | Move Phase 8 (Turso) to now. Turso free tier has 9GB; no OOM. Removes MongoDB VM entirely. |
| **E) Reduce MongoDB cache further** | Low | In `mongod.conf`, try `cacheSizeGB: 0.2` (from 0.3). Frees RAM; may slow queries. |

### 0.2 Recommended Immediate Actions

1. **Backup now** — Run `python tools/mongo_local_dump_export.py` locally (or from a machine that can reach the VM), dump to a safe location. You need a restore point.
2. **Add off-VM backup cron** — Daily `mongodump` of `watcher` DB → upload to GCS (or other off-VM storage). The VM disk is where MongoDB lives; when it corrupts, you need a copy elsewhere. Document restore steps in DEPLOYMENT.md.
3. **Move Trakt check off recommend path** (Phase 2) — `check_trakt_last_activities_and_sync` runs on every Recommend and can trigger a full watch_history sync. That write burst may be the OOM trigger. Run Trakt sync only via sync_worker (or a dedicated job), not on every recommend.
4. **Consider B** — If backups aren't enough (restoring is painful), upgrade to e2-small. Atlas free tier is too small (512MB limit; tmdb_metadata is ~1.34GB).

### 0.3 Restore When Empty

If tmdb_metadata is empty and you have a dump:

```bash
# From machine with dump
mongorestore --drop --db watcher --collection tmdb_metadata mongo_dumps/tmdb_metadata.bson --host <MONGO_VM_IP> -u <user> -p <pass> --authenticationDatabase admin
```

---

## Current State Summary

| Area | Current | Gap |
|------|---------|-----|
| **Users** | Single user (one token file, last auth wins) | No user isolation |
| **TMDB sync** | Manual via Admin; `sync_worker` not deployed | No automatic new-item sync in production |
| **DB** | MongoDB on e2-micro (0.3GB cache, size limits) | tmdb_metadata keeps vanishing (OOM/crash); scaling and cost concerns |
| **Recommendations** | Linear flow: KNN → LLM | No agentic orchestration, no human-in-loop |
| **Performance** | N+1 in clusters, no indexes | Slower than necessary |

---

## Phase 1: Quick Wins (1–2 days)

### 1.1 MongoDB Indexes

Add indexes in `app/db.py` at startup (idempotent):

```python
watch_history_collection.create_index("media_type")
tmdb_metadata_collection.create_index([("id", 1), ("media_type", 1)])
sync_meta_collection.create_index([("status", 1), ("finished_at", -1)])
tmdb_failures_collection.create_index([("id", 1), ("media_type", 1)])
```

### 1.2 Batch Metadata in Visual Explorer

Replace N+1 `find_one` loop in `app/api.py` (lines 976–991) with a single `$or` query and lookup map:

```python
# Build $or conditions for all unique_pairs
or_conditions = [
    {"id": int(tmdb_id), "media_type": str(mt).lower()}
    for (tmdb_id, mt), v in zip(unique_pairs, vecs) if v is not None
]
if or_conditions:
    # Batch into chunks of ~500 if large (MongoDB $or limit)
    metadata_map = {}
    for i in range(0, len(or_conditions), 500):
        chunk = or_conditions[i:i+500]
        cursor = tmdb_metadata_collection.find(
            {"$or": chunk},
            {"_id": 0, "id": 1, "media_type": 1, "genres": 1, "overview": 1, "title": 1, "poster_path": 1, "backdrop_path": 1}
        )
        for doc in cursor:
            metadata_map[(doc["id"], doc["media_type"])] = doc
else:
    metadata_map = {}

# In loop: doc = metadata_map.get((int(tmdb_id), str(mt).lower()))
```

### 1.3 Ruff Formatting and Import Sorting

Add Ruff for consistent formatting and import sorting. Add `pyproject.toml` or `ruff.toml`; run `ruff format` and `ruff check --fix` as pre-commit or CI step.

---

## Phase 2: Recommendation & Feature Quick Wins

### 2.1 Recommendations (`app/process/recommendation.py`)

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Fix rewatch weighting bug** | Low | `build_user_vector_from_history` uses `rewatch_count` but history has `rewatch_engagement`. Use `h.get("rewatch_engagement", 1.0)` in `app/embeddings.py`. |
| **Batch title resolution** | Low | `_resolve_title` can call `find_one` per candidate. Batch-fetch all candidate metadata in one query. |
| **Use cached watch history** | Low | Ensure recommend path uses `get_watch_history` cache; avoid triggering `check_trakt_last_activities_and_sync` on every recommend (or make it non-blocking). |

### 2.2 Will I Like (`app/mcp_will_like.py`)

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Threshold tuning** | Low | Make 0.65 configurable via env `WILL_LIKE_THRESHOLD` or Admin. |

### 2.3 Similar Items / KNN

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Watched filter option** | Low | Add optional `exclude_watched: bool` to Similar Items UI/API. |

### 2.4 Watch History UI (`ui/streamlit_app.py`)

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Genre/year filters** | Low | Add sidebar filters; extend `/history` API with `genre`, `year_from`, `year_to`. |
| **Copy TMDB link** | Low | Add copy-to-clipboard for TMDB URL on each card. |

### 2.5 Admin Panel

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Sync status** | Low | Show `last_sync` per TMDB media type, last Trakt sync time. "Force full sync" button. |
| **Cache controls** | Exists | Add "Clear history cache" button (calls existing endpoint). |

---

## Phase 3: TMDB Sync Fixes and Automation

### 3.0 Sync Behavior Verification (Script Review)

**Findings from code review:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| **Full sync never sets `last_sync`** | **Bug** | `app/tmdb_sync.py` | After `_sync_from_export` succeeds, call `_set_last_sync_timestamp(media_type, int(time.time()))`. After discover-path full sync, call `_set_last_sync_timestamp` with `max(max_seen)` or `int(time.time())`. Without this, every TMDB sync falls back to full sync (export/discover) because `last_sync` is always None. |
| **sync_worker not started** | Gap | `start.sh`, `Dockerfile`, `DEPLOYMENT.md` | `start.sh` only runs backend + frontend. Dockerfile CMD runs uvicorn only. sync_worker must be run separately. |
| **Trakt check on every recommend** | Latency | `app/api.py` line 100 | `check_trakt_last_activities_and_sync()` blocks every `/recommend` request. Consider making it async/background or moving to sync_worker only. |
| **TMDB sync only via sync_worker** | Gap | `app/scheduler.py` | `run_tmdb_periodic_sync` is only called from sync_worker. No fallback. If sync_worker isn't running, TMDB never auto-updates. |
| **Trakt has fallback** | OK | `app/api.py` | Recommend endpoint triggers Trakt check, so Trakt can sync even without sync_worker (when user hits recommend). |

**Trakt sync logic:** Correct. Uses `last_activities` to compare with DB; hash check avoids redundant writes. First run (empty DB) correctly forces sync.

**TMDB incremental logic:** Correct when `last_sync` exists. Pagination, `_fetch_changes`, `_process_details_batch` all work. The bug is that full sync never populates `last_sync`.

### 3.1 Fix: Set `last_sync` After Full Sync (Required)

In `app/tmdb_sync.py`:

1. **Export path:** In `sync_tmdb`, inside the `if export_summary is not None` block (around line 921), before `return export_summary`, add:
   ```python
   _set_last_sync_timestamp(media_type, int(time.time()))
   ```

2. **Discover path:** In the discover full-sync loop (lines 957–974), add `max_seen_global = 0` before the loop and `max_seen_global = max(max_seen_global, max_seen or 0)` inside. After the loop (before line 976), add:
   ```python
   if max_seen_global:
       _set_last_sync_timestamp(media_type, max_seen_global)
   else:
       _set_last_sync_timestamp(media_type, int(time.time()))
   ```

### 3.2 Sync Worker Not Deployed

**Finding:** `sync_worker.py` runs Trakt + TMDB periodic sync, but `start.sh` and `DEPLOYMENT.md` do not start it. GCP deployment has no scheduler.

**Options:**
- **A) Cloud Run Job** — Deploy `sync_worker` as a Cloud Run Job, scheduled via Cloud Scheduler (cron) every 6–72 hours.
- **B) In-process scheduler** — Run APScheduler inside the FastAPI app (e.g. in lifespan). Simpler but ties sync to backend lifecycle.
- **C) Document manual flow** — Add to DEPLOYMENT.md: run `python sync_worker.py` on a cron VM or locally.

**Recommendation:** A — Cloud Run Job + Cloud Scheduler. Keeps sync decoupled, scales to zero.

### 3.3 Incremental Sync Behavior

**Flow:** `sync_tmdb_changes` → if no `last_sync` → falls back to full sync (export/discover). First run or empty DB triggers full sync; subsequent runs use `/movie/changes` and `/tv/changes`.

**Action:** Add a "Sync status" section in Admin: show `last_sync` per media type, and a "Force full sync" button. Log clearly when falling back to full sync.

---

## Phase 4: Where Have I Seen This Actor Before?

**New feature (priority).** Search watch history by actor name; return titles where they appear.

**Implementation:**
- New endpoint: `GET /history/actor?name=...` or `POST /search/actor` with `{ "name": "..." }`.
- Query `watch_history` for items in user's history; join with `tmdb_metadata` where `credits.cast` (or `cast`) contains matching actor name.
- TMDB credits shape: `credits.cast[]` with `name`, `order`. Match case-insensitive, support partial (e.g. "Leo" → "Leonardo DiCaprio").
- Return: `[{ id, title, media_type, role?, poster_path }]` — titles from watch history where actor appears.
- UI: New tab or section "Actor Search" — input actor name, show grid of matching titles from history.

**Files:** `app/api.py`, `app/dao/history.py` (or new `app/dao/actors.py`), `ui/streamlit_app.py`.

---

## Phase 5: LangGraph-Powered Agentic Recommendations

### 5.1 Why LangGraph

- **Cyclic flows:** Loop on user feedback (e.g. "more like X, less like Y").
- **Tool orchestration:** Choose among KNN, will-like, similar, history-based recommend.
- **Human-in-the-loop:** Pause for approval, refine, or reject before final output.
- **State management:** Keep conversation context across steps.

### 5.2 Proposed Graph

```
User Query → Parse Intent → Route
  Route → similar → KNN Tool
  Route → predict → Will-Like Tool
  Route → recommend → Recommendation Tool
  → Format Results → Present to User → Feedback?
    Feedback: refine → Parse Intent (loop)
    Feedback: accept → End
```

### 5.3 Implementation Sketch

- **Nodes:** `parse_intent`, `call_knn`, `call_will_like`, `call_recommend`, `format_results`, `wait_for_feedback`.
- **Edges:** Conditional on intent; loop back to `parse_intent` on "refine" feedback.
- **State:** `messages`, `last_results`, `user_history_summary`.
- **Tools:** Wrap existing `call_mcp_knn`, `compute_will_like`, `MediaRecommender.get_recommendations` as LangGraph tools.

### 5.4 Integration

- New endpoint: `POST /recommend/agent` — accepts natural language, returns streaming or structured response.
- Streamlit: "Chat with Watcher" tab — conversational UI.
- Learning value: Good way to learn LangGraph's `StateGraph`, conditional edges, and tool nodes.

---

## Phase 6: Recommendation & Feature Medium Improvements

### 6.1 Recommendations

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Diversity injection** | Medium | After LLM selection, ensure genre/vibe spread. If LLM picks 5 sci-fi, inject 1–2 from different clusters. |
| **Negative signals** | Medium | Support "exclude" or "less like X" — add optional exclude-genres or exclude-ids to prompt and filter candidates. |
| **Candidate pool size** | Low | Consider `recommend_count * 30` or configurable; 200 may be limiting for small counts. |
| **Prompt versioning** | Low | Add `prompt_version` as API param for future A/B testing. |
| **Streaming reasoning** | Medium | Stream LLM reasoning tokens for perceived speed. |

### 6.2 Will I Like

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **LLM explanation** | Medium | Replace static explanation with short LLM-generated: "Based on your love of [genre] and [title], this fits because..." |
| **Confidence bands** | Low | Return `confidence: "high" \| "medium" \| "low"` based on score ranges. |
| **Similar titles** | Low | Include 2–3 "You might also like" from KNN on the same item. |
| **Will I Like text search fix** | Low | Bug fix for text-based Will I Like (title resolution, enrichment). |

### 6.3 Visual Explorer

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Batch metadata** | Low | Phase 1 — fixes N+1. |
| **Cluster caching** | Low | Cache results by `(media_type, n_clusters)` with TTL (e.g. 1 hour). Invalidate on history sync. |
| **UMAP option** | Medium | Add UMAP as alternative to t-SNE (often faster). Make reducer configurable. |
| **Cluster drill-down** | Low | Return cluster_id per item; UI can filter "show only cluster X." |
| **LLM cluster names** | Low | Cache cluster names keyed by sorted genre+titles hash. |

### 6.4 Similar Items / KNN

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Similar from watch history only** | Low | Add option to restrict results to items *in* watch history. |
| **Infer media_type from query** | Low | When searching by title/text, infer movie vs TV from TMDB result. |
| **Hybrid search** | Medium | Combine vector similarity with genre filter. "Similar to Inception but only Sci-Fi." |
| **Result ordering** | Low | Consider secondary sort by popularity or release date for tie-breaking. |

### 6.5 Watch History UI

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Sort options** | Low | Add "Recently watched", "By year", "By title." |
| **UI state handling** | Low | Streamlit session state, tab persistence, form state. |
| **Language filter** | Low | Filter history/recommendations by `original_language`. |

### 6.6 Embeddings

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Recency decay config** | Low | Document `RECENCY_DECAY_DAYS` in README; expose in Admin. |
| **User vector caching** | Low | Cache by `(media_type, history_hash)` with short TTL. |
| **Embedding model choice** | Medium | Support larger models; make configurable per index build. |
| **Configurable dim** | Low | `dim` is hardcoded to 384; make configurable via env or sidecar_meta for model switching. |

### 6.7 Admin Panel

| Improvement | Effort | Description |
|-------------|--------|-------------|
| **Embedding stats** | Low | Show count of items with embeddings vs total metadata; % coverage. |

---

## Phase 7: Feature Ideas (Beyond Core)

- **"Chat with Watcher"** — LangGraph agent as main entry point.
- **"Mood-based discovery"** — "I want something light and funny" → agent uses KNN + genre filters + LLM.
- **"Watch party"** — Share recommendation link; others see "Will I Like?" (multi-user prerequisite).
- **"Trakt list sync"** — Import Trakt watchlist/custom lists as signals for recommendations.
- **Use ratings/likes/reviews** — Trakt ratings as signals for user vector or LLM prompt.
- **Poster embedding / understanding** — Multimodal: embed poster images for similarity (higher effort).
- **Plot summary or script understanding** — Richer content signals for embeddings.

---

## Phase 8: MongoDB → Turso Migration

### 8.1 Why Turso

- **Free tier:** 9GB storage, 500 DBs, 25B row reads/month.
- **SQLite-compatible:** Familiar, single-file for local dev.
- **libSQL:** Vector search support (future).
- **No server to manage:** Unlike self-hosted MongoDB on e2-micro.

### 8.2 Schema Design (High Level)

- **watch_history:** `(user_id, id, media_type, ...)` — compound PK.
- **tmdb_metadata:** `(id, media_type)` — JSON/JSONB for nested fields.
- **sync_meta:** `(key TEXT PRIMARY KEY, value JSON)`.
- **tmdb_failures:** `(id, media_type)`.

### 8.3 Migration Steps

1. Define SQL schema and migration scripts.
2. Implement DB abstraction layer that can switch backend.
3. One-time migration script: read from MongoDB, write to Turso.
4. Swap `MONGODB_URI` for `TURSO_URL` + `TURSO_AUTH_TOKEN`.
5. Update DEPLOYMENT.md for Turso setup.

### 8.4 Alternative

Stay on MongoDB but add indexes and optimize; migrate only when size/cost becomes an issue.

---

## Phase 9: Full UI Overhaul

Replace Streamlit with a proper frontend (e.g. React, Next.js, or Vue) for better UX, layout control, and mobile support.

**Scope:**
- New SPA or SSR app that consumes the existing FastAPI backend.
- Recreate all current views: Home, Watch History, Visual Explorer, Recommendations, Will I Like, Similar Items, Admin, Actor Search.
- Improve: responsive layout, state management, no full-page re-runs, richer interactions (e.g. Visual Explorer drill-down).
- Auth: session/cookie handling for Trakt OAuth flow.

**Considerations:**
- Backend API stays unchanged; UI is a new client.
- Deploy as separate Cloud Run service (like current Streamlit) or static + CDN.
- Effort: medium–high. Do after core features (actor search, LangGraph, Turso) are stable.

---

## Phase 10: Multi-User Support (Last Priority)

Single user is fine for now. Defer until needed.

### 10.1 Architecture

- **Token storage:** Move from single `.env.trakt_token` to per-user storage (e.g. `user_id` → token in DB or encrypted store).
- **User identity:** Use Trakt user slug (or sub) from OAuth as stable `user_id`.
- **Data isolation:**
  - `watch_history`: add `user_id`, compound index `(user_id, id, media_type)`.
  - `sync_meta`: scope job docs by `user_id` (e.g. `trakt_sync_job:{user_id}:{job_id}`).
  - `tmdb_metadata`, `tmdb_failures`: keep global (shared catalog).
- **API:** Add auth middleware; resolve `user_id` from session/cookie or Bearer token.

### 10.2 Auth Flow

- OAuth callback: persist token keyed by Trakt user slug.
- Session: issue HTTP-only cookie or JWT with `user_id` after OAuth.
- Logout: clear session.

### 10.3 Migration Path

1. Add `user_id` field; backfill existing data with default `user_id` (e.g. `"legacy"`).
2. Update all history/sync queries to filter by `user_id`.
3. Add login/logout UI and API auth checks.

---

## Files to Touch (Summary)

| Phase | Files |
|-------|------|
| 0 | `tools/backup_to_gcs.sh`, `tools/restore_from_gcs.sh`, `DEPLOYMENT.md`, VM/cron config |
| 1 | `app/db.py`, `app/api.py`, `pyproject.toml` (Ruff) |
| 2 | `app/embeddings.py`, `app/process/recommendation.py`, `app/mcp_will_like.py`, `app/utils/llm_orchestrator.py`, `app/utils/knn_utils.py`, `ui/streamlit_app.py` |
| 3 | `app/tmdb_sync.py`, `sync_worker.py`, `Dockerfile`, `DEPLOYMENT.md`, `app/api.py` |
| 4 | `app/api.py`, `app/dao/` (actors), `ui/streamlit_app.py` |
| 5 | New: `app/agent/`, `app/api.py`, `ui/streamlit_app.py` |
| 6 | `app/embeddings.py`, `app/process/recommendation.py`, `app/mcp_will_like.py`, `app/api.py`, `app/utils/llm_orchestrator.py`, `app/utils/knn_utils.py`, `ui/streamlit_app.py` |
| 7 | Various — Trakt sync, ratings, etc. |
| 8 | New: `app/db_turso.py`, migration script, `app/config/settings.py`, `DEPLOYMENT.md` |
| 9 | New frontend app (e.g. `frontend/` or `ui-next/`), `Dockerfile`, `DEPLOYMENT.md` |
| 10 | `app/auth/`, `app/dao/history.py`, `app/db.py`, `app/api.py`, `ui/streamlit_app.py` |
