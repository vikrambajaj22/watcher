from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List
import os
import subprocess
import sys
import time as _time
import traceback as _traceback
import uuid

from fastapi import APIRouter, BackgroundTasks, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
import numpy as np
from pymongo import DESCENDING
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from app.auth.trakt_auth import exchange_code_for_token, get_auth_url, save_token_data
from app.config.settings import settings
from app.dao.history import get_watch_history, clear_history_cache
from app.db import tmdb_metadata_collection, sync_meta_collection
from app.faiss_index import INDEX_DIR, INDEX_FILE, build_faiss_index
from app.mcp_will_like import compute_will_like, WillLikeError
from app.process.recommendation import MediaRecommender
from app.scheduler import check_trakt_last_activities_and_sync
from app.schemas.api import (
    AdminAckResponse,
    AdminCancelJobRequest,
    AdminEmbedFullPayload,
    AdminEmbedItemPayload,
    AdminFaissRebuildPayload,
    AdminFaissRebuildResponse,
    AdminJobAcceptedResponse,
    AdminSyncTMDBRequest,
    HistoryItem,
    JobStatusModel,
    KNNRequest,
    KNNResponse,
    SyncStatusResponse,
    TMDBMetadata,
    WillLikeRequest,
    WillLikeResponse,
    AdminFaissUpsertPayload,
    AdminFaissUpsertResponse,
)
from app.schemas.recommendations.recommendations import (
    RecommendationsResponse,
    RecommendRequest,
)
from app.tmdb_sync import sync_tmdb
from app.trakt_sync import sync_trakt_history
from app.utils.helpers import generate_cluster_name
from app.utils.llm_orchestrator import call_mcp_knn
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "service": "watcher"}


@router.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.classless.min.css">
            <title>Watcher</title>
        </head>
        <body>
            <main>
                <h1>Watcher</h1>
                <h4>watchu lookin at?</h4>
                <p>You are not logged in. Click <a href="/auth/trakt/start">here</a> to authenticate</p>
            </main>
        </body>
    </html>
    """


@router.post("/recommend/{media_type}", response_model=RecommendationsResponse)
def recommend(media_type: str, payload: RecommendRequest):
    """Generate recommendations for the given media_type (movie|tv).

    Expects JSON body of type RecommendRequest.
    """
    try:
        if media_type not in ("movie", "tv", "all"):
            raise HTTPException(status_code=400, detail="media_type must be 'movie' or 'tv'")

        recommend_count = payload.recommend_count

        check_trakt_last_activities_and_sync()
        recommender = MediaRecommender()
        return recommender.generate_recommendations(media_type=media_type, recommend_count=recommend_count)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("recommend error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[HistoryItem])
def get_history(media_type: str = None, include_posters: bool = True):
    """Get watch history from database, optionally filtered by media_type and get posters.

    Query params:
      - media_type: optional 'movie'|'tv'
      - include_posters: boolean (default true) - when false, skip poster enrichment for faster responses
    """
    try:
        history = get_watch_history(media_type=media_type, include_posters=include_posters)
        return history
    except Exception as e:
        logger.error("get_history error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/sync/trakt", response_model=AdminJobAcceptedResponse, status_code=202)
def admin_sync(background_tasks: BackgroundTasks):
    """Trigger Trakt history sync in background."""
    try:
        # create a job id and persist a job document in sync_meta_collection
        job_id = str(uuid.uuid4())
        job_key = f"trakt_sync_job:{job_id}"
        started_at = int(_time.time())
        sync_meta_collection.update_one(
            {"_id": job_key},
            {"$set": {"status": "pending", "started_at": started_at}},
            upsert=True,
        )

        # background wrapper to update job status
        def _run_sync_job(jid: str):
            job_key_inner = f"trakt_sync_job:{jid}"
            try:
                sync_meta_collection.update_one(
                    {"_id": job_key_inner}, {"$set": {"status": "running"}}
                )
                sync_trakt_history()
                sync_meta_collection.update_one(
                    {"_id": job_key_inner},
                    {
                        "$set": {
                            "status": "completed",
                            "finished_at": int(_time.time()),
                        }
                    },
                )
            except Exception as e:
                # record failure and error
                try:
                    sync_meta_collection.update_one(
                        {"_id": job_key_inner},
                        {
                            "$set": {
                                "status": "failed",
                                "finished_at": int(_time.time()),
                                "error": str(e),
                                "trace": _traceback.format_exc(),
                            }
                        },
                    )
                except Exception:
                    pass

        background_tasks.add_task(_run_sync_job, job_id)
        return AdminJobAcceptedResponse(status="accepted", job_id=job_id)
    except Exception as e:
        logger.error("admin_sync error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/sync/job/{job_id}", response_model=JobStatusModel)
def admin_sync_job_status(job_id: str, job_type: str = Query(..., description="Job type: 'trakt' or 'tmdb'")):
    """Return job status document for the given job_id (from sync_meta_collection).

    Requires explicit query param `job_type` which must be either 'trakt' or 'tmdb'.
    If `job_id` is already a full key (e.g. 'tmdb_sync_job:<id>') it must match `job_type`.
    """
    try:
        from app.db import sync_meta_collection

        t = (job_type or "").lower()
        if t not in ("trakt", "tmdb"):
            raise HTTPException(status_code=400, detail="job_type must be 'trakt' or 'tmdb'")

        # if caller passed a fully-prefixed key, ensure it matches the requested type.
        if job_id.startswith("trakt_sync_job:") or job_id.startswith("tmdb_sync_job:"):
            # validate prefix matches type
            expected_prefix = f"{t}_sync_job:"
            if not job_id.startswith(expected_prefix):
                raise HTTPException(status_code=400, detail=f"job_id prefix does not match job_type '{t}'")
            key = job_id
        else:
            key = f"{t}_sync_job:{job_id}"

        doc = sync_meta_collection.find_one({"_id": key}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="job not found")

        try:
            return JobStatusModel.model_validate(doc)
        except Exception:
            return JobStatusModel(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_sync_job_status error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/trakt/start")
def trakt_auth_start(request: Request, from_ui: bool = False):
    """Redirect user to Trakt OAuth authorization URL.

    Args:
        from_ui: If True, user will be redirected to UI after auth. Otherwise, to API docs.
    """
    state = "ui" if from_ui else "api"
    return RedirectResponse(get_auth_url(state=state))


@router.get("/auth/trakt/callback")
def trakt_auth_callback(request: Request):
    """Handle Trakt OAuth callback and exchange code for token.

    Redirects to UI homepage if state=ui, otherwise to API docs.
    """
    code = request.query_params.get("code")
    state = request.query_params.get("state", "api")

    if not code:
        # redirect to start if code is missing
        return RedirectResponse("/auth/trakt/start")
    try:
        token_data = exchange_code_for_token(code)
        save_token_data(token_data)
        if state == "ui":
            ui_base = getattr(settings, "UI_BASE_URL", "http://localhost:8501")
            return RedirectResponse(ui_base)
        else:
            return RedirectResponse("/docs")  # API docs
    except Exception:
        return RedirectResponse("/auth/trakt/start")


@router.get("/auth/status")
def auth_status() -> Dict[str, bool]:
    """Return current authentication status."""
    try:
        if settings.TRAKT_ACCESS_TOKEN != "":
            return {
                "authenticated": True
            }
        return {"authenticated": False}
    except Exception as e:
        logger.error("auth_status error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/logout")
def auth_logout():
    """Log out the current user by deleting the stored token file."""
    try:
        if settings.TRAKT_ACCESS_TOKEN:
            os.remove(settings.TRAKT_TOKEN_FILE)
        return {"status": "logged_out"}
    except Exception as e:
        logger.error("auth_logout error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/embed/item", response_model=AdminAckResponse, status_code=200)
def admin_embed_item(background_tasks: BackgroundTasks, payload: AdminEmbedItemPayload):
    """Attempt a fast incremental upsert for a single TMDB item.

    Behavior:
      - Try `upsert_single_item` to update sidecars/index in-place.
      - If upsert reports a full rebuild is required, schedule a background `build_faiss_index` and return a scheduled status.
    """
    try:
        tmdb_id = payload.id
        if not tmdb_id:
            raise HTTPException(status_code=400, detail="id is required")
        media_type = (payload.media_type or "movie").lower()
        # ensure item exists
        doc = tmdb_metadata_collection.find_one({"id": tmdb_id, "media_type": media_type}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="item not found")

        force = bool(getattr(payload, "force_regenerate", False))

        # perform incremental upsert
        try:
            from app.faiss_index import upsert_single_item, clear_index_cache
        except Exception:
            raise HTTPException(status_code=500, detail="faiss index module unavailable")

        res = upsert_single_item(tmdb_id, media_type, force_regenerate=force)

        status = res.get("status")
        message = res.get("message")

        # if upsert couldn't be applied in-place, schedule a full rebuild in background
        if status in ("rebuild_required", "rebuild_scheduled"):
            # clear cache and schedule full rebuild
            try:
                clear_index_cache()
            except Exception:
                pass
            reuse = not force
            logger.warning("Upsert single item failed; scheduling full FAISS rebuild")
            background_tasks.add_task(build_faiss_index, int(384), reuse_sidecars=reuse)
            return AdminAckResponse(status="rebuild_scheduled", message="Full FAISS rebuild scheduled; " + (message or ""))

        # otherwise return the upsert result
        return AdminAckResponse(status=str(status or "ok"), message=str(message or "upsert attempted"))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("admin_embed_item error: %s", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/embed/full", response_model=AdminAckResponse, status_code=202)
def admin_embed_full(background_tasks: BackgroundTasks, payload: AdminEmbedFullPayload):
    """Trigger full embedding generation (background). Expects JSON: {"batch_size": <int>}"""
    try:
        # Trigger a full FAISS rebuild which computes embeddings from metadata and writes sidecars.
        dims = 384
        reuse = not bool(getattr(payload, "force_regenerate", False))
        # clear in-memory index cache before starting rebuild so we don't keep serving stale index
        try:
            from app.faiss_index import clear_index_cache
            clear_index_cache()
        except Exception:
            pass
        background_tasks.add_task(build_faiss_index, dims, reuse_sidecars=reuse)
        return AdminAckResponse(status="accepted", message="faiss rebuild started")
    except Exception as e:
        logger.error("admin_embed_full error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/faiss/rebuild", response_model=AdminFaissRebuildResponse, status_code=202)
def admin_faiss_rebuild(payload: AdminFaissRebuildPayload):
    """Trigger FAISS rebuild in a detached process. Expects JSON: {"dim": <int>, "factory": "..."}"""
    try:
        dims = int(payload.dim or 384)
        factory = payload.factory or "IDMap,IVF100,Flat"

        python_exe = sys.executable or "python"
        cmd = [
            python_exe,
            "-m",
            "app.faiss_rebuild_cli",
            "--dim",
            str(dims),
            "--factory",
            factory,
        ]
        if getattr(payload, "force_regenerate", False):
            cmd.append("--force-regenerate")

        os.makedirs(INDEX_DIR, exist_ok=True)
        log_path = os.path.join(INDEX_DIR, "rebuild.log")
        log_file = open(log_path, "a")

        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            p = subprocess.Popen(
                cmd, creationflags=creation_flags, stdout=log_file, stderr=log_file
            )
        else:
            p = subprocess.Popen(
                cmd, start_new_session=True, stdout=log_file, stderr=log_file
            )

        try:
            log_file.close()
        except Exception as e:
            logger.error(
                "error closing log file after fais rebuild: %s", repr(e), exc_info=True
            )
            pass

        logger.info(
            "Spawned FAISS rebuild process pid=%s (logs at %s)",
            getattr(p, "pid", None),
            log_path,
        )
        # clear in-memory index cache in this process so we don't continue using stale index
        try:
            from app.faiss_index import clear_index_cache

            clear_index_cache()
        except Exception:
            pass
        return AdminFaissRebuildResponse(status="accepted", message="faiss rebuild started",
                                          pid=getattr(p, "pid", None), log=log_path)
    except Exception as e:
        logger.error("admin_faiss_rebuild error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/faiss/status")
def admin_faiss_status():
    """Return sidecar metadata (if any) and index presence info, plus whether the index is cached in this process."""
    try:
        from app.faiss_index import load_sidecar_meta, is_index_cached

        meta = load_sidecar_meta()
        present = os.path.exists(INDEX_FILE) or (meta is not None)
        cached = False
        try:
            cached = bool(is_index_cached())
        except Exception:
            cached = False
        return {"present": bool(present), "sidecar_meta": meta, "cached": cached}
    except Exception as e:
        logger.error("admin_faiss_status error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/faiss/clear-cache")
def admin_faiss_clear_cache():
    """Clear the in-process FAISS index cache."""
    try:
        from app.faiss_index import clear_index_cache

        clear_index_cache()
        return {"status": "cleared"}
    except Exception as e:
        logger.error("admin_faiss_clear_cache error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/faiss/upsert-item", response_model=AdminFaissUpsertResponse)
def admin_faiss_upsert(payload: AdminFaissUpsertPayload):
    """Attempt an incremental upsert of a single TMDB item into FAISS sidecars and index.

    Returns status indicating whether the item was added/updated locally, or whether a full rebuild is required.
    """
    try:
        from app.faiss_index import upsert_single_item

        res = upsert_single_item(payload.id, payload.media_type or "movie", force_regenerate=bool(payload.force_regenerate))
        # coerce to response model
        return AdminFaissUpsertResponse(status=res.get("status"), message=res.get("message"))
    except Exception as e:
        logger.exception("admin_faiss_upsert error: %s", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/tmdb/{tmdb_id}", response_model=List[TMDBMetadata])
def admin_get_tmdb_metadata(tmdb_id: int, media_type: str = None):
    """Debug endpoint: return stored TMDB metadata documents for the given tmdb_id.

    Optional query param `media_type` can be 'movie' or 'tv' to filter results.
    """
    try:
        q: Dict[str, object] = {"id": int(tmdb_id)}
        if media_type:
            q["media_type"] = str(media_type).lower()
        cursor = list(tmdb_metadata_collection.find(q, {"_id": 0}))
        if not cursor:
            raise HTTPException(status_code=404, detail="not found")
        return cursor
    except Exception as e:
        logger.error("admin_get_tmdb_metadata error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/mcp/knn",
    response_model=KNNResponse,
    summary="Find top-k nearest neighbors",
    description="""
Return the top-k nearest TMDB items for a tmdb_id, free-text query, or an embedding vector.
Provide exactly one of tmdb_id, text, or vector. The request separates the *input* media type (when providing a tmdb_id or doing a TMDB title lookup) and the *results* media type (a filter applied to returned neighbors).

Fields:
 - input_media_type: optional for free-text but required ('movie'|'tv') to resolve a provided tmdb_id or title.
 - results_media_type: required, 'movie'|'tv'|'all' and used to filter returned neighbors.
""",
)
def mcp_knn(payload: KNNRequest) -> KNNResponse:
    try:
        res = call_mcp_knn(payload)
        # validate/convert response via MCPResponse
        return KNNResponse.model_validate(res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("mcp_knn error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/will-like", response_model=WillLikeResponse)
def mcp_will_like(payload: WillLikeRequest) -> WillLikeResponse:
    """Determine whether the current user is likely to like the given TMDB item.

    Accepts either:
      - {"tmdb_id": <int>, "media_type": "movie"|"tv"}
      - {"title": "Some Movie/Show Name", "media_type": "movie"|"tv"}

    Returns: {will_like: bool, score: float, explanation: str, item: {...}}
    """
    try:
        try:
            res = compute_will_like(payload.tmdb_id, payload.title, payload.media_type)
            return WillLikeResponse.model_validate(res)
        except WillLikeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("mcp_will_like error: %s", repr(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("mcp_will_like error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/clear-history-cache", response_model=AdminAckResponse)
def admin_clear_history_cache():
    """Admin endpoint to clear in-memory history cache."""
    try:
        ok = clear_history_cache()
        if ok:
            return AdminAckResponse(status="cleared", message="history cache cleared")
        else:
            raise HTTPException(status_code=500, detail="failed")
    except Exception as e:
        logger.error("admin_clear_history_cache error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/sync/status", response_model=SyncStatusResponse)
def admin_sync_status():
    """Return last sync timestamps for Trakt and TMDB (movie, tv).

    Response shape:
    {
        "trakt_last_activity": <iso string or null>,
        "tmdb_movie_last_sync": <iso string or null>,
        "tmdb_tv_last_sync": <iso string or null>
    }
    """
    try:
        from app.db import sync_meta_collection

        def _get(key, media_type: str = None):
            try:
                filter = {
                    "_id": {"$regex": f"^{key}"},
                    "status": "completed",
                }
                if media_type:
                    filter["media_type"] = media_type
                doc = sync_meta_collection.find(
                    filter,
                    {
                        "_id": 0,
                        "finished_at": 1,
                    },
                ).sort("finished_at", DESCENDING).limit(1)
                doc = list(doc)
                if not doc:
                    return None
                # convert timestamp to iso string
                finished_at = doc[0]["finished_at"]
                if finished_at:
                    return datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat()
                return finished_at
            except Exception:
                return None

        status = {
            "trakt_last_activity": _get("trakt_sync_job"),
            "tmdb_movie_last_sync": _get("tmdb_sync_job", "movie"),
            "tmdb_tv_last_sync": _get("tmdb_sync_job", "tv"),
        }
        return SyncStatusResponse(**status)
    except Exception as e:
        logger.error("admin_sync_status error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/sync/tmdb", response_model=AdminJobAcceptedResponse, status_code=202)
def admin_sync_tmdb(background_tasks: BackgroundTasks, payload: AdminSyncTMDBRequest):
    """Trigger a TMDB sync job in background.

    Payload options:
      - media_type: 'movie' or 'tv' (required)
      - full_sync: bool (default False)
      - embed_updated: bool (default True)
      - force_refresh: bool (default False)
    Returns job_id to poll status.
    """
    try:
        media_type = payload.media_type
        if media_type not in ("movie", "tv"):
            raise HTTPException(status_code=400, detail="media_type must be 'movie' or 'tv'")
        full_sync = bool(payload.full_sync)
        embed_updated = bool(payload.embed_updated)
        force_refresh = bool(payload.force_refresh)

        job_id = str(uuid.uuid4())
        job_key = f"tmdb_sync_job:{job_id}"
        started_at = int(_time.time())
        sync_meta_collection.update_one({"_id": job_key}, {
            "$set": {"status": "pending", "started_at": started_at, "media_type": media_type}}, upsert=True)

        def _run_tmdb_job(jid: str, mtype: str, full: bool, embed_u: bool, force: bool):
            key = f"tmdb_sync_job:{jid}"
            try:
                # check if job was canceled before the worker started; if so, honor it and skip running
                existing = sync_meta_collection.find_one({"_id": key})
                if existing and (existing.get("cancel") or existing.get("status") == "canceled"):
                    # ensure canceled state persisted with finished_at
                    try:
                        sync_meta_collection.update_one({"_id": key}, {
                            "$set": {"status": "canceled", "finished_at": int(_time.time())}})
                    except Exception:
                        pass
                    return

                sync_meta_collection.update_one({"_id": key},
                                                {"$set": {"status": "running", "last_update": int(_time.time())}})
                res = sync_tmdb(mtype, full_sync=full, embed_updated=embed_u, force_refresh=force, job_id=jid)
                # if the sync reported it was canceled, mark job as canceled; otherwise complete it
                try:
                    current = sync_meta_collection.find_one({"_id": key}) or {}
                    already_canceled = bool(current.get("cancel") or current.get("status") == "canceled")
                    if already_canceled or (isinstance(res, dict) and res.get("canceled")):
                        sync_meta_collection.update_one({"_id": key}, {
                            "$set": {"status": "canceled", "finished_at": int(_time.time()), "result": res}},
                                                        upsert=True)
                    else:
                        sync_meta_collection.update_one({"_id": key}, {
                            "$set": {"status": "completed", "finished_at": int(_time.time()), "result": res}},
                                                        upsert=True)
                except Exception:
                    # best-effort final update
                    try:
                        sync_meta_collection.update_one({"_id": key}, {
                            "$set": {"status": "completed", "finished_at": int(_time.time()), "result": res}},
                                                        upsert=True)
                    except Exception:
                        pass
            except Exception as e:
                # If job was canceled by user, avoid overwriting canceled status with 'failed'.
                try:
                    current = sync_meta_collection.find_one({"_id": key}) or {}
                    already_canceled = bool(current.get("cancel") or current.get("status") == "canceled")
                    if already_canceled:
                        # preserve canceled state and record the error in result/error fields
                        try:
                            sync_meta_collection.update_one({"_id": key}, {
                                "$set": {"status": "canceled", "finished_at": int(_time.time()), "error": str(e),
                                         "trace": _traceback.format_exc()}}, upsert=True)
                        except Exception:
                            pass
                    else:
                        try:
                            sync_meta_collection.update_one({"_id": key}, {
                                "$set": {"status": "failed", "finished_at": int(_time.time()), "error": str(e),
                                         "trace": _traceback.format_exc()}}, upsert=True)
                        except Exception:
                            pass
                except Exception:
                    pass

        background_tasks.add_task(_run_tmdb_job, job_id, media_type, full_sync, embed_updated, force_refresh)
        return AdminJobAcceptedResponse(status="accepted", job_id=job_id)
    except Exception as e:
        logger.error("admin_sync_tmdb error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/sync/tmdb/cancel", response_model=AdminAckResponse, status_code=202)
def admin_cancel_tmdb(payload: AdminCancelJobRequest):
    """Set cancel flag on running TMDB sync job. Expects JSON: {"job_id": "..."}"""
    try:
        job_id = payload.job_id
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")
        key = f"tmdb_sync_job:{job_id}"
        doc = sync_meta_collection.find_one({"_id": key})
        if not doc:
            raise HTTPException(status_code=404, detail="job not found")
        now = int(_time.time())
        # If job hasn't started (pending) mark it canceled immediately
        status = doc.get("status")
        if status in (None, "pending", "running"):
            sync_meta_collection.update_one({"_id": key}, {
                "$set": {"cancel": True, "cancel_requested_at": now, "status": "canceled", "finished_at": now}},
                                            upsert=True)
            return AdminAckResponse(status="accepted", message="job canceled")
        # Otherwise (already completed/failed/canceled), just set the cancel flag for record
        sync_meta_collection.update_one({"_id": key}, {"$set": {"cancel": True, "cancel_requested_at": now}})
        return AdminAckResponse(status="accepted", message="cancel requested")
    except Exception as e:
        logger.error("admin_cancel_tmdb error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/sync/jobs")
def admin_list_sync_jobs():
    """Return a list of sync job documents that are not completed.

    This endpoint is used by the UI to let admins select and manage
    running or recent jobs. It returns a list of documents with fields:
      - key: the full _id stored in sync_meta (e.g. 'tmdb_sync_job:<uuid>')
      - job_type: 'tmdb'|'trakt' (derived from key)
      - status and other stored metadata (processed, embed_queued, started_at, ...)
    """
    try:
        from app.db import sync_meta_collection

        # find job docs whose _id ends with '_sync_job:<id>' and which are not completed
        # Using regex to roughly match keys like 'tmdb_sync_job:' or 'trakt_sync_job:'
        cursor = sync_meta_collection.find({"_id": {"$regex": "_sync_job:"}, "status": {"$ne": "completed"}},
                                           {"_id": 1, "status": 1, "processed": 1, "embed_queued": 1, "started_at": 1,
                                            "last_update": 1, "cancel": 1})
        results = []
        for d in cursor:
            key = d.get("_id")
            job_type = None
            if isinstance(key, str):
                if key.startswith("tmdb_sync_job:"):
                    job_type = "tmdb"
                elif key.startswith("trakt_sync_job:"):
                    job_type = "trakt"
            entry = {"key": key, "job_type": job_type}
            # merge selected metadata fields
            for f in ("status", "processed", "embed_queued", "started_at", "last_update", "cancel"):
                if f in d:
                    entry[f] = d.get(f)
            results.append(entry)
        return {"jobs": results}
    except Exception as e:
        logger.error("admin_list_sync_jobs error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/clusters")
def get_watch_history_clusters(
        media_type: str = None,
        n_clusters: int = Query(default=6, ge=3, le=15)
):
    """Generate 2D clustered visualization of watch history using embeddings.

    Query params:
      - media_type: optional 'movie'|'tv' filter
      - n_clusters: number of clusters (3-15, default 6)

    Returns:
      - items: list of items with x, y coordinates, cluster labels, and metadata
      - cluster_summaries: dict mapping cluster_id to summary info with LLM-generated names
    """
    try:
        # fetch watch history with embeddings
        history = get_watch_history(media_type=media_type, include_posters=True)

        if not history:
            logger.warning("No watch history found")
            return {"items": [], "cluster_summaries": {}}

        logger.info(f"Found {len(history)} items in watch history")

        # collect all (id, media_type) pairs from history
        id_media_pairs = []
        for item in history:
            tmdb_id = item.get("id") or (item.get("ids") or {}).get("tmdb")
            item_media_type = item.get("media_type")
            if tmdb_id and item_media_type:
                try:
                    id_media_pairs.append((int(tmdb_id), item_media_type))
                except (ValueError, TypeError):
                    continue

        if not id_media_pairs:
            logger.warning("No valid TMDB IDs found in watch history")
            return {"items": [], "cluster_summaries": {}}

        # build unique (id, media_type) pairs and fetch embeddings from FAISS sidecars
        unique_pairs = list(set(id_media_pairs))
        ids_list = [p[0] for p in unique_pairs]
        mts_list = [p[1] for p in unique_pairs]
        logger.info(f"Fetching embeddings for {len(unique_pairs)} unique TMDB (id,media_type) pairs")

        from app.faiss_index import get_vectors_for_ids

        vecs = get_vectors_for_ids(ids_list, media_types=mts_list)

        # build lookup map keyed by (id, media_type)
        embeddings_map = {}
        for (tmdb_id, mt), v in zip(unique_pairs, vecs):
            try:
                key = (int(tmdb_id), str(mt).lower())
            except (ValueError, TypeError):
                continue
            if v is None:
                continue
            # fetch metadata doc to pull genres and overview (non-embedding fields)
            doc = tmdb_metadata_collection.find_one({"id": int(tmdb_id), "media_type": str(mt).lower()}, {"_id": 0, "genres": 1, "overview": 1, "title": 1, "poster_path": 1, "backdrop_path": 1})
            entry = {"embedding": list(v)}
            if doc:
                entry.update(doc)
            embeddings_map[key] = entry

        # filter items that have embeddings
        items_with_embeddings = []
        for item in history:
            tmdb_id = item.get("id") or (item.get("ids") or {}).get("tmdb")
            item_media_type = item.get("media_type")

            if not tmdb_id or not item_media_type:
                continue

            try:
                key = (int(tmdb_id), str(item_media_type).lower())
                doc = embeddings_map.get(key)
                if doc and doc.get("embedding"):
                    item["embedding"] = doc["embedding"]
                    if doc.get("genres"):
                        item["genres"] = doc.get("genres")
                    # copy overview if present
                    if doc.get("overview"):
                        item["overview"] = doc.get("overview")
                    items_with_embeddings.append(item)
            except (ValueError, TypeError):
                continue

        logger.info(f"Matched {len(items_with_embeddings)} items with embeddings")

        if len(items_with_embeddings) < n_clusters:
            # not enough items with embeddings for clustering
            logger.warning(f"Only {len(items_with_embeddings)} items with embeddings, requested {n_clusters} clusters")
            n_clusters = max(2, len(items_with_embeddings) // 2) if len(items_with_embeddings) >= 4 else 1

        if not items_with_embeddings:
            return {"items": [], "cluster_summaries": {}}

        # extract embeddings matrix
        embeddings = np.array([item["embedding"] for item in items_with_embeddings], dtype=np.float32)

        # perform clustering
        if len(items_with_embeddings) > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
        else:
            cluster_labels = np.array([0])

        # dimensionality reduction to 2D using t-SNE
        if embeddings.shape[0] > 1:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1))
            coords_2d = reducer.fit_transform(embeddings)
        else:
            coords_2d = np.array([[0.0, 0.0]])

        # build result items with coordinates and cluster labels
        result_items = []
        for idx, item in enumerate(items_with_embeddings):
            result_items.append({
                "id": item.get("id") or item.get("ids", {}).get("tmdb"),
                "title": item.get("title", "Unknown"),
                "media_type": item.get("media_type"),
                "poster_path": item.get("poster_path"),
                "year": item.get("year"),
                "x": float(coords_2d[idx, 0]),
                "y": float(coords_2d[idx, 1]),
                "cluster": int(cluster_labels[idx]),
                "watch_count": item.get("watch_count", 1),
                "completion_ratio": item.get("completion_ratio", 0),  # for TV shows
                "overview": item.get("overview", ""),
                "genres": item.get("genres", [])
            })

        # generate cluster summaries
        cluster_groups = defaultdict(list)
        for item in result_items:
            cluster_groups[item["cluster"]].append(item)

        # generate LLM-based cluster names
        cluster_summaries = {}
        for cluster_id, cluster_items in cluster_groups.items():
            titles = [item["title"] for item in cluster_items]
            media_types = [item["media_type"] for item in cluster_items]
            movie_count = sum(1 for mt in media_types if mt == "movie")
            tv_count = sum(1 for mt in media_types if mt == "tv")

            # extract genres from cluster items
            all_genres = []
            for item in cluster_items:
                if item.get("genres"):
                    for genre in item["genres"]:
                        # genres can be dicts like {"id": 28, "name": "Action"} or strings
                        if isinstance(genre, dict):
                            genre_name = genre.get("name")
                            if genre_name:
                                all_genres.append(genre_name)
                        elif isinstance(genre, str):
                            all_genres.append(genre)

            # get top genres
            genre_counts = Counter(all_genres)
            top_genres = [g for g, _ in genre_counts.most_common(3)]

            # generate cluster name using LLM
            cluster_name = generate_cluster_name(
                titles[:20],  # use up to 20 titles
                top_genres,
                movie_count,
                tv_count
            )

            cluster_summaries[str(cluster_id)] = {
                "size": len(cluster_items),
                "sample_titles": titles[:5],
                "movie_count": movie_count,
                "tv_count": tv_count,
                "name": cluster_name,
                "top_genres": top_genres
            }

        return {
            "items": result_items,
            "cluster_summaries": cluster_summaries,
            "total_items": len(result_items),
            "total_in_history": len(history),
            "n_clusters": n_clusters,
            "method": "tsne"
        }

    except Exception as e:
        logger.error("get_watch_history_clusters error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
