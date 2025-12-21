from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Dict, List

from app.auth.trakt_auth import exchange_code_for_token, get_auth_url, save_token_data
from app.config.settings import settings
from app.dao.history import get_watch_history, clear_history_cache
from app.db import tmdb_metadata_collection, sync_meta_collection
from app.embeddings import embed_item_and_store, embed_all_items
from app.faiss_index import INDEX_DIR
from app.process.recommendation import MediaRecommender
from app.scheduler import check_trakt_last_activities_and_sync
from app.schemas.api import KNNRequest, KNNResponse, WillLikeRequest, WillLikeResponse
from app.schemas.api import (
    AdminAckResponse,
    AdminJobAcceptedResponse,
    AdminFaissRebuildResponse,
    SyncStatusResponse,
    JobStatusModel,
    HistoryItem,
    TMDBMetadata,
)
from app.schemas.recommendations.recommendations import (
    RecommendationsResponse,
    RecommendRequest,
)
from app.trakt_sync import sync_trakt_history
from app.tmdb_sync import sync_tmdb
from app.schemas.api import (
    AdminEmbedItemPayload,
    AdminEmbedFullPayload,
    AdminFaissRebuildPayload,
    AdminSyncTMDBRequest,
    AdminCancelJobRequest,
)

import uuid
import time as _time
import traceback as _traceback

from app.utils.llm_orchestrator import call_mcp_knn
from app.mcp_will_like import compute_will_like, WillLikeError
from app.utils.logger import get_logger

import subprocess
import sys
import os

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
def admin_sync_job_status(job_id: str):
    """Return job status document for the given job_id (from sync_meta_collection)."""
    try:
        from app.db import sync_meta_collection

        key = f"trakt_sync_job:{job_id}"
        doc = sync_meta_collection.find_one({"_id": key}, {"_id": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            return JobStatusModel.model_validate(doc)
        except Exception:
            return JobStatusModel(**doc)
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


@router.post("/admin/embed/item", response_model=AdminAckResponse, status_code=202)
def admin_embed_item(background_tasks: BackgroundTasks, payload: AdminEmbedItemPayload):
    """Trigger embedding of a single TMDB item in background. Expects JSON: {"id": <int>, "media_type": "movie"}"""
    try:
        tmdb_id = payload.id
        if not tmdb_id:
            raise HTTPException(status_code=400, detail="id is required")
        media_type = payload.media_type or "movie"
        doc = tmdb_metadata_collection.find_one(
            {"id": tmdb_id, "media_type": media_type}, {"_id": 0}
        )
        if not doc:
            raise HTTPException(status_code=404, detail="item not found")
        background_tasks.add_task(embed_item_and_store, doc)
        return AdminAckResponse(status="accepted", message="embedding started")
    except Exception as e:
        logger.error("admin_embed_item error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/embed/full", response_model=AdminAckResponse, status_code=202)
def admin_embed_full(background_tasks: BackgroundTasks, payload: AdminEmbedFullPayload):
    """Trigger full embedding generation (background). Expects JSON: {"batch_size": <int>}"""
    try:
        batch_size = int(payload.batch_size or 256)
        background_tasks.add_task(embed_all_items, batch_size)
        return AdminAckResponse(status="accepted", message="full embedding generation started")
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
        return AdminFaissRebuildResponse(status="accepted", message="faiss rebuild started", pid=getattr(p, "pid", None), log=log_path)
    except Exception as e:
        logger.error("admin_faiss_rebuild error: %s", repr(e), exc_info=True)
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
        "tmdb_movie_last_sync": <epoch int or null>,
        "tmdb_tv_last_sync": <epoch int or null>
    }
    """
    try:
        from app.db import sync_meta_collection

        def _get(key):
            try:
                doc = sync_meta_collection.find_one({"_id": key})
                if not doc:
                    return None
                return doc.get("last_sync") or doc.get("last_activity")
            except Exception:
                return None

        status = {
            "trakt_last_activity": _get("trakt_last_activity"),
            "tmdb_movie_last_sync": _get("tmdb_movie_last_sync"),
            "tmdb_tv_last_sync": _get("tmdb_tv_last_sync"),
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
        sync_meta_collection.update_one({"_id": job_key}, {"$set": {"status": "pending", "started_at": started_at, "media_type": media_type}}, upsert=True)

        def _run_tmdb_job(jid: str, mtype: str, full: bool, embed_u: bool, force: bool):
            key = f"tmdb_sync_job:{jid}"
            try:
                sync_meta_collection.update_one({"_id": key}, {"$set": {"status": "running", "last_update": int(_time.time())}})
                res = sync_tmdb(mtype, full_sync=full, embed_updated=embed_u, force_refresh=force, job_id=jid)
                sync_meta_collection.update_one({"_id": key}, {"$set": {"status": "completed", "finished_at": int(_time.time()), "result": res}}, upsert=True)
            except Exception as e:
                try:
                    sync_meta_collection.update_one({"_id": key}, {"$set": {"status": "failed", "finished_at": int(_time.time()), "error": str(e), "trace": _traceback.format_exc()}}, upsert=True)
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
        sync_meta_collection.update_one({"_id": key}, {"$set": {"cancel": True, "cancel_requested_at": int(_time.time())}})
        return AdminAckResponse(status="accepted", message="cancel requested")
    except Exception as e:
        logger.error("admin_cancel_tmdb error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
