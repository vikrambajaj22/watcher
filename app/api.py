from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.auth.trakt_auth import exchange_code_for_token, get_auth_url, save_token_data
from app.config.settings import settings
from app.dao.history import get_watch_history, clear_history_cache
from app.db import tmdb_metadata_collection, sync_meta_collection
from app.embeddings import embed_item_and_store, embed_all_items
from app.faiss_index import INDEX_DIR
from app.process.recommendation import MediaRecommender
from app.scheduler import check_trakt_last_activities_and_sync
from app.schemas.api import KNNRequest, KNNResponse, WillLikeRequest, WillLikeResponse
from app.schemas.recommendations.recommendations import (
    RecommendationsResponse,
    RecommendRequest,
)
from app.trakt_sync import sync_trakt_history
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


@router.get("/history")
def get_history(media_type: str = None, include_posters: bool = True):
    """Get watch history from database, optionally filtered by media_type and get posters.

    Query params:
      - media_type: optional 'movie'|'tv'
      - include_posters: boolean (default true) - when false, skip poster enrichment for faster responses
    """
    try:
        history = get_watch_history(media_type=media_type, include_posters=include_posters)
        return JSONResponse(history)
    except Exception as e:
        logger.error("get_history error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/sync/trakt")
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
        return JSONResponse(
            {"status": "accepted", "message": "sync started", "job_id": job_id},
            status_code=202,
        )
    except Exception as e:
        logger.error("admin_sync error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/admin/sync/job/{job_id}")
def admin_sync_job_status(job_id: str):
    """Return job status document for the given job_id (from sync_meta_collection)."""
    try:
        from app.db import sync_meta_collection

        key = f"trakt_sync_job:{job_id}"
        doc = sync_meta_collection.find_one({"_id": key}, {"_id": 0})
        if not doc:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return JSONResponse(doc)
    except Exception as e:
        logger.error("admin_sync_job_status error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


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


@router.post("/admin/embed/item")
def admin_embed_item(background_tasks: BackgroundTasks, payload: dict):
    """Trigger embedding of a single TMDB item in background. Expects JSON: {"id": <int>, "media_type": "movie"}"""
    try:
        tmdb_id = payload.get("id")
        if not tmdb_id:
            return JSONResponse({"error": "id is required"}, status_code=400)
        media_type = payload.get("media_type") or "movie"
        doc = tmdb_metadata_collection.find_one(
            {"id": tmdb_id, "media_type": media_type}, {"_id": 0}
        )
        if not doc:
            return JSONResponse({"error": "item not found"}, status_code=404)
        background_tasks.add_task(embed_item_and_store, doc)
        return JSONResponse(
            {"status": "accepted", "message": "embedding started"}, status_code=202
        )
    except Exception as e:
        logger.error("admin_embed_item error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/admin/embed/full")
def admin_embed_full(background_tasks: BackgroundTasks, payload: dict):
    """Trigger full embedding generation (background). Expects JSON: {"batch_size": <int>}"""
    try:
        batch_size = int(payload.get("batch_size", 256))
        background_tasks.add_task(embed_all_items, batch_size)
        return JSONResponse(
            {"status": "accepted", "message": "full embedding generation started"}, status_code=202
        )
    except Exception as e:
        logger.error("admin_embed_full error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/admin/faiss/rebuild")
def admin_faiss_rebuild(payload: dict):
    """Trigger FAISS rebuild in a detached process. Expects JSON: {"dim": <int>, "factory": "..."}"""
    try:
        dims = int(payload.get("dim", 384))
        factory = payload.get("factory") or "IDMap,IVF100,Flat"

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
        return JSONResponse(
            {
                "status": "accepted",
                "message": "faiss rebuild started",
                "pid": getattr(p, "pid", None),
                "log": log_path,
            },
            status_code=202,
        )
    except Exception as e:
        logger.error("admin_faiss_rebuild error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/admin/tmdb/{tmdb_id}")
def admin_get_tmdb_metadata(tmdb_id: int, media_type: str = None):
    """Debug endpoint: return stored TMDB metadata documents for the given tmdb_id.

    Optional query param `media_type` can be 'movie' or 'tv' to filter results.
    """
    try:
        q = {"id": int(tmdb_id)}
        if media_type:
            q["media_type"] = str(media_type).lower()
        cursor = list(tmdb_metadata_collection.find(q, {"_id": 0}))
        if not cursor:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(cursor)
    except Exception as e:
        logger.error("admin_get_tmdb_metadata error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


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


@router.post("/admin/clear-history-cache")
def admin_clear_history_cache():
    """Admin endpoint to clear in-memory history cache."""
    try:
        ok = clear_history_cache()
        if ok:
            return JSONResponse({"status": "cleared"}, status_code=200)
        else:
            return JSONResponse({"error": "failed"}, status_code=500)
    except Exception as e:
        logger.error("admin_clear_history_cache error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/admin/sync/status")
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
        return JSONResponse(status)
    except Exception as e:
        logger.error("admin_sync_status error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

