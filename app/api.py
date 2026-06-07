from datetime import datetime, timezone
from typing import Dict, List
import os
import time as _time
import traceback as _traceback
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Request, HTTPException, Query, Security
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pymongo import DESCENDING

from app.auth.trakt_auth import exchange_code_for_token, get_auth_url, save_token_data
from app.config.settings import settings
from app.dao.history import get_watch_history, clear_history_cache
from app.db import sync_meta_collection
from app.process.tmdb_recommendation import TmdbRecommender
from app.tmdb_client import get_metadata, search_by_title, search_multi
from app.tmdb_discover import fetch_cross_type_similar, fetch_similar_and_recommendations
from app.will_like import compute_will_like, WillLikeError
from app.taste_profile import compute_taste_profile
from app.scheduler import check_trakt_last_activities_and_sync
from app.schemas.api import (
    AdminAckResponse,
    AdminCancelJobRequest,
    AdminJobAcceptedResponse,
    HistoryItem,
    JobStatusModel,
    SimilarRequest,
    SimilarResponse,
    SimilarResultItem,
    SyncStatusResponse,
    WillLikeRequest,
    WillLikeResponse,
)
from app.schemas.recommendations.recommendations import (
    TmdbRecommendationsResponse,
    RecommendRequest,
)
from app.trakt_sync import sync_trakt_history
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_admin_key(api_key: str = Security(_api_key_header)):
    if not _ADMIN_API_KEY:
        return
    if api_key != _ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "watcher"}


@router.get("/search")
def search_titles(q: str = Query(""), limit: int = Query(6, ge=1, le=10)):
    """Search TMDB for movies and TV shows (typeahead helper)."""
    if not q.strip():
        return {"results": []}
    try:
        return {"results": search_multi(q, limit=limit)}
    except Exception as e:
        logger.error("search_titles error: %s", repr(e))
        raise HTTPException(status_code=500, detail="Search failed")


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


@router.post("/recommend/tmdb/{media_type}", response_model=TmdbRecommendationsResponse)
def recommend_tmdb(media_type: str, payload: RecommendRequest):
    """LLM taste plan + TMDB discover — no FAISS or metadata DB."""
    try:
        if media_type not in ("movie", "tv", "all"):
            raise HTTPException(
                status_code=400, detail="media_type must be 'movie', 'tv', or 'all'"
            )
        recommender = TmdbRecommender()
        result, debug = recommender.generate(
            media_type=media_type,
            recommend_count=payload.recommend_count,
        )
        return TmdbRecommendationsResponse(
            recommendations=result.recommendations,
            debug=debug,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("recommend_tmdb error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history", response_model=List[HistoryItem])
def get_history(media_type: str = None, include_posters: bool = True):
    try:
        history = get_watch_history(
            media_type=media_type, include_posters=include_posters
        )
        return history
    except Exception as e:
        logger.error("get_history error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/similar",
    response_model=SimilarResponse,
    dependencies=[Depends(require_admin_key)],
)
def similar_items(payload: SimilarRequest) -> SimilarResponse:
    """Find similar movies/TV shows using TMDB's similar/recommendations API."""
    try:
        tmdb_id = payload.tmdb_id
        media_type = payload.media_type
        k = payload.k

        source_title: str | None = None

        if tmdb_id is None:
            md = search_by_title(payload.title, media_type=media_type)
            if not md or not md.get("id"):
                raise HTTPException(
                    status_code=404,
                    detail=f"'{payload.title}' not found on TMDB",
                )
            tmdb_id = int(md["id"])
            source_title = md.get("title") or md.get("name") or payload.title
        else:
            try:
                md = get_metadata(tmdb_id, media_type=media_type)
                source_title = md.get("title") or md.get("name")
            except Exception:
                pass

        per_endpoint = max(10, (k + 1) // 2)
        if payload.cross_type:
            raw = fetch_cross_type_similar(tmdb_id, media_type, per_endpoint=per_endpoint)
        else:
            raw = fetch_similar_and_recommendations(tmdb_id, media_type, per_endpoint=per_endpoint)

        seen: set[int] = set()
        results: list[SimilarResultItem] = []
        for item in raw:
            iid = int(item["id"])
            if iid in seen or iid == tmdb_id:
                continue
            seen.add(iid)
            results.append(
                SimilarResultItem(
                    id=iid,
                    title=item.get("title"),
                    media_type=item.get("media_type"),
                    poster_path=item.get("poster_path"),
                    overview=item.get("overview"),
                    release_date=item.get("release_date"),
                )
            )
            if len(results) >= k:
                break

        return SimilarResponse(source_title=source_title, results=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("similar_items error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/will-like",
    response_model=WillLikeResponse,
    dependencies=[Depends(require_admin_key)],
)
def will_like(payload: WillLikeRequest) -> WillLikeResponse:
    """LLM prediction of whether the user will like a given title."""
    try:
        res = compute_will_like(payload.tmdb_id, payload.title, payload.media_type)
        return WillLikeResponse.model_validate(res)
    except WillLikeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("will_like error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/taste-profile",
    dependencies=[Depends(require_admin_key)],
)
def taste_profile():
    """LLM-generated taste profile from watch history."""
    try:
        return compute_taste_profile()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("taste_profile error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/admin/sync/trakt",
    response_model=AdminJobAcceptedResponse,
    status_code=202,
    dependencies=[Depends(require_admin_key)],
)
def admin_sync(background_tasks: BackgroundTasks):
    """Trigger Trakt history sync in background."""
    try:
        job_id = str(uuid.uuid4())
        job_key = f"trakt_sync_job:{job_id}"
        started_at = int(_time.time())
        sync_meta_collection.update_one(
            {"_id": job_key},
            {"$set": {"status": "pending", "started_at": started_at}},
            upsert=True,
        )

        def _run_sync_job(jid: str):
            job_key_inner = f"trakt_sync_job:{jid}"
            try:
                sync_meta_collection.update_one(
                    {"_id": job_key_inner}, {"$set": {"status": "running"}}
                )
                sync_trakt_history()
                sync_meta_collection.update_one(
                    {"_id": job_key_inner},
                    {"$set": {"status": "completed", "finished_at": int(_time.time())}},
                )
            except Exception as e:
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
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/admin/sync/job/{job_id}",
    response_model=JobStatusModel,
    dependencies=[Depends(require_admin_key)],
)
def admin_sync_job_status(
    job_id: str, job_type: str = Query(..., description="Job type: 'trakt'")
):
    try:
        t = (job_type or "").lower()
        if t not in ("trakt",):
            raise HTTPException(status_code=400, detail="job_type must be 'trakt'")

        if job_id.startswith("trakt_sync_job:"):
            key = job_id
        else:
            key = f"trakt_sync_job:{job_id}"

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
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/auth/trakt/start")
def trakt_auth_start(request: Request, from_ui: bool = False):
    state = "ui" if from_ui else "api"
    return RedirectResponse(get_auth_url(state=state))


@router.get("/auth/trakt/callback")
def trakt_auth_callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state", "api")
    if not code:
        return RedirectResponse("/auth/trakt/start")
    try:
        token_data = exchange_code_for_token(code)
        save_token_data(token_data)
        if state == "ui":
            ui_base = getattr(settings, "UI_BASE_URL", "http://localhost:8501")
            return RedirectResponse(ui_base)
        else:
            return RedirectResponse("/docs")
    except Exception:
        return RedirectResponse("/auth/trakt/start")


@router.get("/auth/status")
def auth_status() -> Dict[str, bool]:
    try:
        if settings.TRAKT_ACCESS_TOKEN != "":
            return {"authenticated": True}
        return {"authenticated": False}
    except Exception as e:
        logger.error("auth_status error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/auth/logout", dependencies=[Depends(require_admin_key)])
def auth_logout():
    try:
        if settings.TRAKT_ACCESS_TOKEN:
            os.remove(settings.TRAKT_TOKEN_FILE)
        return {"status": "logged_out"}
    except Exception as e:
        logger.error("auth_logout error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/admin/clear-history-cache",
    response_model=AdminAckResponse,
    dependencies=[Depends(require_admin_key)],
)
def admin_clear_history_cache():
    try:
        ok = clear_history_cache()
        if ok:
            return AdminAckResponse(status="cleared", message="history cache cleared")
        else:
            raise HTTPException(status_code=500, detail="failed")
    except Exception as e:
        logger.error("admin_clear_history_cache error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/admin/sync/status",
    response_model=SyncStatusResponse,
    dependencies=[Depends(require_admin_key)],
)
def admin_sync_status():
    try:
        def _get(key):
            try:
                doc = (
                    sync_meta_collection.find(
                        {"_id": {"$regex": f"^{key}"}, "status": "completed"},
                        {"_id": 0, "finished_at": 1},
                    )
                    .sort("finished_at", DESCENDING)
                    .limit(1)
                )
                doc = list(doc)
                if not doc:
                    return None
                finished_at = doc[0]["finished_at"]
                if finished_at:
                    return datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat()
                return None
            except Exception:
                return None

        return SyncStatusResponse(
            trakt_last_activity=_get("trakt_sync_job"),
        )
    except Exception as e:
        logger.error("admin_sync_status error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/admin/sync/jobs", dependencies=[Depends(require_admin_key)])
def admin_list_sync_jobs():
    try:
        cursor = sync_meta_collection.find(
            {"_id": {"$regex": "_sync_job:"}, "status": {"$ne": "completed"}},
            {"_id": 1, "status": 1, "processed": 1, "started_at": 1, "last_update": 1, "cancel": 1},
        )
        results = []
        for d in cursor:
            key = d.get("_id")
            job_type = None
            if isinstance(key, str):
                if key.startswith("trakt_sync_job:"):
                    job_type = "trakt"
            entry = {"key": key, "job_type": job_type}
            for f in ("status", "processed", "started_at", "last_update", "cancel"):
                if f in d:
                    entry[f] = d.get(f)
            results.append(entry)
        return {"jobs": results}
    except Exception as e:
        logger.error("admin_list_sync_jobs error: %s", repr(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
