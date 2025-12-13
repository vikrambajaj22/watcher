from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.auth.trakt_auth import exchange_code_for_token, get_auth_url, save_token_data
from app.db import tmdb_metadata_collection
from app.embeddings import embed_item_and_store, index_all_items
from app.faiss_index import INDEX_DIR
from app.process.recommendation import MediaRecommender
from app.scheduler import check_trakt_last_activities_and_sync
from app.schemas.api import MCPPayload
from app.schemas.recommendations.recommendations import (
    RecommendationsResponse,
    RecommendRequest,
)
from app.utils.llm_orchestrator import call_mcp_knn
from app.utils.logger import get_logger

import subprocess
import sys
import os

logger = get_logger(__name__)

router = APIRouter()


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
        if media_type not in ("movie", "tv"):
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


@router.get("/auth/trakt/start")
def trakt_auth_start():
    """Redirect user to Trakt OAuth authorization URL."""
    return RedirectResponse(get_auth_url())


@router.get("/auth/trakt/callback")
def trakt_auth_callback(request: Request):
    """Handle Trakt OAuth callback and exchange code for token."""
    code = request.query_params.get("code")
    if not code:
        # redirect to start if code is missing
        return RedirectResponse("/auth/trakt/start")
    try:
        token_data = exchange_code_for_token(code)
        save_token_data(token_data)
        return RedirectResponse("/docs")
    except Exception:
        return RedirectResponse("/auth/trakt/start")


@router.post("/admin/reindex/item")
def admin_reindex_item(background_tasks: BackgroundTasks, payload: dict):
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
        logger.error("admin_reindex_item error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/admin/reindex/full")
def admin_reindex_full(background_tasks: BackgroundTasks, payload: dict):
    """Trigger full indexing (background). Expects JSON: {"batch_size": <int>}"""
    try:
        batch_size = int(payload.get("batch_size", 256))
        background_tasks.add_task(index_all_items, batch_size)
        return JSONResponse(
            {"status": "accepted", "message": "full indexing started"}, status_code=202
        )
    except Exception as e:
        logger.error("admin_reindex_full error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/admin/faiss/rebuild")
def admin_faiss_rebuild(payload: dict):
    """Trigger FAISS rebuild in a detached process. Expects JSON: {"dim": <int>, "factory": "..."}"""
    try:
        dims = int(payload.get("dim", 768))
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


@router.post(
    "/mcp/knn",
    summary="Find top-k nearest neighbors",
    description="Return the top-k nearest TMDB items for a tmdb_id, free-text query, or an embedding vector. Provide exactly one of tmdb_id, text, or vector.",
)
def mcp_knn(payload: MCPPayload):
    try:
        # delegate to consolidated MCP handler
        res = call_mcp_knn(payload)
        return JSONResponse(res)
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        logger.error("mcp_knn error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
