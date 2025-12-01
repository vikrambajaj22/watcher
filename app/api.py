from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.auth.trakt_auth import (exchange_code_for_token, get_auth_url,
                                 save_token_data)
from app.process.recommendation import MovieRecommender
from app.scheduler import check_trakt_last_activities_and_sync
from app.schemas.recommendations.movies import MovieRecommendationsResponse
from app.schemas.api import MCPPayload, AdminReindexPayload
from app.utils.logger import get_logger
from app.db import tmdb_metadata_collection

from app.embeddings import embed_item_and_store, index_all_items
from app.vector_store import rebuild_index, query, load_index
from app.utils.llm_orchestrator import resolve_query_vector, call_mcp_knn

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


@router.get("/recommend/movies")
def get_movie_recommendations() -> MovieRecommendationsResponse | JSONResponse:
    try:
        check_trakt_last_activities_and_sync()
        recommender = MovieRecommender()
        return recommender.generate_recommendations()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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


@router.post("/admin/reindex")
def admin_reindex(background_tasks: BackgroundTasks, payload: AdminReindexPayload):
    try:
        if payload.id:
            tmdb_id = payload.id
            media_type = payload.media_type or "movie"
            doc = tmdb_metadata_collection.find_one({"id": tmdb_id, "media_type": media_type}, {"_id": 0})
            if not doc:
                return JSONResponse({"error": "item not found"}, status_code=404)
            background_tasks.add_task(embed_item_and_store, doc)
            return JSONResponse({"status": "accepted", "message": "embedding started"}, status_code=202)

        if payload.full:
            batch_size = int(payload.batch_size or 256)
            background_tasks.add_task(index_all_items, batch_size)
            return JSONResponse({"status": "accepted", "message": "full indexing started"}, status_code=202)

        if payload.build_faiss:
            dims = int(payload.dim or 768)
            factory = payload.factory or "IDMAP,IVF100,Flat"
            background_tasks.add_task(rebuild_index, dims, factory)
            return JSONResponse({"status": "accepted", "message": "faiss build started"}, status_code=202)

        return JSONResponse({"error": "invalid payload"}, status_code=400)
    except Exception as e:
        logger.error("admin_reindex error: %s", repr(e), exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/mcp/knn", summary="Find top-k nearest neighbors", description="Return the top-k nearest TMDB items for a tmdb_id, free-text query, or an embedding vector. Provide exactly one of tmdb_id, text, or vector.")
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
