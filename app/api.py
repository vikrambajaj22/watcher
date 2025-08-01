from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.auth.trakt_auth import (exchange_code_for_token, get_auth_url,
                                 save_token_data)
from app.process.recommendation import MovieRecommender
from app.scheduler import check_trakt_last_activities_and_sync
from app.schemas.recommendations.movies import MovieRecommendationsResponse

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Watcher: watchu lookin at?"}

@router.get("/recommend/movies")
def get_movie_recommendations() -> MovieRecommendationsResponse:
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
        return JSONResponse({"error": "Missing code in callback."}, status_code=400)
    try:
        token_data = exchange_code_for_token(code)
        save_token_data(token_data)
        return JSONResponse({"message": "Trakt authentication successful."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
