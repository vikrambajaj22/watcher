from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse

from app.recommender import recommend
from app.auth.trakt_auth import get_auth_url, exchange_code_for_token, save_token_data
from app.trakt_sync import sync_trakt_history

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Watcher: watchu lookin at?"}

@router.get("/recommend")
def get_recommendations():
    # Sync Trakt history before recommending
    sync_trakt_history()
    return {"recommendations": recommend()}

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
