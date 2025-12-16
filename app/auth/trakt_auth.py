import json
import os

import requests

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

CLIENT_ID = settings.TRAKT_CLIENT_ID
CLIENT_SECRET = settings.TRAKT_CLIENT_SECRET
REDIRECT_URI = settings.TRAKT_REDIRECT_URI
TOKEN_FILE = ".env.trakt_token"

AUTH_URL = "https://trakt.tv/oauth/authorize"
TOKEN_URL = "https://api.trakt.tv/oauth/token"

HEADERS = {
    "Content-Type": "application/json",
    "trakt-api-version": "2",
    "trakt-api-key": CLIENT_ID,
}


def get_auth_url(state: str = None):
    """Generate Trakt OAuth authorization URL with optional state parameter.

    Args:
        state: Optional state parameter to track where auth was initiated (e.g., "ui" or "api")

    Returns:
        OAuth authorization URL
    """
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
    }

    if state:
        params["state"] = state

    import urllib.parse

    url = f"{AUTH_URL}?" + urllib.parse.urlencode(params)
    return url


def exchange_code_for_token(code):
    resp = requests.post(
        TOKEN_URL,
        json={
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


def save_token_data(data):
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Token saved to %s", TOKEN_FILE)


def load_token_data():
    if not os.path.exists(TOKEN_FILE):
        logger.warning("%s not found. Running initial authentication...", TOKEN_FILE)
        main()  # Run the auth flow to create the token file
    with open(TOKEN_FILE) as f:
        return json.load(f)


def refresh_token():
    data = load_token_data()
    refresh_token = data["refresh_token"]

    resp = requests.post(
        TOKEN_URL,
        json={
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    new_token_data = resp.json()
    save_token_data(new_token_data)
    logger.info("Token refreshed.")
    return new_token_data


def main():
    if os.path.exists(TOKEN_FILE):
        logger.info("Token already exists. Use --refresh to update it.")
        return
    logger.info("Go to the following URL in your browser and authorize the app:")
    logger.info(get_auth_url())
    code = input("\nPaste the code you received here: ").strip()
    token_data = exchange_code_for_token(code)
    save_token_data(token_data)
    logger.info("Initial token retrieved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh existing token")
    args = parser.parse_args()

    if args.refresh:
        refresh_token()
    else:
        main()
