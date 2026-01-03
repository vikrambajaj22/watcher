import json
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")),
        extra="allow",
    )

    TRAKT_CLIENT_ID: str
    TRAKT_CLIENT_SECRET: str
    TRAKT_TOKEN_FILE: str = ".env.trakt_token"
    TRAKT_REDIRECT_URI: str
    TMDB_API_KEY: str
    OPENAI_API_KEY: str
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "watcher"
    TRAKT_LAST_ACTIVITIES_API_URL: str = "https://api.trakt.tv/sync/last_activities"
    TRAKT_HISTORY_API_URL: str = "https://api.trakt.tv/sync/history"
    TMDB_API_URL: str = "https://api.themoviedb.org/3"
    OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
    # Base URL for the UI (Streamlit app). Used for OAuth redirect back to the UI when auth flow starts from the UI.
    UI_BASE_URL: str = "http://localhost:8501"

    @property
    def TRAKT_ACCESS_TOKEN(self):
        if os.path.exists(self.TRAKT_TOKEN_FILE):
            with open(self.TRAKT_TOKEN_FILE) as f:
                data = json.load(f)
                return data.get("access_token", "")
        return ""

    @property
    def trakt_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.TRAKT_ACCESS_TOKEN}",
            "trakt-api-version": "2",
            "trakt-api-key": self.TRAKT_CLIENT_ID,
        }


settings = Settings()
