import requests

from app.config.settings import settings
from app.dao.history import get_watch_history
from app.trakt_sync import sync_trakt_history
from app.utils.logger import get_logger
from app.tmdb_sync import sync_tmdb_changes

logger = get_logger(__name__)


def check_trakt_last_activities_and_sync():
    """Check user's last activity on trakt and sync changes if necessary."""
    logger.info("Checking Trakt last_activities and syncing changes...")
    try:
        resp = requests.get(
            settings.TRAKT_LAST_ACTIVITIES_API_URL,
            headers=settings.trakt_headers
        )
        logger.info("Trakt last_activities response %s", resp.status_code)
        if resp.status_code != 200:
            logger.warning(
                "Failed to fetch Trakt last_activities: %s", resp.status_code)
            return

        activity = resp.json()
        relevant = {
            k: activity[k] for k in ("all", "movies", "episodes", "shows") if isinstance(activity.get(k), str)
        }

        trakt_latest = max(relevant.values(), default=None)

        db_history = get_watch_history()
        db_latest = max(
            (item.get("latest_watched_at")
             for item in db_history if item.get("latest_watched_at")),
            default=None
        )
        logger.info("Trakt latest activity: %s", trakt_latest)
        logger.info("DB latest activity: %s", db_latest)
        if db_latest and trakt_latest and trakt_latest <= db_latest:
            logger.info("No new Trakt activity since last DB update.")
            return

        logger.info("New Trakt activity detected. Syncing history...")
        sync_trakt_history()

    except Exception as e:
        logger.error("Error during Trakt last_activities check: %s", repr(e), exc_info=True)


def run_tmdb_periodic_sync():
    """Run TMDB sync for both movies and TV shows. Designed to be scheduled infrequently (e.g., every 6 hours)."""
    try:
        logger.info("Starting TMDB changes sync (movie)")
        sync_tmdb_changes(media_type="movie")
    except Exception as e:
        logger.error("TMDB movie sync failed: %s", repr(e), exc_info=True)
    try:
        logger.info("Starting TMDB changes sync (tv)")
        sync_tmdb_changes(media_type="tv")
    except Exception as e:
        logger.error("TMDB tv sync failed: %s", repr(e), exc_info=True)
