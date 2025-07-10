import requests

from app.config.settings import settings
from app.dao.history import get_watch_history
from app.trakt_sync import sync_trakt_history
from app.utils.logger import get_logger

logger = get_logger(__name__)


def check_trakt_last_activities_and_sync():
    """Check user's last activity on trakt and sync changes if necessary."""
    logger.info("Checking Trakt last_activities and syncing changes...")
    try:
        resp = requests.get(
            settings.TRAKT_LAST_ACTIVITIES_API_URL,
            headers=settings.trakt_headers
        )
        logger.info(f"Trakt last_activities response: {resp.status_code}")
        if resp.status_code != 200:
            logger.warning(
                f"Failed to fetch Trakt last_activities: {resp.status_code}")
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
        logger.info(f"Trakt latest activity: {trakt_latest}")
        logger.info(f"DB latest activity: {db_latest}")
        if db_latest and trakt_latest and trakt_latest <= db_latest:
            logger.info("No new Trakt activity since last DB update.")
            return

        logger.info("New Trakt activity detected. Syncing history...")
        sync_trakt_history()

    except Exception as e:
        logger.error(f"Error during Trakt last_activities check: {e}")
