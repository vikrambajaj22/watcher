import requests
import time
from app.trakt_sync import sync_trakt_history
from app.utils.logger import get_logger
from app.config.settings import settings


def poll_trakt_last_activity(interval_seconds=300):
    """
    Polls the Trakt /sync/last_activities endpoint every interval_seconds (default: 5 min).
    If activity timestamps change and are newer than the DB, triggers a full sync and updates the DB.
    """
    from app.dao.history import get_watch_history
    logger = get_logger(__name__)
    last_activity = None
    logger.info(f"Starting Trakt last_activity polling every {interval_seconds} seconds.")
    while True:
        try:
            resp = requests.get(
                f"{settings.TRAKT_ACTIVITY_API_URL}",
                headers=settings.trakt_headers
            )
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch Trakt last_activity: {resp.status_code}")
                time.sleep(interval_seconds)
                continue
            activity = resp.json()
            relevant = {k: activity[k] for k in ("all", "movies", "episodes", "shows") if k in activity}
            # Get latest activity from DB (if available)
            db_history = get_watch_history()
            db_latest = None
            if db_history:
                # Find the latest watched_at timestamp in the DB
                db_latest = max(
                    (item.get("latest_watched_at") for item in db_history if item.get("latest_watched_at")),
                    default=None
                )
            # Find the newest activity timestamp from Trakt
            trakt_latest = max(
                (v for k, v in relevant.items() if isinstance(v, str)),
                default=None
            )
            # Only sync if Trakt's latest activity is newer than DB's
            if db_latest and trakt_latest and trakt_latest <= db_latest:
                logger.debug("No new Trakt activity since last DB update.")
            elif last_activity is not None and relevant == last_activity:
                logger.debug("No Trakt activity changes detected.")
            else:
                logger.info("Trakt activity changed and is newer than DB, syncing history.")
                sync_trakt_history()
                last_activity = relevant
        except Exception as e:
            logger.error(f"Error during Trakt last_activity polling: {e}")
        time.sleep(interval_seconds)
