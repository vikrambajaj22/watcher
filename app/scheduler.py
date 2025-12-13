import requests
from dateutil import parser as _dateutil_parser

from app.config.settings import settings
from app.dao.history import get_watch_history
from app.db import sync_meta_collection
from app.tmdb_sync import sync_tmdb_changes
from app.trakt_sync import sync_trakt_history
from app.utils.logger import get_logger

logger = get_logger(__name__)


def check_trakt_last_activities_and_sync():
    """Check user's last activity on trakt and sync changes if necessary."""
    logger.info("Checking Trakt last_activities and syncing changes...")
    try:
        resp = requests.get(
            settings.TRAKT_LAST_ACTIVITIES_API_URL, headers=settings.trakt_headers
        )
        logger.info("Trakt last_activities response %s", resp.status_code)
        if resp.status_code != 200:
            logger.warning(
                "Failed to fetch Trakt last_activities: %s", resp.status_code
            )
            return

        activity = resp.json()
        relevant = {
            k: activity[k]
            for k in ("all", "movies", "episodes", "shows")
            if isinstance(activity.get(k), str)
        }

        trakt_latest = max(relevant.values(), default=None)

        db_history = get_watch_history()
        db_latest = max(
            (
                item.get("latest_watched_at")
                for item in db_history
                if item.get("latest_watched_at")
            ),
            default=None,
        )
        logger.info("Trakt latest activity: %s", trakt_latest)
        logger.info("DB latest activity: %s", db_latest)

        def _parse_iso(s):
            if not s:
                return None
            try:
                return _dateutil_parser.isoparse(s)
            except Exception:
                logger.debug("Failed to parse ISO timestamp: %s", s)
                return None

        t_trakt = _parse_iso(trakt_latest)
        t_db = _parse_iso(db_latest)

        # check stored last-seen trakt timestamp to avoid unnecessary full fetches
        try:
            meta = sync_meta_collection.find_one({"_id": "trakt_last_activity"})
            stored_trakt = meta.get("last_activity") if meta else None
            t_stored = _parse_iso(stored_trakt)
        except Exception:
            t_stored = None

        # if trakt reported time is <= stored remote time, nothing new since our last check
        # only skip if we already have watch history in DB; if DB is empty, force initial sync
        if db_history and t_stored and t_trakt and t_trakt <= t_stored:
            logger.info("No new Trakt activity since last checked timestamp.")
            return

        if t_db and t_trakt:
            if t_trakt <= t_db:
                logger.info("No new Trakt activity since last DB update (by datetime).")
                try:
                    if trakt_latest:
                        sync_meta_collection.update_one(
                            {"_id": "trakt_last_activity"},
                            {"$set": {"last_activity": trakt_latest}},
                            upsert=True,
                        )
                except Exception:
                    pass
                return
        else:
            # fallback to string compare if parsing failed
            if db_latest and trakt_latest and trakt_latest <= db_latest:
                logger.info("No new Trakt activity since last DB update (by string).")
                try:
                    if trakt_latest:
                        sync_meta_collection.update_one(
                            {"_id": "trakt_last_activity"},
                            {"$set": {"last_activity": trakt_latest}},
                            upsert=True,
                        )
                except Exception:
                    pass
                return

        logger.info("New Trakt activity detected. Syncing history...")
        sync_trakt_history()

        # after syncing (even if history unchanged), persist the trakt_latest timestamp so we don't re-fetch repeatedly
        try:
            if trakt_latest:
                sync_meta_collection.update_one(
                    {"_id": "trakt_last_activity"},
                    {"$set": {"last_activity": trakt_latest}},
                    upsert=True,
                )
        except Exception:
            pass

    except Exception as e:
        logger.error(
            "Error during Trakt last_activities check: %s", repr(e), exc_info=True
        )


def run_tmdb_periodic_sync():
    """Run TMDB sync for both movies and TV shows. Designed to be scheduled infrequently (e.g., every 6 hours)."""
    try:
        logger.info("Starting TMDB changes sync (movie)")
        summary = sync_tmdb_changes(media_type="movie")
        logger.info("TMDB movie sync finished: %s", summary)
    except Exception as e:
        logger.error("TMDB movie sync failed: %s", repr(e), exc_info=True)
    try:
        logger.info("Starting TMDB changes sync (tv)")
        summary = sync_tmdb_changes(media_type="tv")
        logger.info("TMDB tv sync finished: %s", summary)
    except Exception as e:
        logger.error("TMDB tv sync failed: %s", repr(e), exc_info=True)
