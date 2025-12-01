import datetime
import time

from apscheduler.schedulers.background import BlockingScheduler

from app.scheduler import check_trakt_last_activities_and_sync, run_tmdb_periodic_sync
from app.utils.logger import get_logger

logger = get_logger(__name__)


def start_scheduler(trakt_interval_hours=6, tmdb_interval_hours=6):
    """ Start the scheduler to periodically check Trakt last activity and sync history.
    Runs every `trakt_interval_sec` seconds for Trakt (default 5 minutes) and every `tmdb_interval_hours` hours for TMDB.
    """
    logger.info("Initializing sync scheduler...")
    scheduler = BlockingScheduler()
    scheduler.add_job(
        check_trakt_last_activities_and_sync,
        "interval",
        hours=trakt_interval_hours,
        next_run_time=datetime.datetime.now(),
        id="trakt_sync_job",
        replace_existing=True
    )

    # schedule TMDB sync (less frequent)
    scheduler.add_job(
        run_tmdb_periodic_sync,
        "interval",
        hours=tmdb_interval_hours,
        next_run_time=datetime.datetime.now(),
        id="tmdb_sync_job",
        replace_existing=True
    )

    logger.info("Starting sync scheduler (Trakt every %s hours, TMDB every %s hours)...", trakt_interval_hours, tmdb_interval_hours)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    # Default: Trakt every hour (3600s) and TMDB every 6 hours
    start_scheduler(trakt_interval_hours=6, tmdb_interval_hours=6)
