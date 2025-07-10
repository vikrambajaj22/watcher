import datetime
import time

from apscheduler.schedulers.background import BlockingScheduler

from app.scheduler import check_trakt_last_activities_and_sync
from app.utils.logger import get_logger

logger = get_logger(__name__)


def start_scheduler(interval=300):
    """ Start the scheduler to periodically check Trakt last activity and sync history.
    Runs every 5 minutes by default.
    """
    logger.info("Initializing Trakt sync scheduler...")
    scheduler = BlockingScheduler()
    scheduler.add_job(
        check_trakt_last_activities_and_sync,
        "interval",
        seconds=300,
        next_run_time=datetime.datetime.now(),
        id="trakt_sync_job",
        replace_existing=True
    )
    logger.info(f"Starting Trakt sync scheduler (every {interval} seconds)...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")

if __name__ == "__main__":
    start_scheduler()
