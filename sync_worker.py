import time

from apscheduler.schedulers.background import BackgroundScheduler

from app.scheduler import check_trakt_last_activity_and_sync
from app.utils.logger import get_logger

logger = get_logger(__name__)


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        check_trakt_last_activity_and_sync,
        "interval",
        seconds=300,
        id="trakt_sync_job",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started for Trakt sync.")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shut down.")

if __name__ == "__main__":
    start_scheduler()
