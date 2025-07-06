from apscheduler.schedulers.background import BackgroundScheduler

from app.trakt_sync import sync_trakt_history


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(sync_trakt_history, 'interval', hours=6)
    scheduler.start()
