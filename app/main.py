from fastapi import FastAPI

from app.api import router
from app.scheduler import start_scheduler

app = FastAPI(title="Watcher")
app.include_router(router)
start_scheduler()
