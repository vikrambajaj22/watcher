from fastapi import FastAPI

from app.api import router

app = FastAPI(title="Watcher")
app.include_router(router)
