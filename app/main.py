import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Watcher")

_cors_raw = os.getenv(
    "WATCHER_CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
)
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
