from contextlib import asynccontextmanager

import os

# limit OpenMP/MKL threads to avoid thread contention/hangs when using MPS/GPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faiss

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.faiss_index import load_faiss_index
from app.utils.logger import get_logger

logger = get_logger(__name__)
faiss.omp_set_num_threads(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: load indexes (FAISS) into app.state for fast access."""
    try:
        idx = load_faiss_index()
        app.state.faiss_index = idx  # type: ignore[attr-defined]
        if idx is not None:
            logger.info("FAISS index loaded into app.state.faiss_index")
        else:
            logger.info("FAISS index not found on disk or failed to load")
        yield
    except Exception as e:
        logger.warning(
            "Failed to load FAISS index at startup: %s", repr(e), exc_info=True
        )
        yield
    finally:
        pass


app = FastAPI(title="Watcher", lifespan=lifespan)

# Browser UI (Vite dev server or deployed static origin). Comma-separated in WATCHER_CORS_ORIGINS.
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
