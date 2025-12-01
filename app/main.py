from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import router
from app.faiss_index import load_faiss_index
from app.utils.logger import get_logger

logger = get_logger(__name__)


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
        logger.warning("Failed to load FAISS index at startup: %s", repr(e), exc_info=True)
        yield
    finally:
        pass

app = FastAPI(title="Watcher", lifespan=lifespan)
app.include_router(router)
