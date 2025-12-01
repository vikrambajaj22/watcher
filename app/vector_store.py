"""Vector store adapter wrapping FAISS index operations.

This module provides a small, stable interface for the rest of the app to use FAISS as the
vector store while keeping Mongo as the metadata store. It exposes rebuild/load/query helpers.

Design choices:
- Rebuild: calls the FAISS builder which persists a CPU index and optionally transfers to GPU.
- Load: loads the persisted index into memory (and transfers to GPU per FAISS config).
- Query: ensures an index is loaded and delegates to `app.faiss_index.query_faiss`.

Note: incremental per-item upserts are expensive to implement correctly with id->index mapping.
For simplicity and correctness this adapter exposes `rebuild_index` which reindexes from the
Mongo embeddings (cheap to implement and robust)."""
from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np

from app.utils.logger import get_logger
from app.faiss_index import build_faiss_index, load_faiss_index, query_faiss

logger = get_logger(__name__)

_index = None


def rebuild_index(dim: int = 768, factory: str = "IDMAP,IVF100,Flat") -> Optional[object]:
    """Rebuilds the FAISS index from Mongo embeddings by calling the FAISS builder.
    Returns the in-memory index (maybe a GPU index if FAISS was configured to use GPU).
    """
    global _index
    logger.info("Rebuilding FAISS index (dim=%s, factory=%s)", dim, factory)
    idx = build_faiss_index(dim, index_factory=factory)
    _index = idx
    return _index


def load_index() -> Optional[object]:
    """Load persisted FAISS index into memory (returns GPU/CPU index depending on config)."""
    global _index
    if _index is None:
        logger.info("Loading FAISS index from disk into memory")
        _index = load_faiss_index()
    return _index


def query(vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    """Query the loaded index, loading it first if necessary. Returns list of (tmdb_id, score)."""
    idx = load_index()
    if idx is None:
        raise RuntimeError("FAISS index not available; run rebuild_index first")
    return query_faiss(idx, vector, k)

