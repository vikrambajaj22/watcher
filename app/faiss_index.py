from __future__ import annotations

import os
from typing import Optional, List, Tuple

import numpy as np
import faiss

from app.db import tmdb_metadata_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "tmdb.index")
META_FILE = os.path.join(INDEX_DIR, "tmdb_meta.npy")

# GPU config
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "false").lower() == "true"
FAISS_GPU_ID = int(os.getenv("FAISS_GPU_ID", "0"))


def _to_gpu_index(index: faiss.Index, gpu_id: int = 0) -> faiss.Index:
    # create GPU resources and transfer index to GPU (requires faiss-gpu installation)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    return gpu_index


def build_faiss_index(dim: int, index_factory: str = "IDMAP,IVF100,Flat") -> Optional[faiss.Index]:
    """Build FAISS index from embeddings stored in Mongo and persist CPU index; optionally return GPU index."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = list(tmdb_metadata_collection.find({"embedding": {"$exists": True}}, {"_id": 0, "embedding": 1, "id": 1, "media_type": 1}))
    if not docs:
        logger.info("No items with embeddings found to build FAISS index")
        return None

    ids = np.array([d["id"] for d in docs], dtype=np.int64)
    vecs = np.array([d["embedding"] for d in docs], dtype=np.float32)

    if "," in index_factory:
        index = faiss.index_factory(vecs.shape[1], index_factory)
    else:
        index = faiss.IndexFlatIP(vecs.shape[1])

    if hasattr(index, "train") and not index.is_trained:
        logger.info("Training FAISS index...")
        index.train(vecs)
    logger.info("Adding vectors to FAISS index...")
    index.add(vecs)

    # save CPU index and ids
    faiss.write_index(index, INDEX_FILE)
    np.save(META_FILE, ids)
    logger.info("Built and saved FAISS index with %s vectors at %s", len(ids), INDEX_FILE)

    if FAISS_USE_GPU:
        try:
            gpu_index = _to_gpu_index(index, FAISS_GPU_ID)
            logger.info("Transferred FAISS index to GPU")
            return gpu_index
        except Exception as e:
            logger.warning("Failed to transfer FAISS index to GPU, continuing with CPU index: %s", repr(e), exc_info=True)
            return index
    return index


def load_faiss_index() -> Optional[faiss.Index]:
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        logger.info("FAISS index files missing")
        return None
    try:
        cpu_index = faiss.read_index(INDEX_FILE)
        if FAISS_USE_GPU:
            try:
                gpu_index = _to_gpu_index(cpu_index, FAISS_GPU_ID)
                logger.info("Loaded FAISS index and moved to GPU")
                return gpu_index
            except Exception as e:
                logger.warning("Failed to move FAISS index to GPU, using CPU index: %s", repr(e), exc_info=True)
                return cpu_index
        return cpu_index
    except Exception as e:
        logger.error("Failed to read FAISS index: %s", repr(e), exc_info=True)
        return None


def query_faiss(index: faiss.Index, query_vec: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    q = np.array(query_vec, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    D, I = index.search(q, k)
    ids = np.load(META_FILE)
    results = []
    for score_arr, idx_arr in zip(D, I):
        for score, idx in zip(score_arr, idx_arr):
            if idx < 0:
                continue
            tmdb_id = int(ids[idx])
            results.append((tmdb_id, float(score)))
    return results
