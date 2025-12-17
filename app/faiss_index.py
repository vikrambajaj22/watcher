from __future__ import annotations

import os
from typing import List, Optional, Tuple

import faiss
import numpy as np

from app.db import tmdb_metadata_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "tmdb.index")
# No separate metadata file: TMDB ids are stored as labels inside the FAISS index

# GPU config
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "false").lower() == "true"
FAISS_GPU_ID = int(os.getenv("FAISS_GPU_ID", "0"))


def _to_gpu_index(index: "faiss.Index", gpu_id: int = 0) -> "faiss.Index":
    # create GPU resources and transfer index to GPU (requires faiss-gpu installation)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)
    return gpu_index


def _normalize_factory(factory: str) -> str:
    """Normalize common FAISS factory tokens in a case-insensitive way.

    Currently, normalizes 'idmap' -> 'IDMap' which is a common user error.
    Preserves other tokens as-is (stripped).
    """
    parts = [p.strip() for p in factory.split(",") if p.strip()]
    normalized = []
    for p in parts:
        if p.lower() == "idmap":
            normalized.append("IDMap")
        else:
            normalized.append(p)
    return ",".join(normalized)


def build_faiss_index(
    dim: int, index_factory: str = "IDMap,IVF100,Flat"
) -> Optional["faiss.Index"]:
    """Build FAISS index from embeddings stored in Mongo and persist CPU index; optionally return GPU index."""
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = list(
        tmdb_metadata_collection.find(
            {"embedding": {"$exists": True}},
            {"_id": 0, "embedding": 1, "id": 1, "media_type": 1},
        )
    )
    if not docs:
        logger.info("No items with embeddings found to build FAISS index")
        return None

    # build composite integer labels from (id, media_type) -> unique int64
    def _encode_label(tmdb_id: int, media_type: str) -> int:
        # encode: shift id left by 1 and set low bit: 0=movie,1=tv
        try:
            mt = 0 if (str(media_type or "").lower() == "movie") else 1
        except Exception:
            mt = 0
        return (int(tmdb_id) << 1) | int(mt)

    labels = np.array(
        [_encode_label(d.get("id"), d.get("media_type")) for d in docs],
        dtype=np.int64,
    )
    vecs = np.array([d["embedding"] for d in docs], dtype=np.float32)

    # normalize factory string to avoid user-provided casing issues (e.g., IDMAP vs IDMap)
    index_factory = _normalize_factory(index_factory)

    if "," in index_factory:
        index = faiss.index_factory(vecs.shape[1], index_factory)
    else:
        index = faiss.IndexFlatIP(vecs.shape[1])

    if hasattr(index, "train") and not index.is_trained:
        logger.info("Training FAISS index...")
        index.train(vecs)
    logger.info("Adding vectors to FAISS index...")

    # ensure the index stores explicit labels (TMDB ids); wrap with IndexIDMap if needed
    if not hasattr(index, "add_with_ids"):
        # wrap the base index so we can add vectors with explicit ids
        index = faiss.IndexIDMap(index)

    # add_with_ids expects ids as int64 array; ensure dtype
    try:
        index.add_with_ids(vecs, labels)
    except Exception as e:
        logger.error(
            "Failed to add vectors with ids to FAISS index: %s", repr(e), exc_info=True
        )
        raise

    # save CPU index (labels inside index) -- no separate meta file
    faiss.write_index(index, INDEX_FILE)
    logger.info(
        "Built and saved FAISS index with %s vectors at %s", len(labels), INDEX_FILE
    )

    if FAISS_USE_GPU:
        try:
            gpu_index = _to_gpu_index(index, FAISS_GPU_ID)
            logger.info("Transferred FAISS index to GPU")
            return gpu_index
        except Exception as e:
            logger.warning(
                "Failed to transfer FAISS index to GPU, continuing with CPU index: %s",
                repr(e),
                exc_info=True,
            )
            return index
    return index


def load_faiss_index() -> Optional["faiss.Index"]:
    if not os.path.exists(INDEX_FILE):
        logger.info("FAISS index file missing")
        return None
    try:
        cpu_index = faiss.read_index(INDEX_FILE)
        logger.info("Loaded FAISS index from %s", INDEX_FILE)
        if FAISS_USE_GPU:
            try:
                gpu_index = _to_gpu_index(cpu_index, FAISS_GPU_ID)
                logger.info("Loaded FAISS index and moved to GPU")
                return gpu_index
            except Exception as e:
                logger.warning(
                    "Failed to move FAISS index to GPU, using CPU index: %s",
                    repr(e),
                    exc_info=True,
                )
                return cpu_index
        return cpu_index
    except Exception as e:
        logger.error("Failed to read FAISS index: %s", repr(e), exc_info=True)
        return None


def query_faiss(
    index: "faiss.Index", query_vec: np.ndarray, k: int = 10
) -> List[Tuple[int, str, float]]:
    """Query FAISS index with the given vector and return list of (tmdb_id, media_type, score) tuples.

    Labels are decoded from the composite label encoding used during build.
    """

    def _decode_label(label: int) -> tuple[int, str]:
        # low bit indicates media_type: 0=movie,1=tv; id is label >> 1
        try:
            lid = int(label)
        except Exception:
            lid = int(label.item()) if hasattr(label, "item") else int(label)
        mt_bit = lid & 1
        tmdb_id = lid >> 1
        media = "movie" if mt_bit == 0 else "tv"
        return tmdb_id, media

    try:
        q = np.array(query_vec, dtype=np.float32)
        logger.info("Querying FAISS index using vector with shape: %s", q.shape)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # ensure contiguous C-order float32 array for FAISS
        if not q.flags["C_CONTIGUOUS"]:
            q = np.ascontiguousarray(q, dtype=np.float32)

        # ensure k is reasonable
        k = int(k) if k is not None else 10
        if k <= 0:
            k = 10

        # call search and return raw outputs (scores are distances)
        try:
            res = index.search(q, k)
        except Exception as e:
            logger.exception(
                "FAISS index.search raised exception: %s", repr(e), exc_info=True
            )
            raise

        if isinstance(res, tuple) or isinstance(res, list):
            if len(res) >= 2:
                D, idxs = res[0], res[1]
            else:
                logger.error(
                    "Unexpected FAISS search return shape (len==1); res=%s", type(res)
                )
                return []
        else:
            # some SWIG wrappers may return objects; attempt to unpack
            try:
                D = np.array(res.distances)
                idxs = np.array(res.labels)
            except Exception as e:
                logger.error(
                    "Could not interpret FAISS search return type %s: %s",
                    type(res),
                    repr(e),
                    exc_info=True,
                )
                return []

        # convert to numpy arrays
        D = np.asarray(D)
        idxs = np.asarray(idxs)

        # Some GPU indexes return float labels; coerce to int64 safely
        try:
            idxs = idxs.astype(np.int64)
        except Exception:
            # fallback element-wise cast
            idxs = np.vectorize(lambda x: int(x))(idxs)
            idxs = np.asarray(idxs, dtype=np.int64)

        # now build results: idxs shape (nq, k), D shape (nq, k)
        if D.ndim == 1:
            D = D.reshape(1, -1)
        if idxs.ndim == 1:
            idxs = idxs.reshape(1, -1)

        nq = idxs.shape[0]
        results: List[Tuple[int, str, float]] = []
        for i in range(nq):
            for j in range(min(k, idxs.shape[1])):
                idx = int(idxs[i, j])
                if idx < 0:
                    continue
                score = float(D[i, j])
                tmdb_id, media = _decode_label(idx)
                results.append((tmdb_id, media, score))
        logger.info("FAISS query returned %s total hits", len(results))
        return results
    except Exception as e:
        logger.error("FAISS query failed: %s", repr(e), exc_info=True)
        return []

