from __future__ import annotations

import os
from typing import List, Optional, Tuple
import json
import time

import faiss
import numpy as np
import threading

from app.db import tmdb_metadata_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "tmdb.index")
LABELS_FILE = os.path.join(INDEX_DIR, "labels.npy")
VECS_FILE = os.path.join(INDEX_DIR, "vecs.npy")
SIDECAR_META_FILE = os.path.join(INDEX_DIR, "sidecar_meta.json")
# No separate metadata file: TMDB ids are stored as labels inside the FAISS index or sidecars

# GPU config
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "false").lower() == "true"
FAISS_GPU_ID = int(os.getenv("FAISS_GPU_ID", "0"))

# in-memory sidecar cache
# _labels and _vecs are None until sidecars are loaded; _label_to_index is an empty dict to avoid undefined warnings
_labels: Optional[np.ndarray] = None
_vecs: Optional[np.ndarray] = None
_label_to_index: dict = {}

# in-memory FAISS index cache to avoid repeated disk reads
_cached_index: Optional["faiss.Index"] = None
_index_load_lock = threading.Lock()


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


def _encode_label(tmdb_id: int, media_type: Optional[str]) -> int:
    # encode: shift id left by 1 and set low bit: 0=movie,1=tv
    try:
        mt = 0 if (str(media_type or "").lower() == "movie") else 1
    except Exception:
        mt = 0
    return (int(tmdb_id) << 1) | int(mt)


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


def load_sidecars() -> None:
    """Load labels.npy and vecs.npy into module-level cache. No-op if already loaded."""
    global _labels, _vecs, _label_to_index
    if _labels is not None and _vecs is not None:
        return
    if not os.path.exists(LABELS_FILE) or not os.path.exists(VECS_FILE):
        logger.info("Sidecar files missing: %s or %s", LABELS_FILE, VECS_FILE)
        _labels = None
        _vecs = None
        _label_to_index = {}
        return
    try:
        _labels = np.load(LABELS_FILE)
        _vecs = np.load(VECS_FILE)
        # build map for fast lookup
        _label_to_index = {int(l): i for i, l in enumerate(_labels.tolist())}
        logger.info("Loaded sidecars: %s vectors", len(_labels))
    except Exception as e:
        logger.error("Failed to load sidecars: %s", repr(e), exc_info=True)
        _labels = None
        _vecs = None
        _label_to_index = {}


def load_sidecar_meta() -> Optional[dict]:
    """Load sidecar meta JSON if present. Returns dict or None."""
    try:
        if not os.path.exists(SIDECAR_META_FILE):
            return None
        with open(SIDECAR_META_FILE, "r") as f:
            meta = json.load(f)
            return meta
    except Exception as e:
        logger.warning("Failed to read sidecar_meta: %s", repr(e))
        return None


def write_sidecar_meta(meta: dict) -> None:
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(SIDECAR_META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Wrote sidecar meta: %s", SIDECAR_META_FILE)
    except Exception as e:
        logger.warning("Failed to write sidecar_meta: %s", repr(e), exc_info=True)


def get_vectors_for_ids(ids: List[int], media_types: Optional[List[str]] = None) -> List[Optional[np.ndarray]]:
    """Return list of vectors corresponding to provided ids and optional media_types.

    If a vector is missing, the corresponding list entry will be None.
    This function prefers sidecars (labels.npy/vecs.npy). If sidecars are missing, it will
    attempt to load the FAISS index and call index.reconstruct on each label when supported.
    """
    load_sidecars()
    results: List[Optional[np.ndarray]] = []

    # normalize media_types
    mts = media_types or [None] * len(ids)
    if len(mts) == 1 and len(ids) > 1:
        mts = mts * len(ids)
    for i, tmdb_id in enumerate(ids):
        mt = mts[i] if i < len(mts) else None
        label = _encode_label(tmdb_id, mt)
        if _label_to_index is not None:
            idx = _label_to_index.get(label)
            if idx is not None and _vecs is not None:
                results.append(np.asarray(_vecs[idx], dtype=np.float32))
                continue
        # fallback: try index.reconstruct if sidecars aren't present
        try:
            idx = _encode_label(tmdb_id, mt)
            idx = int(idx)
            # attempt to load index and reconstruct
            index = load_faiss_index()
            if index is not None and hasattr(index, "reconstruct"):
                try:
                    vec = index.reconstruct(idx)
                    results.append(np.asarray(vec, dtype=np.float32))
                    continue
                except Exception:
                    pass
        except Exception:
            pass
        results.append(None)
    return results


def build_faiss_index(
    dim: int,
    index_factory: str = "IDMap,IVF100,Flat",
    batch_size: int = 256,
    reuse_sidecars: bool = True,
) -> Optional["faiss.Index"]:
    """Build FAISS index from metadata by computing embeddings and persist CPU index and sidecars; optionally return GPU index."""
    # declare module-level caches as globals since we will assign to them later
    global _labels, _vecs, _label_to_index
    os.makedirs(INDEX_DIR, exist_ok=True)

    # fetch all metadata docs (we need fields for embedding computation)
    docs = list(
        tmdb_metadata_collection.find({}, {"_id": 0})
    )
    if not docs:
        logger.info("No TMDB metadata docs found to build FAISS index")
        return None

    # lazy import to avoid circular import at module load
    from app.embeddings import build_weighted_embedding_for_item

    labels_list = []
    vecs_list = []

    # if requested, try to reuse existing sidecars to avoid recomputing vectors
    existing_labels = None
    existing_vecs = None
    existing_label_to_index = None
    if reuse_sidecars:
        load_sidecars()
        sidecar_meta = load_sidecar_meta()
        # auto-detect embedding_model changes: if model differs, ignore sidecars
        try:
            from app.embeddings import EMBED_MODEL_NAME
            if sidecar_meta and sidecar_meta.get("embedding_model") and sidecar_meta.get("embedding_model") != EMBED_MODEL_NAME:
                logger.warning(
                    "Sidecar meta embedding_model (%s) differs from current model (%s); ignoring sidecars and recomputing",
                    sidecar_meta.get("embedding_model"), EMBED_MODEL_NAME,
                )
                existing_labels = None
                existing_vecs = None
                existing_label_to_index = None
        except Exception:
            # if we can't import EMBED_MODEL_NAME for any reason, continue without auto-detect
            pass

        if _labels is not None and _vecs is not None and _label_to_index is not None:
            existing_labels = _labels
            existing_vecs = _vecs
            existing_label_to_index = _label_to_index
            # check dimension match
            if existing_vecs is not None and existing_vecs.size and existing_vecs.shape[1] != dim:
                logger.warning(
                    "Existing sidecar vectors have dim=%s but requested dim=%s; ignoring sidecars and recomputing",
                    existing_vecs.shape[1], dim,
                )
                existing_labels = None
                existing_vecs = None
                existing_label_to_index = None

    logger.info("Preparing vectors for %s docs (batch_size=%s, reuse_sidecars=%s)...", len(docs), batch_size, bool(existing_labels is not None))
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        # attempt to compute embeddings for items missing from sidecars in this batch
        # construct output lists in the same order as docs
        for d in batch:
            try:
                lbl = _encode_label(d.get("id"), d.get("media_type"))
                if existing_label_to_index is not None and int(lbl) in existing_label_to_index:
                    idx = existing_label_to_index[int(lbl)]
                    vec = np.asarray(existing_vecs[idx], dtype=np.float32)
                    labels_list.append(int(lbl))
                    vecs_list.append(vec)
                    continue

                # compute vector since not found in existing sidecars
                v = build_weighted_embedding_for_item(d)
                if v is None:
                    continue
                vec = np.asarray(v, dtype=np.float32)
                labels_list.append(lbl)
                vecs_list.append(vec)
            except Exception as e:
                logger.warning("Failed to compute or reuse embedding for id=%s: %s", d.get("id"), repr(e))

    if not vecs_list:
        logger.info("No vectors computed; aborting FAISS build")
        return None

    labels = np.array(labels_list, dtype=np.int64)
    vecs = np.array(vecs_list, dtype=np.float32)

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

    # persist sidecars for robust vector retrieval
    try:
        np.save(LABELS_FILE, labels)
        np.save(VECS_FILE, vecs)
        logger.info("Saved sidecars: %s (labels) and %s (vecs)", LABELS_FILE, VECS_FILE)
        # refresh cache
        _labels = labels
        _vecs = vecs
        _label_to_index = {int(l): i for i, l in enumerate(_labels.tolist())}
        # write sidecar meta
        try:
            from app.embeddings import EMBED_MODEL_NAME
            meta = {
                "embedding_model": EMBED_MODEL_NAME,
                "embedding_ts": int(time.time()),
                "embedding_dims": int(vecs.shape[1]),
                "num_vectors": int(labels.shape[0]),
            }
            write_sidecar_meta(meta)
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed to write sidecars: %s", repr(e), exc_info=True)

    # persist index and update module-level cache so future calls don't re-read from disk
    try:
        _set_cached_index(index)
    except Exception:
        pass

    if FAISS_USE_GPU:
        try:
            gpu_index = _to_gpu_index(index, FAISS_GPU_ID)
            logger.info("Transferred FAISS index to GPU")
            _set_cached_index(gpu_index)
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
    global _cached_index
    # fast path: return cached index if present
    if _cached_index is not None:
        logger.info("Loaded FAISS index from in-memory cache")
        return _cached_index

    # acquire lock to ensure only one thread loads index from disk at a time
    with _index_load_lock:
        # double-check cached value after acquiring lock
        if _cached_index is not None:
            return _cached_index

        if not os.path.exists(INDEX_FILE):
            logger.info("FAISS index file missing")
            return None
        try:
            cpu_index = faiss.read_index(INDEX_FILE)
            logger.info("Loaded FAISS index from %s", INDEX_FILE)
            # if GPU is requested, attempt transfer; otherwise keep CPU index
            if FAISS_USE_GPU:
                try:
                    gpu_index = _to_gpu_index(cpu_index, FAISS_GPU_ID)
                    logger.info("Loaded FAISS index and moved to GPU")
                    _cached_index = gpu_index
                    return gpu_index
                except Exception as e:
                    logger.warning(
                        "Failed to move FAISS index to GPU, using CPU index: %s",
                        repr(e),
                        exc_info=True,
                    )
                    _cached_index = cpu_index
                    return cpu_index
            _cached_index = cpu_index
            return cpu_index
        except Exception as e:
            logger.error("Failed to read FAISS index: %s", repr(e), exc_info=True)
            return None


def _set_cached_index(index: "faiss.Index") -> None:
    """Set module-level cached FAISS index."""
    global _cached_index
    try:
        _cached_index = index
    except Exception:
        _cached_index = None


def clear_index_cache() -> None:
    """Clear the in-memory FAISS index cache (call after external writes/rebuilds)."""
    global _cached_index
    _cached_index = None


def is_index_cached() -> bool:
    """Return True if the FAISS index is currently cached in this process."""
    global _cached_index
    return _cached_index is not None


def query_faiss(
    index: "faiss.Index", query_vec: np.ndarray, k: int = 10
) -> List[Tuple[int, str, float]]:
    """Query FAISS index with the given vector and return list of (tmdb_id, media_type, score) tuples.

    Labels are decoded from the composite label encoding used during build.
    """

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


def upsert_single_item(tmdb_id: int, media_type: str = "movie", force_regenerate: bool = False) -> dict:
    """Incrementally upsert a single item's vector into sidecars and FAISS index.

    Behavior:
      - If sidecars/index are missing, or force_regenerate=True: trigger full rebuild via build_faiss_index and return {'status': 'rebuild_scheduled'}.
      - Otherwise, compute vector for item, update sidecars (replace or append), and try to update the persisted FAISS index in-place.
      - If in-place update isn't supported by the installed FAISS (no remove_ids/add_with_ids available), fall back to scheduling a full rebuild and return that status.

    Returns a small dict with status: 'updated'|'added'|'rebuild_scheduled' and optional detail.
    """
    # assign module-level sidecar caches below; declare globals up front
    global _labels, _vecs, _label_to_index

    try:
        # lazy import for embedding builder and DB access
        from app.embeddings import build_weighted_embedding_for_item
        doc = tmdb_metadata_collection.find_one({"id": int(tmdb_id), "media_type": str(media_type).lower()}, {"_id": 0})
        if not doc:
            return {"status": "not_found", "message": "metadata not found"}

        # compute vector
        vec = build_weighted_embedding_for_item(doc)
        if vec is None:
            return {"status": "no_vector", "message": "embedding computation returned None"}
        vec = np.asarray(vec, dtype=np.float32)
        label = int(_encode_label(int(tmdb_id), str(media_type).lower()))

        # if force regenerate or sidecars missing -> schedule rebuild
        load_sidecars()
        if force_regenerate or _labels is None or _vecs is None:
            # schedule full rebuild (caller should call build_faiss_index in background)
            return {"status": "rebuild_required", "message": "sidecars missing or force_regenerate requested"}

        # ensure caches available
        if (not _label_to_index) and (_labels is not None):
            _label_to_index = {int(l): i for i, l in enumerate(_labels.tolist())}

        if label in _label_to_index:
            # replace existing vector in sidecars
            idx = int(_label_to_index[label])
            _vecs[idx] = vec
            try:
                np.save(VECS_FILE, _vecs)
                # update sidecar_meta timestamp
                try:
                    meta = load_sidecar_meta() or {}
                    meta["embedding_ts"] = int(time.time())
                    write_sidecar_meta(meta)
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Failed saving vecs after upsert: %s", repr(e))

            # attempt to update FAISS index in-place
            try:
                idx_obj = load_faiss_index()
                if idx_obj is None:
                    return {"status": "rebuild_required", "message": "index not found on disk"}

                # remove old id if supported
                try:
                    if hasattr(idx_obj, "remove_ids"):
                        idx_obj.remove_ids(np.array([label], dtype=np.int64))
                except Exception as e:
                    logger.warning("remove_ids not supported or failed: %s", repr(e))

                # ensure index supports add_with_ids
                if not hasattr(idx_obj, "add_with_ids"):
                    idx_obj = faiss.IndexIDMap(idx_obj)

                idx_obj.add_with_ids(np.ascontiguousarray(vec.reshape(1, -1)), np.array([label], dtype=np.int64))
                # persist updated index
                try:
                    faiss.write_index(idx_obj, INDEX_FILE)
                    # update in-memory cached index to the modified object
                    try:
                        _set_cached_index(idx_obj)
                    except Exception:
                        pass
                except Exception:
                    # if writing back fails, still consider sidecars updated
                    logger.warning("Failed to write updated FAISS index after upsert; sidecars updated")
                return {"status": "updated", "message": "vector updated in sidecars and index"}
            except Exception as e:
                logger.warning("In-place index update not supported/failure: %s", repr(e))
                return {"status": "rebuild_required", "message": "in-place index update not supported"}
        else:
            # append new label/vector to sidecars
            try:
                new_labels = np.concatenate([_labels, np.array([label], dtype=np.int64)]) if _labels is not None else np.array([label], dtype=np.int64)
                new_vecs = np.concatenate([_vecs, vec.reshape(1, -1)]) if _vecs is not None else vec.reshape(1, -1)
                np.save(LABELS_FILE, new_labels)
                np.save(VECS_FILE, new_vecs)
                # refresh cache
                _labels = new_labels
                _vecs = new_vecs
                _label_to_index = {int(l): i for i, l in enumerate(_labels.tolist())}
                # update sidecar_meta
                try:
                    meta = load_sidecar_meta() or {}
                    meta["embedding_ts"] = int(time.time())
                    meta["embedding_dims"] = int(_vecs.shape[1])
                    meta["num_vectors"] = int(_labels.shape[0])
                    write_sidecar_meta(meta)
                except Exception:
                    pass
            except Exception as e:
                logger.error("Failed to append to sidecars: %s", repr(e), exc_info=True)
                return {"status": "error", "message": "failed updating sidecars"}

            # try to add to existing FAISS index without full rebuild
            try:
                idx_obj = load_faiss_index()
                if idx_obj is None:
                    return {"status": "rebuild_required", "message": "index missing; sidecars updated"}

                if not hasattr(idx_obj, "add_with_ids"):
                    idx_obj = faiss.IndexIDMap(idx_obj)

                idx_obj.add_with_ids(np.ascontiguousarray(vec.reshape(1, -1)), np.array([label], dtype=np.int64))
                try:
                    faiss.write_index(idx_obj, INDEX_FILE)
                except Exception:
                    logger.warning("Failed to write FAISS index after append; sidecars updated")
                return {"status": "added", "message": "vector appended to sidecars and index (if supported)"}
            except Exception as e:
                logger.warning("Failed to add vector to index in-place: %s", repr(e))
                return {"status": "rebuild_required", "message": "in-place add failed; sidecars updated"}
    except Exception as e:
        logger.exception("upsert_single_item failed: %s", repr(e))
        return {"status": "error", "message": str(e)}
