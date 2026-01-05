from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.db import tmdb_metadata_collection
from app.utils.logger import get_logger

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
logger = get_logger(__name__)

# retrieval-optimized bi-encoder recommended for semantic search/recommendation
# Use a 384-dim MiniLM model by default for compact embeddings
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE")  # can be cuda, mps or cpu

DEFAULT_WEIGHTS = {
    "text": 1.0,
    "genres": 0.6,
    "actors": 0.8,
    "directors": 1.0,
    "keywords": 0.3,
    "popularity": 0.05,
}

_model: Optional[Any] = None
_embed_lock = threading.Lock()


def _select_device() -> str:
    if EMBED_DEVICE:
        return EMBED_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    # pyTorch mps backend for Apple silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model() -> SentenceTransformer:
    global _model
    with _embed_lock:
        if _model is None:
            device = _select_device()
            logger.info(
                "Loading sentence-transformers model: %s on device %s",
                EMBED_MODEL_NAME,
                device,
            )
            # first load on CPU (no meta tensor issues)
            _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
            # IMPORTANT: run a dummy encode to force layer materialization and warmup before threaded use
            _model.encode(["dummy"], show_progress_bar=False)
            # now move safely to GPU device if needed
            if device != "cpu":
                _model.to(device)
    return _model


def _extract_text(item: Dict) -> str:
    parts: List[str] = []
    title = item.get("title") or ""
    year = None
    for date_field in ("release_date", "first_air_date"):
        if item.get(date_field):
            year = str(item.get(date_field))[:4]
            break
    if title:
        title_part = f"Title: {title}"
        if year:
            title_part += f" ({year})"
        parts.append(title_part)
    tagline = item.get("tagline")
    if tagline:
        parts.append(f"Tagline: {tagline}")
    overview = item.get("overview")
    if overview:
        parts.append(f"Overview: {overview[:2000]}")
    return ". ".join(parts)


def _extract_genres(item: Dict) -> str:
    genres = [g.get("name") for g in item.get("genres", []) if g.get("name")]
    return ", ".join(sorted(set(genres))) if genres else ""


def _extract_actors(item: Dict, top_n: int = 5) -> str:
    credits = item.get("credits") or {}
    cast_list = credits.get("cast") or item.get("cast") or []
    if not isinstance(cast_list, list):
        return ""
    try:
        sorted_cast = sorted(
            (c for c in cast_list if isinstance(c, dict)),
            key=lambda c: c.get("order", 9999),
        )
    except Exception:
        sorted_cast = [c for c in cast_list if isinstance(c, dict)]
    actors = [c.get("name") for c in sorted_cast if c.get("name")]
    return ", ".join(actors[:top_n])


def _extract_directors(item: Dict, top_n: int = 3) -> str:
    credits = item.get("credits") or {}
    crew_list = credits.get("crew") or item.get("crew") or []
    if not isinstance(crew_list, list):
        return ""
    directors = [
        c.get("name")
        for c in crew_list
        if isinstance(c, dict)
        and (c.get("job") == "Director" or c.get("department") == "Directing")
        and c.get("name")
    ]
    return ", ".join(directors[:top_n])


def _extract_keywords(item: Dict, top_n: int = 12) -> str:
    keywords = item.get("keywords") or item.get("keywords_list")
    if isinstance(keywords, dict) and keywords.get("keywords"):
        keywords = keywords.get("keywords")
    if isinstance(keywords, list):
        kw_names = [k.get("name") if isinstance(k, dict) else str(k) for k in keywords]
        kw_names = [k for k in kw_names if k]
        return ", ".join(kw_names[:top_n])
    return ""


def _extract_popularity(item: Dict) -> str:
    pop = item.get("popularity")
    if pop is None:
        return ""
    try:
        # scale/pop as string
        return f"Popularity: {float(pop):.4f}"
    except Exception:
        return ""


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def embed_text(texts: List[str]) -> np.ndarray:
    """Wrap model encode; returns numpy array of shape (len(texts), dim). Empty input returns empty array."""
    if not texts:
        return np.array([])
    try:
        model = _get_model()
        vectors = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.array(vectors)
    except Exception as e:
        logger.error("Embedding error: %s", repr(e), exc_info=True)
        # return zeros of expected dim if we can determine model dim, else empty
        try:
            # attempt a fallback dummy vec
            dummy = _get_model().encode(["dummy"], convert_to_numpy=True)
            return np.zeros((len(texts), dummy.shape[-1]))
        except Exception:
            return np.array([])


def _combine_features(
    feature_vecs: Dict[str, Optional[np.ndarray]], weights: Dict[str, float]
) -> np.ndarray:
    dim = None
    for v in feature_vecs.values():
        if v is not None and v.size:
            dim = v.shape[0]
            break
    if dim is None:
        # no features available, return zeros vector of common dim (try model dim)
        try:
            dummy = _get_model().encode(["dummy"], convert_to_numpy=True)
            dim = dummy.shape[-1]
        except Exception:
            dim = 384
    combined = np.zeros(dim, dtype=np.float32)
    for name, vec in feature_vecs.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0 or vec is None or not vec.size:
            continue
        combined += w * vec
    return _safe_normalize(combined)


def build_weighted_embedding_for_item(
    item: Dict, weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """Compute weighted embedding for a single item by embedding each feature separately."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # extract feature strings
    text = _extract_text(item)
    genres = _extract_genres(item)
    actors = _extract_actors(item)
    directors = _extract_directors(item)
    keywords = _extract_keywords(item)
    popularity = _extract_popularity(item)

    # embed each non-empty feature individually
    features = {
        "text": text,
        "genres": genres,
        "actors": actors,
        "directors": directors,
        "keywords": keywords,
        "popularity": popularity,
    }

    inputs = []
    keys = []
    for k, s in features.items():
        keys.append(k)
        inputs.append(s if s else "")

    embs = embed_text(inputs)
    # embs shape (len(inputs), dim) but some inputs may be empty; we'll zero them out
    feature_vecs: Dict[str, Optional[np.ndarray]] = {}
    if embs.size:
        for idx, k in enumerate(keys):
            s = inputs[idx]
            vec = embs[idx] if idx < embs.shape[0] else None
            if not s:
                # treat empty as zero vector
                feature_vecs[k] = np.zeros(embs.shape[1], dtype=np.float32)
            else:
                feature_vecs[k] = _safe_normalize(np.asarray(vec, dtype=np.float32))
    else:
        # no embeddings produced, make zeros
        for k in keys:
            feature_vecs[k] = None

    combined = _combine_features(feature_vecs, w)
    return combined


def embed_item(item: Dict, weights: Optional[Dict[str, float]] = None) -> Dict:
    """Compute weighted embedding for a TMDB item and return it (doesn't persist to Mongo) with embedding model metadata."""
    vec = build_weighted_embedding_for_item(item, weights=weights)
    embed_list = vec.tolist() if vec.size else []
    meta = {
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_ts": int(time.time()),
        "embedding_dims": len(embed_list),
    }
    item_copy = dict(item)
    item_copy["embedding"] = embed_list
    item_copy["embedding_meta"] = meta
    return item_copy


def compute_embeddings_batch(
    docs: List[Dict], weights: Optional[Dict[str, float]] = None
) -> List[np.ndarray]:
    """Compute embeddings for a batch of metadata docs and return list of vectors (no DB writes)."""
    from concurrent.futures import ThreadPoolExecutor

    results: List[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [
            ex.submit(build_weighted_embedding_for_item, d, weights) for d in docs
        ]
        for f in futures:
            try:
                v = f.result()
                results.append(np.asarray(v, dtype=np.float32))
            except Exception as e:
                logger.warning(
                    "Batch embedding computation failed: %s", repr(e), exc_info=True
                )
                results.append(np.zeros(384, dtype=np.float32))
    return results


def _process_batch(batch: List[Dict], weights: Optional[Dict[str, float]] = None):
    """Compute embeddings for a batch and don't persist to DB (legacy persisted behavior removed)."""
    vecs = compute_embeddings_batch(batch, weights=weights)
    # return list of item copies with embeddings attached for callers that expect them
    results = []
    now_ts = int(time.time())
    for idx, it in enumerate(batch):
        vec = vecs[idx] if idx < len(vecs) else np.zeros_like(vecs[0])
        embed_list = vec.tolist() if vec.size else []
        meta = {
            "embedding_model": EMBED_MODEL_NAME,
            "embedding_ts": now_ts,
            "embedding_dims": len(embed_list),
        }
        item_copy = dict(it)
        item_copy["embedding"] = embed_list
        item_copy["embedding_meta"] = meta
        results.append(item_copy)
    return results


def embed_all_items(batch_size: int = 256) -> int:
    """Iterate over all items in tmdb_metadata_collection and compute embeddings for those missing them.
    Returns number of items processed.
    """
    cursor = tmdb_metadata_collection.find({}, {"_id": 0})
    processed = 0
    batch: List[Dict] = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) >= batch_size:
            _process_batch(batch)
            processed += len(batch)
            batch = []
    if batch:
        _process_batch(batch)
        processed += len(batch)
    logger.info("Embedded %s items", processed)
    return processed


def build_user_vector_from_history(
    history_items: List[Dict],
    decay_days: Optional[float] = None,
    min_weight: float = 0.3,
) -> Optional[np.ndarray]:
    """Build a user embedding by weighted average of item embeddings. Weights decay by recency.

    Args:
        history_items: list of TMDB item docs (must include 'id' and optionally 'latest_watched_at' or 'watched_at')
        decay_days: decay factor in days (larger = slower decay). Default from env RECENCY_DECAY_DAYS or 120.0
        min_weight: minimum weight for older items (prevents total decay). Default 0.3

    With decay_days=120 and min_weight=0.3:
    - Items watched today: weight = 1.0
    - Items watched 60 days ago: weight ≈ 0.61
    - Items watched 120 days ago: weight ≈ 0.37
    - Items watched 1+ year ago: weight = 0.3 (floor)
    """
    if decay_days is None:
        decay_days = float(os.getenv("RECENCY_DECAY_DAYS", "120.0"))
    # history_items are watch-history records (no embeddings). Fetch embeddings
    # from FAISS/sidecar using ids in the history.
    if not history_items:
        return None

    now_ts = time.time()
    # collect embeddings and timestamps
    embs = []
    age_days_list = []
    rewatch_engagement_list = []

    # gather ids referenced in history
    ids = [it.get("id") for it in history_items if it.get("id") is not None]
    mts = [it.get("media_type") for it in history_items if it.get("id") is not None]
    if not ids:
        return None

    # lazy import to avoid circular imports
    from app.faiss_index import get_vectors_for_ids

    vecs = get_vectors_for_ids(ids, media_types=mts)

    for idx, v in enumerate(vecs):
        if v is None:
            continue
        embs.append(np.asarray(v, dtype=np.float32))
        # find watched timestamp on history item
        h = history_items[idx]
        ts = None
        for fld in ("latest_watched_at", "watched_at"):
            if h.get(fld):
                try:
                    ts = float(h.get(fld))
                    break
                except Exception:
                    pass
        if ts is None:
            # if no timestamp, treat as older
            age_days_list.append(365.0)
        else:
            age_days_list.append(max(0.0, (now_ts - ts) / 86400.0))
        # engagement / rewatch factor
        rewatch_engagement_list.append(float(h.get("rewatch_count", 1.0) or 1.0))

    if not embs:
        return None

    # compute weights
    weights = []
    for age_d, rewatch in zip(age_days_list, rewatch_engagement_list):
        decay = max(min_weight, np.exp(-age_d / decay_days))
        weights.append(decay * float(rewatch))

    # weighted average then normalize
    mat = np.vstack(embs)
    w = np.asarray(weights, dtype=np.float32)
    w = w.reshape(-1, 1)
    try:
        avg = np.sum(mat * w, axis=0) / (np.sum(w) if np.sum(w) > 0 else 1.0)
        return _safe_normalize(avg)
    except Exception as e:
        logger.error(
            "Failed to build user vector from history: %s", repr(e), exc_info=True
        )
        return None
