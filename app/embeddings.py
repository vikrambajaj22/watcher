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
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "multi-qa-mpnet-base-dot-v1")
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
        sorted_cast = sorted((c for c in cast_list if isinstance(c, dict)), key=lambda c: c.get("order", 9999))
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


def _combine_features(feature_vecs: Dict[str, Optional[np.ndarray]], weights: Dict[str, float]) -> np.ndarray:
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
            dim = 768
    combined = np.zeros(dim, dtype=np.float32)
    for name, vec in feature_vecs.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0 or vec is None or not vec.size:
            continue
        combined += w * vec
    return _safe_normalize(combined)


def build_weighted_embedding_for_item(item: Dict, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
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


def embed_item_and_store(item: Dict, weights: Optional[Dict[str, float]] = None) -> Dict:
    """Compute weighted embedding for a TMDB item and store in Mongo."""
    vec = build_weighted_embedding_for_item(item, weights=weights)
    embed_list = vec.tolist() if vec.size else []
    meta = {
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_ts": int(time.time()),
        "embedding_dims": len(embed_list),
    }
    tmdb_metadata_collection.update_one(
        {"id": item.get("id"), "media_type": item.get("media_type")},
        {"$set": {"embedding": embed_list, "embedding_meta": meta}},
        upsert=False,
    )
    item_copy = dict(item)
    item_copy["embedding"] = embed_list
    item_copy["embedding_meta"] = meta
    return item_copy


def _process_batch(batch: List[Dict], weights: Optional[Dict[str, float]] = None):
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    texts = [_extract_text(it) for it in batch]
    genres = [_extract_genres(it) for it in batch]
    actors = [_extract_actors(it) for it in batch]
    directors = [_extract_directors(it) for it in batch]
    keywords = [_extract_keywords(it) for it in batch]
    pops = [_extract_popularity(it) for it in batch]

    emb_text = embed_text(texts)
    emb_genres = embed_text(genres)
    emb_actors = embed_text(actors)
    emb_directors = embed_text(directors)
    emb_keywords = embed_text(keywords)
    emb_pops = embed_text(pops)

    dim = None
    for arr in (emb_text, emb_genres, emb_actors, emb_directors, emb_keywords, emb_pops):
        if arr.size:
            dim = arr.shape[1]
            break
    if dim is None:
        try:
            dim = _get_model().encode(["dummy"]).shape[-1]
        except Exception:
            dim = 768

    now_ts = int(time.time())

    for idx, it in enumerate(batch):
        feature_vecs: Dict[str, Optional[np.ndarray]] = {}
        # helper to pick vec or zeros
        def pick(arr):
            if arr is None or not getattr(arr, "size", 0):
                return np.zeros(dim, dtype=np.float32)
            if idx >= arr.shape[0]:
                return np.zeros(dim, dtype=np.float32)
            s = arr[idx]
            # if original string was empty, treat as zero
            return _safe_normalize(np.asarray(s, dtype=np.float32)) if np.linalg.norm(s) > 0 else np.zeros(dim, dtype=np.float32)

        feature_vecs["text"] = pick(emb_text) if emb_text.size else np.zeros(dim, dtype=np.float32)
        feature_vecs["genres"] = pick(emb_genres) if emb_genres.size else np.zeros(dim, dtype=np.float32)
        feature_vecs["actors"] = pick(emb_actors) if emb_actors.size else np.zeros(dim, dtype=np.float32)
        feature_vecs["directors"] = pick(emb_directors) if emb_directors.size else np.zeros(dim, dtype=np.float32)
        feature_vecs["keywords"] = pick(emb_keywords) if emb_keywords.size else np.zeros(dim, dtype=np.float32)
        feature_vecs["popularity"] = pick(emb_pops) if emb_pops.size else np.zeros(dim, dtype=np.float32)

        combined = _combine_features(feature_vecs, w)

        embed_list = combined.tolist() if combined.size else []
        meta = {
            "embedding_model": EMBED_MODEL_NAME,
            "embedding_ts": now_ts,
            "embedding_dims": len(embed_list),
        }
        try:
            tmdb_metadata_collection.update_one(
                {"id": it.get("id"), "media_type": it.get("media_type")},
                {"$set": {"embedding": embed_list, "embedding_meta": meta}},
                upsert=False,
            )
        except Exception as e:
            logger.warning("Failed to update embedding for %s: %s", it.get("id"), repr(e), exc_info=True)


def index_all_items(batch_size: int = 256) -> int:
    """Iterate over all items in tmdb_metadata_collection and compute embeddings for those missing them.
    Returns number of items processed."""
    cursor = tmdb_metadata_collection.find({}, {"_id": 0})
    processed = 0
    batch: List[Dict] = []
    for doc in cursor:
        if not doc.get("embedding"):
            batch.append(doc)
        if len(batch) >= batch_size:
            _process_batch(batch)
            processed += len(batch)
            batch = []
    if batch:
        _process_batch(batch)
        processed += len(batch)
    logger.info("Indexed %s items with embeddings", processed)
    return processed


def build_user_vector_from_history(
    history_items: List[Dict], decay_days: float = 30.0
) -> Optional[np.ndarray]:
    """Build a user embedding by weighted average of item embeddings. Weights decay by recency.
    history_items: list of TMDB item docs (must include 'embedding' and 'latest_watched_at' or 'watched_at')
    decay_days: decay factor in days (larger = slower decay) (default: 30.0)
    """

    if not history_items:
        return None

    now_ts = time.time()
    # collect embeddings and timestamps
    embs = []
    age_days_list = []

    for it in history_items:
        emb = it.get("embedding")
        if emb is None:
            continue
        embs.append(np.asarray(emb, dtype=np.float32))
        ts_str = it.get("latest_watched_at") or it.get("watched_at")
        age_days = 0.0
        if ts_str:
            try:
                age_ts = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                ).timestamp()
                age_days = (now_ts - age_ts) / 86400.0
            except Exception:
                age_days = 0.0
        age_days_list.append(age_days)

    if not embs:
        return None

    embs_arr = np.stack(embs)  # shape: (n_items, embedding_dim)
    ages_arr = np.array(age_days_list, dtype=np.float32)

    # compute exponential decay weights
    weights = np.exp(-ages_arr / max(1.0, decay_days))  # shape: (n_items,)

    # apply weights and sum
    weighted_embs = embs_arr * weights[:, None]
    user_vec = np.sum(weighted_embs, axis=0)

    norm = np.linalg.norm(user_vec)
    if norm < 1e-12:
        return None

    return user_vec / norm
