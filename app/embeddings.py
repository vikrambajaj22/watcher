from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Any

import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.db import tmdb_metadata_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)

# retrieval-optimized bi-encoder recommended for semantic search/recommendation
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "multi-qa-mpnet-base-dot-v1")
EMBED_DEVICE = os.getenv("EMBED_DEVICE")  # can be cuda, mps or cpu

_model: Optional[Any] = None


def _select_device() -> str:
    if EMBED_DEVICE:
        return EMBED_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    # pyTorch mps backend for Apple silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model() -> Any:
    global _model
    if _model is None:
        device = _select_device()
        logger.info("Loading sentence-transformers model: %s on device %s", EMBED_MODEL_NAME, device)
        _model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return _model


def update_genre_mapping_file(media_type: str="movie"):
    """Update genre mapping file."""
    assert media_type in {"movie", "tv"}, f"media_type {media_type} not supported for genre retrieval."
    genre_mapping_file = os.path.join("static", f"{media_type}_genres.json")
    url = f"{settings.TMDB_API_URL}/genre/{media_type}/list"
    params = {"api_key": os.getenv("TMDB_API_KEY")}
    response = requests.get(url, params=params)
    response.raise_for_status()

    genres = response.json().get("genres", [])
    with open(genre_mapping_file, "w") as f:
        json.dump({g["id"]: g["name"] for g in genres}, f, indent=4)


def get_or_load_genre_mapping(media_type: str = "movie") -> Dict[int, str]:
    """Get or load genre mapping from TMDB."""
    assert media_type in {"movie", "tv"}, f"media_type {media_type} not supported for genre retrieval."
    genre_mapping_file = os.path.join("static", f"{media_type}_genres.json")
    if not os.path.exists(genre_mapping_file):
        logger.info("Genre mapping file %s does not exist, creating it.", genre_mapping_file)
        update_genre_mapping_file(media_type)
    with open(genre_mapping_file, "r") as f:
        return json.load(f)


def get_genres(genre_ids: List[int], media_type: str):
    """Get genres if available."""
    genre_mapping = get_or_load_genre_mapping(media_type)
    if not all(i in genre_mapping for i in genre_ids):
        # the ids might have been updated, repopulate the genre mapping file
        update_genre_mapping_file(media_type)
    genre_mapping = get_or_load_genre_mapping(media_type)
    return [genre_mapping.get(g) for g in genre_ids if genre_mapping.get(g)]


def build_text_for_item(item: Dict) -> str:
    """Construct canonical text input for embedding from a TMDB item document."""
    parts: List[str] = []
    title = item.get("title") or ""
    year = None
    # try multiple fields for release year (different for movie and tv shows)
    for date_field in ("release_date", "first_air_date",):
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
        # keep a reasonable limit to avoid huge inputs
        parts.append(f"Overview: {overview[:2000]}")

    genres = get_genres(item.get("genre_ids", []), item.get("media_type"))
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")

    cast = item.get("credits", {}).get("cast") if item.get("credits") else item.get("cast")
    if isinstance(cast, list):
        top_cast = ", ".join([c.get("name") if isinstance(c, dict) else str(c) for c in cast][:6])
        if top_cast:
            parts.append(f"Cast: {top_cast}")

    keywords = item.get("keywords") or item.get("keywords_list")
    if isinstance(keywords, list):
        keyword_list = list(map(str, keywords))[:12]
        parts.append(f"Keywords: {', '.join(keyword_list)}")

    popularity = item.get("popularity")
    if popularity is not None:
        parts.append(f"Popularity: {popularity}")

    return ". ".join(parts)


def embed_text(texts: List[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return np.array(vectors)


def embed_item_and_store(item: Dict) -> Dict:
    """Compute embedding for a TMDB item dict and store it in mongo under `embedding` and `embedding_meta`."""
    text = build_text_for_item(item)
    vec = embed_text([text])[0]
    embed_list = vec.tolist()
    meta = {
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_ts": int(time.time()),
        "embedding_dims": len(embed_list)
    }
    # update document in mongo
    tmdb_metadata_collection.update_one(
        {"id": item.get("id"), "media_type": item.get("media_type")},
        {"$set": {"embedding": embed_list, "embedding_meta": meta}},
        upsert=False
    )
    item_copy = dict(item)
    item_copy["embedding"] = embed_list
    item_copy["embedding_meta"] = meta
    return item_copy


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


def _process_batch(batch: List[Dict]):
    texts = [build_text_for_item(it) for it in batch]
    vecs = embed_text(texts)
    now_ts = int(time.time())
    for it, vec in zip(batch, vecs):
        embed_list = vec.tolist()
        meta = {"embedding_model": EMBED_MODEL_NAME, "embedding_ts": now_ts, "embedding_dims": len(embed_list)}
        tmdb_metadata_collection.update_one(
            {"id": it.get("id"), "media_type": it.get("media_type")},
            {"$set": {"embedding": embed_list, "embedding_meta": meta}},
            upsert=False
        )


def build_user_vector_from_history(history_items: List[Dict], decay_days: float = 30.0) -> Optional[np.ndarray]:
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
                age_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
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
