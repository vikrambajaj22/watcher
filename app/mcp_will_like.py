from typing import Optional, Dict, Any

import numpy as np

from app.db import tmdb_metadata_collection
from app.faiss_index import get_vectors_for_ids
from app.tmdb_client import get_metadata, search_by_title
from app.embeddings import embed_item, build_user_vector_from_history
from app.dao.history import get_watch_history
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WillLikeError(Exception):
    pass


def compute_will_like(
    tmdb_id: Optional[int], title: Optional[str], media_type: str
) -> Dict[str, Any]:
    """Resolve an item (by id or title), get its embedding from FAISS, build user vector,
    compute cosine similarity, and return a result dict.

    Raises WillLikeError on recoverable failures (not found / insufficient history).
    """
    if media_type not in {"movie", "tv"}:
        raise WillLikeError("media_type must be 'movie' or 'tv'")

    resolved_doc = None
    resolved_id = None
    resolved_title = None
    resolved_overview = None
    resolved_poster = None

    # resolve by tmdb_id first
    if tmdb_id is not None:
        resolved_id = int(tmdb_id)
        # get metadata from Mongo (no embeddings)
        resolved_doc = tmdb_metadata_collection.find_one(
            {"id": resolved_id, "media_type": media_type},
            {
                "_id": 0,
                "title": 1,
                "overview": 1,
                "poster_path": 1,
                "backdrop_path": 1,
                "name": 1,
                "original_title": 1,
                "original_name": 1,
            },
        )
        # fallback to TMDB fetch if not in Mongo
        if not resolved_doc:
            try:
                md = get_metadata(resolved_id, media_type=media_type)
            except Exception:
                md = None
            if md:
                md["media_type"] = media_type
                resolved_doc = tmdb_metadata_collection.update_one(
                    {"id": md["id"], "media_type": media_type},
                    {"$set": md},
                    upsert=True,
                )
                resolved_doc = md
                resolved_id = md.get("id")

    # resolve by title if not already found
    if not resolved_doc and title:
        try:
            regex = {"$regex": title, "$options": "i"}
            docs = list(
                tmdb_metadata_collection.find(
                    {
                        "media_type": media_type,
                        "$or": [{"title": regex}, {"name": regex}],
                    },
                    {
                        "_id": 0,
                        "id": 1,
                        "title": 1,
                        "overview": 1,
                        "poster_path": 1,
                        "backdrop_path": 1,
                        "name": 1,
                        "original_title": 1,
                        "original_name": 1,
                    },
                ).limit(1)
            )
            if docs:
                resolved_doc = docs[0]
                resolved_id = resolved_doc.get("id")
        except Exception:
            resolved_doc = None
            resolved_id = None

        if not resolved_doc:
            try:
                md = search_by_title(title, media_type=media_type)
            except Exception:
                md = None
            if md:
                md["media_type"] = media_type
                tmdb_metadata_collection.update_one(
                    {"id": md["id"], "media_type": media_type}, {"$set": md}, upsert=True
                )
                resolved_doc = md
                resolved_id = md.get("id")

    if not resolved_doc or not resolved_id:
        raise WillLikeError("item not found by id or title")

    # ---- get embedding from FAISS ----
    item_emb = get_vectors_for_ids([resolved_id], [media_type])[0]
    if item_emb is None:
        # fallback: compute embedding if not in FAISS
        try:
            md = get_metadata(resolved_id, media_type=media_type)
        except Exception:
            md = None
        if not md:
            raise WillLikeError("failed to fetch metadata from TMDB for embedding")
        md["media_type"] = media_type
        item_with_emb = embed_item(md)
        item_emb = item_with_emb.get("embedding")

    resolved_title = (
        resolved_doc.get("title")
        or resolved_doc.get("name")
        or resolved_doc.get("original_title")
        or resolved_doc.get("original_name")
    )
    resolved_overview = resolved_doc.get("overview")
    resolved_poster = resolved_doc.get("poster_path") or resolved_doc.get("backdrop_path")

    # build user vector from watch history
    history = get_watch_history(media_type=None, include_posters=False)

    already_watched = False
    if resolved_id and history:
        already_watched = any(
            int(h.get("id")) == int(resolved_id) and h.get("media_type") == media_type
            for h in history
        )

    if already_watched:
        return {
            "will_like": False,
            "score": 1.0,
            "explanation": "You have already watched this item.",
            "already_watched": True,
            "item": {
                "id": int(resolved_id),
                "title": resolved_title,
                "media_type": media_type,
                "overview": resolved_overview,
                "poster_path": resolved_poster,
            },
        }

    user_vec = build_user_vector_from_history(history)
    if user_vec is None:
        raise WillLikeError("not enough history embeddings to build user profile")

    # compute cosine similarity
    try:
        item_vec = np.array(item_emb, dtype=np.float32)

        def _norm(v):
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        u = _norm(user_vec)
        v = _norm(item_vec)
        score = float(np.dot(u, v))
    except Exception as e:
        logger.error("Similarity computation failed: %s", repr(e), exc_info=True)
        raise WillLikeError("similarity computation failed")

    will_like = bool(score >= 0.65)
    formatted_score = "{:.3f}".format(float(score))

    explanation = f"Similarity score based on your watch history: {formatted_score}."
    if will_like:
        explanation += " This item closely matches your tastes."
    else:
        explanation += " This item is not closely aligned with your watch history."

    return {
        "will_like": will_like,
        "score": score,
        "explanation": explanation,
        "already_watched": False,
        "item": {
            "id": int(resolved_id),
            "title": resolved_title,
            "media_type": media_type,
            "overview": resolved_overview,
            "poster_path": resolved_poster,
        },
    }
