"""Recommendation processing module."""

import json

from app.dao.history import get_watch_history
from app.db import tmdb_metadata_collection
from app.embeddings import build_user_vector_from_history
from app.faiss_index import load_faiss_index, query_faiss
from app.schemas.recommendations.recommendations import RecommendationsResponse
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_chat_completion
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)


class MediaRecommender:
    """Processor for generating media (movie/tv) recommendations based on watch history."""

    def __init__(self):
        self.prompt_registry = PromptRegistry("app/prompts/recommend")

    def format_watch_history(self, watch_history: list[dict], media_type: str = "movie"):
        """Format watch history and keep the TMDB id for deduplication."""
        formatted_watch_history = []
        for item in watch_history:
            if item.get("action") == "watch":
                formatted_watch_history.append(
                    {
                        k: item.get(k) for k in [
                            "id",
                            "title",
                            "year",
                            "watch_count",
                            "watched_at",
                            "earliest_watched_at",
                            "latest_watched_at",
                        ]
                    }
                )
        return formatted_watch_history

    def load_watch_history(self, media_type: str = "movie"):
        """Load watch history from the database."""
        try:
            # allow None to mean all media types
            watch_history = get_watch_history(media_type=media_type)
            watch_history = self.format_watch_history(
                watch_history=watch_history, media_type=media_type
            )
            logger.info("Loaded %s watch history %s items.", len(watch_history), media_type)
            return watch_history
        except Exception as e:
            logger.error("Failed to load watch history: %s", repr(e), exc_info=True)
            return []

    def get_recommendation_prompt(
        self,
        watch_history: list[dict],
        media_type: str,
        candidates: list[dict],
        recommend_count: int = 5,
        prompt_version: int = 1,
    ):
        """Generate a prompt for movie recommendations based on watch history."""
        prompt_template = self.prompt_registry.load_prompt_template(
            f"{media_type}_recommender", prompt_version
        )
        return prompt_template.render(
            watch_history=watch_history,
            candidates=candidates,
            recommend_count=recommend_count,
        )

    def generate_recommendations(
        self, media_type: str = "movie", recommend_count: int = 5
    ) -> RecommendationsResponse:
        """Generate media recommendations based on watch history.

        This queries the shared FAISS index (movies+tv) and then filters results
        by media_type (unless media_type == 'all'). Excludes items already
        present in the user's watch history.
        """
        # load history; if media_type == 'all' we want history across all types
        history_media_type = None if media_type == "all" else media_type
        watch_history = self.load_watch_history(media_type=history_media_type)

        # build a set of watched ids to exclude from recommendations
        watched_ids = {item.get("id") for item in watch_history if item.get("id") is not None}

        candidates_filtered = []
        try:
            user_vec = build_user_vector_from_history(watch_history)
            if user_vec is not None:
                index = load_faiss_index()
                if index is not None:
                    # query with a larger k to allow for post-query filtering
                    k = max(200, recommend_count * 40)
                    res = query_faiss(index, user_vec, k=k)

                    # fetch docs for returned ids in bulk
                    ids = [r[0] for r in res]
                    docs = list(
                        tmdb_metadata_collection.find({"id": {"$in": ids}}, {"_id": 0})
                    )
                    docs_by_id = {d.get("id"): d for d in docs}

                    # iterate in FAISS order and filter by media_type + watched ids
                    target_pool = max(recommend_count * 5, 20)
                    for iid, score in res:
                        if iid in watched_ids:
                            continue
                        doc = docs_by_id.get(iid)
                        if not doc:
                            continue
                        if media_type != "all" and doc.get("media_type") != media_type:
                            continue
                        # include score and keep the doc shape
                        doc_copy = dict(doc)
                        doc_copy["_score"] = score
                        candidates_filtered.append(doc_copy)
                        if len(candidates_filtered) >= target_pool:
                            break
        except Exception as e:
            logger.warning(
                "Candidate generation via embeddings/FAISS failed: %s",
                repr(e),
                exc_info=True,
            )
            candidates_filtered = []

        # couldn't find many candidates, log a warning (but continue)
        if len(candidates_filtered) < recommend_count:
            logger.warning(
                "Filtered candidate pool small for media_type=%s (requested=%s, found=%s)",
                media_type,
                recommend_count,
                len(candidates_filtered),
            )

        top_candidates = (
            candidates_filtered[: max(recommend_count * 5, 20)] if candidates_filtered else []
        )
        top_candidates = [
            {"id": c.get("id"), "title": c.get("title"), "score": c.get("_score")}
            for c in top_candidates
        ]

        prompt = self.get_recommendation_prompt(
            watch_history=watch_history,
            media_type=media_type,
            candidates=top_candidates,
            recommend_count=recommend_count,
            prompt_version=1,
        )
        logger.info("Generated %s recommendation prompt: %s ...", media_type, prompt[:2000])
        messages = [{"role": "user", "content": prompt}]
        try:
            response = get_openai_chat_completion(
                "gpt-4.1-nano",
                messages=messages,
                response_format={"type": "json_object"},
            )
            completion_text = response.choices[0].message.content
            recommendations = json.loads(completion_text).get("recommendations", [])
        except Exception as e:
            logger.error(
                "Failed to parse structured OpenAI response: %s", repr(e), exc_info=True
            )
            raise
        return RecommendationsResponse(recommendations=recommendations)
