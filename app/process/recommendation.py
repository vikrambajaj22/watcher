"""Recommendation processing module."""

import json
import time

from app.dao.history import get_watch_history
from app.db import tmdb_metadata_collection
from app.embeddings import build_user_vector_from_history
from app.vector_store import query as vector_query
from app.utils.knn_utils import process_knn_results
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
            watch_history = get_watch_history(media_type=media_type, include_posters=False)
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
        # render candidates as a numbered (1-indexed) list for the prompt
        candidates_enumerated = []
        for i, c in enumerate(candidates, start=1):
            candidates_enumerated.append({"index": i, "id": c.get("id"), "title": c.get("title"), "score": c.get("score")})
        return prompt_template.render(
            watch_history=watch_history,
            candidates=candidates,
            candidates_enumerated=candidates_enumerated,
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
        history_media_type = None if media_type == "all" else media_type
        # do not include posters when generating recommendations (faster)
        watch_history = self.load_watch_history(media_type=history_media_type)

        # build a set of watched ids to exclude from recommendations
        watched_ids = {str(item.get("id")) for item in watch_history if item.get("id") is not None}

        def _resolve_title(candidate_id, doc=None):
            # prefer doc-provided title/name fields, else fetch from DB as a last resort
            if doc:
                t = (
                    doc.get("title")
                    or doc.get("name")
                    or doc.get("original_title")
                    or doc.get("original_name")
                )
                if t:
                    return t
            try:
                # prefer exact media_type match if doc available in calling scope
                query = {"id": candidate_id}
                # if caller-provided doc has a media_type, use it to disambiguate
                if doc and doc.get("media_type"):
                    query["media_type"] = doc.get("media_type")
                d = tmdb_metadata_collection.find_one(query, {"_id": 0, "title": 1, "name": 1, "original_title": 1, "original_name": 1})
                if d:
                    return (
                        d.get("title")
                        or d.get("name")
                        or d.get("original_title")
                        or d.get("original_name")
                        or ""
                    )
            except Exception:
                pass
            return ""

        candidates_filtered = []
        try:
            t_start = time.time()
            user_vec = build_user_vector_from_history(watch_history)
            vec_time = time.time() - t_start
            logger.info("User vector build time: %.3fs", vec_time)
            if user_vec is not None:
                # use vector_store.query which loads the FAISS index if necessary
                max_k = 1000
                k = min(max_k, max(200, recommend_count * 20))
                t_q = time.time()
                try:
                    res = vector_query(user_vec, k=k)
                except Exception as e:
                    logger.warning("FAISS/vector query failed: %s", repr(e), exc_info=True)
                    res = []
                q_time = time.time() - t_q
                logger.info("Vector store query time: %.3fs results=%s", q_time, len(res))

                # process knn results via shared helper (prefer exact id+media matches and apply filters)
                requested_media_type = None if media_type == "all" else media_type
                candidates = process_knn_results(
                    vs_res=res,
                    k=max(recommend_count * 5, 20),
                    exclude_id=None,
                    requested_media_type=requested_media_type,
                    watched_ids=watched_ids,
                )

                # normalize candidate docs into the expected shape (score stored as _score)
                for c in candidates:
                    doc_copy = dict(c)
                    doc_copy["_score"] = doc_copy.pop("score", None)
                    candidates_filtered.append(doc_copy)
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
        # ensure titles are resolved for any candidates (in case some lacked title in-memory)
        top_candidates = [
            {
                "id": c.get("id"),
                # prefer candidate-provided title, otherwise resolve using the candidate doc (media_type-aware)
                "title": (c.get("title") or _resolve_title(c.get("id"), doc=c)),
                "score": c.get("_score"),
                "media_type": c.get("media_type"),
                "poster_path": c.get("poster_path"),
                "overview": c.get("overview"),
            }
            for c in top_candidates
        ]
        logger.info("Generated %s candidates for recommendation: %s", len(top_candidates), top_candidates)

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

        logger.info("Generated %s recommendations: %s", len(recommendations), recommendations)
        # validate LLM-returned ids against the candidate docs fetched earlier
        valid_recs = []
        unknown_ids = []

        def _build_rec_obj(rid, title, reasoning=None, metadata=None, media_type=None, poster_path=None, overview=None):
            # Build metadata dict with poster_path and overview if not already provided
            if metadata is None:
                metadata = {}
            if poster_path is not None:
                metadata["poster_path"] = poster_path
            if overview is not None:
                metadata["overview"] = overview

            return {
                "id": str(rid) if rid is not None else "",
                "title": title or "",
                "reasoning": reasoning or "",
                "media_type": media_type or None,
                "metadata": metadata or None,
            }

        # LLM returns indices (1-indexed) into candidates_enumerated - map them back to top_candidates
        # build a mapping index -> candidate
        index_to_candidate = {i + 1: c for i, c in enumerate(top_candidates)}
        for r in recommendations:
            idx = r.get("index")
            try:
                idx_int = int(idx)
            except Exception:
                unknown_ids.append(idx)
                continue
            cand = index_to_candidate.get(idx_int)
            if not cand:
                unknown_ids.append(idx_int)
                continue
            # accept LLM-provided reasoning and metadata fields if present
            reasoning = r.get("reasoning") if isinstance(r.get("reasoning"), str) else None
            metadata = r.get("metadata") if isinstance(r.get("metadata"), dict) else None
            valid_recs.append(_build_rec_obj(
                cand.get("id"),
                cand.get("title"),
                reasoning=reasoning,
                metadata=metadata,
                media_type=cand.get("media_type"),
                poster_path=cand.get("poster_path"),
                overview=cand.get("overview")
            ))

        if unknown_ids:
            logger.warning("LLM returned unknown recommendation ids (hallucination?): %s", unknown_ids)

        # if LLM didn't provide enough valid recommendations, fill from top_candidates (deterministic fallback)
        already = {r["id"] for r in valid_recs}
        for c in top_candidates:
            cid = c.get("id")
            cid_str = str(cid)
            if cid_str in already:
                continue
            score = c.get("score")
            reasoning = f"Fallback recommendation (score={score}) based on candidate ranking."
            valid_recs.append(_build_rec_obj(
                cid,
                c.get("title"),
                reasoning=reasoning,
                media_type=c.get("media_type"),
                poster_path=c.get("poster_path"),
                overview=c.get("overview")
            ))
            if len(valid_recs) >= recommend_count:
                break

        final_recs = valid_recs[:recommend_count]
        return RecommendationsResponse(recommendations=final_recs)
