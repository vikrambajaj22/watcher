from typing import List, Optional

from pydantic import BaseModel, model_validator


class SimilarRequest(BaseModel):
    """Payload for POST /similar.
    Provide exactly one of: tmdb_id or title, along with media_type.
    Set cross_type=True to discover titles of the opposite type: LLM generates thematic
    keywords from the source overview, resolved to TMDB keyword IDs and RRF-merged.
    """

    tmdb_id: Optional[int] = None
    title: Optional[str] = None
    media_type: str  # 'movie' or 'tv'
    k: int = 20
    cross_type: bool = False

    @model_validator(mode="after")
    def check_inputs(self):
        has_id = self.tmdb_id is not None
        has_title = bool(self.title)
        if not has_id and not has_title:
            raise ValueError("one of tmdb_id or title must be provided")
        if has_id and has_title:
            raise ValueError("provide exactly one of tmdb_id or title, not both")
        if str(self.media_type).lower() not in {"movie", "tv"}:
            raise ValueError("media_type must be 'movie' or 'tv'")
        self.media_type = str(self.media_type).lower()
        return self


class SimilarResultItem(BaseModel):
    id: int
    title: Optional[str] = None
    media_type: Optional[str] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None


class SimilarResponse(BaseModel):
    source_title: Optional[str] = None
    results: List[SimilarResultItem]


class WillLikeRequest(BaseModel):
    """Payload for POST /will-like.
    Provide exactly one of: tmdb_id or title, along with media_type.
    """

    tmdb_id: Optional[int] = None
    title: Optional[str] = None
    media_type: str

    @model_validator(mode="after")
    def check_one_of(self):
        has_id = self.tmdb_id is not None
        has_title = bool(self.title)
        if not has_id and not has_title:
            raise ValueError("one of tmdb_id or title must be provided")
        if has_id and has_title:
            raise ValueError("provide exactly one of tmdb_id or title, not both")
        if str(self.media_type).lower() not in {"movie", "tv"}:
            raise ValueError("media_type must be 'movie' or 'tv'")
        self.media_type = str(self.media_type).lower()
        return self


class ItemSummary(BaseModel):
    id: Optional[int] = None
    title: Optional[str] = None
    media_type: Optional[str] = None
    overview: Optional[str] = None
    poster_path: Optional[str] = None


class WillLikeResponse(BaseModel):
    will_like: bool
    score: float
    explanation: str
    already_watched: bool = False
    item: ItemSummary


class AdminCancelJobRequest(BaseModel):
    job_id: str


class AdminAckResponse(BaseModel):
    status: str
    message: Optional[str] = None


class AdminJobAcceptedResponse(BaseModel):
    status: str
    job_id: str


class SyncStatusResponse(BaseModel):
    trakt_last_activity: Optional[str] = None
    tmdb_movie_last_sync: Optional[str] = None
    tmdb_tv_last_sync: Optional[str] = None


class JobStatusModel(BaseModel):
    status: Optional[str] = None
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    processed: Optional[int] = None
    embed_queued: Optional[int] = None
    last_update: Optional[int] = None
    result: Optional[dict] = None
    summary: Optional[dict] = None
    embed_summary: Optional[dict] = None
    error: Optional[str] = None
    trace: Optional[str] = None
    cancel: Optional[bool] = None
    cancel_requested_at: Optional[int] = None


class HistoryItem(BaseModel):
    id: Optional[int] = None
    tmdb_id: Optional[int] = None
    title: Optional[str] = None
    name: Optional[str] = None
    media_type: Optional[str] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    latest_watched_at: Optional[str] = None
    earliest_watched_at: Optional[str] = None
    watch_count: Optional[int] = None
    completion_ratio: Optional[float] = None
    rewatch_engagement: Optional[float] = None
    model_config = {"extra": "allow"}
