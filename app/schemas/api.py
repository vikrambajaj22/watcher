from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator, model_validator


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
    filter_to_history: bool = False

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


class DiscoverItem(BaseModel):
    id: int
    title: Optional[str] = None
    media_type: Optional[str] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    watched: bool = False


class DiscoverFilters(BaseModel):
    media_type: str = "both"
    genres: List[str] = []
    cast: List[str] = []
    keywords: List[str] = []
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    model_config = {"extra": "ignore"}

    @field_validator("media_type", mode="before")
    @classmethod
    def _validate_media_type(cls, v: Any) -> str:
        s = str(v or "both").lower()
        return s if s in ("movie", "tv", "both") else "both"


class DescribeRequest(BaseModel):
    query: str
    limit: int = 20
    media_type: Optional[str] = None

    @model_validator(mode="after")
    def _clamp_limit(self):
        self.limit = max(1, min(self.limit, 40))
        if self.media_type and self.media_type not in ("movie", "tv", "both"):
            self.media_type = None
        return self


class DescribeResponse(BaseModel):
    results: List[DiscoverItem]
    filters: Optional[DiscoverFilters] = None


class PersonSummary(BaseModel):
    id: int
    name: Optional[str] = None
    profile_path: Optional[str] = None
    known_for_department: Optional[str] = None


class ActorHistoryItem(BaseModel):
    id: int
    title: Optional[str] = None
    media_type: Optional[str] = None
    poster_path: Optional[str] = None
    character: Optional[str] = None
    department: Optional[str] = None
    watched_at: Optional[str] = None


class ActorHistoryResponse(BaseModel):
    person: Optional[PersonSummary] = None
    items: List[ActorHistoryItem] = []


class TasteProfile(BaseModel):
    signature: str = ""
    summary: str = ""
    genres: List[str] = []
    themes: List[str] = []
    avoid: List[str] = []
    history_count: int = 0


class ChatRequest(BaseModel):
    thread_id: str
    message: str


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
    year: Optional[int] = None
    genres: Optional[List[str]] = None
    model_config = {"extra": "allow"}


class UpcomingEpisode(BaseModel):
    tmdb_id: int
    show_title: Optional[str] = None
    poster_path: Optional[str] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_title: Optional[str] = None
    first_aired: Optional[str] = None


class UpcomingResponse(BaseModel):
    episodes: List[UpcomingEpisode]


class WatchlistItem(BaseModel):
    tmdb_id: int
    media_type: str
    title: Optional[str] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    genres: Optional[List[str]] = None
    trakt_slug: Optional[str] = None
    synced_at: Optional[str] = None


class AddWatchlistRequest(BaseModel):
    tmdb_id: int
    media_type: str
    title: Optional[str] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self):
        if self.media_type not in ("movie", "tv"):
            raise ValueError("media_type must be 'movie' or 'tv'")
        return self


class WatchlistSyncResponse(BaseModel):
    added: int
    removed: int
