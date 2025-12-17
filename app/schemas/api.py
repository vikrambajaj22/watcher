from typing import List, Optional

from pydantic import BaseModel, model_validator


class KNNRequest(BaseModel):
    """Payload for the /mcp/knn endpoint.
    Provide exactly one of: tmdb_id, text, or vector. k controls number of neighbors.
    """

    tmdb_id: Optional[int] = None
    text: Optional[str] = None
    vector: Optional[List[float]] = None
    k: int = 10
    media_type: str  # required: 'movie', 'tv', or 'all'

    @model_validator(mode="after")
    def check_one_of(self):
        tmdb_id, text, vector = self.tmdb_id, self.text, self.vector
        count = sum(1 for v in (tmdb_id, text, vector) if v is not None)
        if count == 0:
            raise ValueError("one of tmdb_id, text, or vector must be provided")
        if count > 1:
            raise ValueError("provide exactly one of tmdb_id, text, or vector")
        return self

    @model_validator(mode="after")
    def validate_media_type(self):
        # media_type is required and must be one of allowed values
        allowed = {"movie", "tv", "all"}
        if not self.media_type:
            raise ValueError("media_type is required and must be one of: 'movie', 'tv', 'all'")
        if str(self.media_type).lower() not in allowed:
            raise ValueError("media_type must be one of: 'movie', 'tv', 'all'")
        # normalize
        self.media_type = str(self.media_type).lower()
        # special-case: when a tmdb_id is provided, media_type must be specific (movie|tv), not 'all'
        if self.tmdb_id is not None and self.media_type == 'all':
            raise ValueError("media_type cannot be 'all' when tmdb_id is provided; specify 'movie' or 'tv'")
        return self


class AdminReindexPayload(BaseModel):
    """Payload for /admin/reindex to trigger indexing operations.
    Fields are optional and interpreted as commands.
    """

    id: Optional[int] = None
    media_type: Optional[str] = "movie"
    full: Optional[bool] = False
    batch_size: Optional[int] = 256
    build_faiss: Optional[bool] = False
    dim: Optional[int] = 384
    factory: Optional[str] = "IDMap,IVF100,Flat"


class KNNResultItem(BaseModel):
    id: int
    title: Optional[str] = None
    media_type: Optional[str] = None
    score: Optional[float] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None


class KNNResponse(BaseModel):
    """Response from /mcp/knn endpoint."""
    results: List[KNNResultItem]


class WillLikeRequest(BaseModel):
    """Payload for the /mcp/will-like endpoint.
    Provide exactly one of: tmdb_id or title, along with media_type.
    """
    tmdb_id: Optional[int] = None
    title: Optional[str] = None
    media_type: str

    @model_validator(mode="after")
    def check_one_of(self):
        # require exactly one of tmdb_id or title (not both)
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


class WillLikeResponse(BaseModel):
    """Response from /mcp/will-like endpoint."""
    will_like: bool
    score: float
    explanation: str
    item: ItemSummary
