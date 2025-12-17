from typing import List, Optional

from pydantic import BaseModel, model_validator


class KNNRequest(BaseModel):
    """Payload for the /mcp/knn endpoint.
    Provide exactly one of: tmdb_id, title, or text. k controls number of neighbors.

    Contract:
      - input_media_type: optional, 'movie'|'tv' — required when providing tmdb_id or title.
      - results_media_type: required, 'movie'|'tv'|'all' — filter applied to returned neighbors.
    """

    tmdb_id: Optional[int] = None
    title: Optional[str] = None
    text: Optional[str] = None
    k: int = 10
    input_media_type: Optional[str] = None  # required when tmdb_id or title is present
    results_media_type: str = "all"  # required: 'movie', 'tv', or 'all'

    @model_validator(mode="after")
    def check_one_of(self):
        tmdb_id, title, text = self.tmdb_id, self.title, self.text
        count = sum(1 for v in (tmdb_id, title, text) if v is not None)
        if count == 0:
            raise ValueError("one of tmdb_id, title, or text must be provided")
        if count > 1:
            raise ValueError("provide exactly one of tmdb_id, title, or text")
        return self

    @model_validator(mode="after")
    def validate_media_type(self):
        allowed_results = {"movie", "tv", "all"}
        allowed_input = {"movie", "tv"}

        # normalize and validate results_media_type
        if not self.results_media_type:
            raise ValueError("results_media_type is required and must be one of: 'movie', 'tv', 'all'")
        if str(self.results_media_type).lower() not in allowed_results:
            raise ValueError("results_media_type must be one of: 'movie', 'tv', 'all'")
        self.results_media_type = str(self.results_media_type).lower()

        # validate input_media_type when present
        if self.input_media_type is not None:
            if str(self.input_media_type).lower() not in allowed_input:
                raise ValueError("input_media_type must be one of: 'movie', 'tv'")
            self.input_media_type = str(self.input_media_type).lower()

        # when a tmdb_id or title is provided, require input_media_type to disambiguate movie vs tv
        if (self.tmdb_id is not None or self.title) and not self.input_media_type:
            raise ValueError("input_media_type is required when providing a tmdb_id or title; specify 'movie' or 'tv'")

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
    poster_path: Optional[str] = None


class WillLikeResponse(BaseModel):
    """Response from /mcp/will-like endpoint."""
    will_like: bool
    score: float
    explanation: str
    item: ItemSummary
