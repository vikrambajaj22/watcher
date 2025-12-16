from typing import List, Optional

from pydantic import BaseModel, model_validator


class MCPPayload(BaseModel):
    """Payload for the /mcp/knn endpoint.
    Provide exactly one of: tmdb_id, text, or vector. k controls number of neighbors.
    """

    tmdb_id: Optional[int] = None
    text: Optional[str] = None
    vector: Optional[List[float]] = None
    k: int = 10
    media_type: Optional[str] = None  # optional: 'movie', 'tv', or 'all'

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
        if self.media_type is None:
            return self
        allowed = {"movie", "tv", "all"}
        if str(self.media_type).lower() not in allowed:
            raise ValueError("media_type must be one of: 'movie', 'tv', 'all'")
        # normalize
        self.media_type = str(self.media_type).lower()
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
