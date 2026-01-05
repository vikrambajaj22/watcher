"""Generic recommendation schemas for movies and TV."""

from typing import List, Optional

from pydantic import BaseModel, model_validator


class RecommendRequest(BaseModel):
    """Request body for /recommend/{media_type}."""

    recommend_count: int = 5

    @model_validator(mode="after")
    def clamp_recommend_count(self):
        # ensure recommend_count is within reasonable bounds
        self.recommend_count = max(1, min(20, int(self.recommend_count)))
        return self


class Recommendation(BaseModel):
    id: str
    title: str
    reasoning: str
    media_type: Optional[str] = None
    metadata: Optional[dict] = None


class RecommendationsResponse(BaseModel):
    recommendations: List[Recommendation]
