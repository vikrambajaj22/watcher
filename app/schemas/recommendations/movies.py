"""Pydantic schema for movie recommendations."""
from typing import List

from pydantic import BaseModel


class MovieRecommendation(BaseModel):
    title: str
    overview: str
    reasoning: str

class MovieRecommendationsResponse(BaseModel):
    recommendations: List[MovieRecommendation]