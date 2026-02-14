from typing import List
from pydantic import BaseModel, Field


class IndexEmbeddingRequest(BaseModel):
    postId: int
    situation: str
    choice: str
    outcome: str
    reflection: str


class IndexEmbeddingResponse(BaseModel):
    postId: int
    stored: bool


class IndexTagsRequest(BaseModel):
    postId: int
    tagIds: List[int] = Field(default_factory=list)
    tagNames: List[str] = Field(default_factory=list)


class IndexTagsResponse(BaseModel):
    postId: int
    tagIds: List[int]
    tagNames: List[str]
    stored: bool


class RecommendationRequest(BaseModel):
    query: str
    tagWeight: float = Field(default=0.15, ge=0.0, le=1.0)
    minEmbeddingScore: float = Field(default=0.55, ge=0.0, le=1.0)
    minFinalScore: float = Field(default=0.62, ge=0.0, le=2.0)


class RecommendationItem(BaseModel):
    postId: int
    finalScore: float


class RecommendationResponse(BaseModel):
    query: str
    results: List[RecommendationItem]
