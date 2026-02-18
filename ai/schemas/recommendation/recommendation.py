from typing import List
from pydantic import BaseModel, Field


class IndexEmbeddingRequest(BaseModel):
    postId: int
    situation: str
    choice: str


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


class RecommendationItem(BaseModel):
    postId: int
    finalScore: float


class RecommendationResponse(BaseModel):
    query: str
    results: List[RecommendationItem]
