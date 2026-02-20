from typing import List
from pydantic import BaseModel


class TagRequest(BaseModel):
    postId: int
    situation: str
    action: str
    reflection: str


class TagResult(BaseModel):
    tagId: int
    tagName: str
    confidence: float


class TagResponse(BaseModel):
    postId: int
    modelVersion: str
    threshold: float
    tags: List[TagResult]