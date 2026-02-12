from pydantic import BaseModel
from typing import List


class TagRequest(BaseModel):
    postId: int
    situation: str
    choice: str
    outcome: str
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