from fastapi import APIRouter
from ai.schemas.tagging import TagRequest, TagResponse
from ai.services.tagging import TagService

router = APIRouter()


@router.post("/ai/tagging", response_model=TagResponse)
def tag_post(req: TagRequest):
    return TagService.tag_classification(req)