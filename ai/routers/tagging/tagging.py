from fastapi import APIRouter, HTTPException

from ai.schemas.tagging.tagging import TagRequest, TagResponse
from ai.services.tagging.tagging import TagService

router = APIRouter(prefix="/ai")


# 태그 분류 API
@router.post("/tagging", response_model=TagResponse)
def tag_post(req: TagRequest):
    try:
        return TagService.classify_tags(req)
    except (RuntimeError, OSError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"태그 분류 모델을 일시적으로 사용할 수 없습니다: {type(e).__name__}",
        )