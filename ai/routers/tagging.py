from fastapi import APIRouter, HTTPException
from ai.schemas.tagging import TagRequest, TagResponse
from ai.services.tagging import TagService

router = APIRouter()


@router.post("/ai/tagging", response_model=TagResponse)
def tag_post(req: TagRequest):
    try:
        return TagService.tag_classification(req)
    except (RuntimeError, OSError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"태그 분류 모델을 일시적으로 사용할 수 없습니다: {type(e).__name__}",
        )