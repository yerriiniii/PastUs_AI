from fastapi import APIRouter, HTTPException

from ai.schemas.recommendation.recommendation import (
    IndexEmbeddingRequest,
    IndexEmbeddingResponse,
    IndexTagsRequest,
    IndexTagsResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from ai.services.recommendation.recommendation import RecommendationService

router = APIRouter(prefix="/ai")


# 게시글 본문 임베딩 저장 API
@router.post("/index-embedding", response_model=IndexEmbeddingResponse)
def index_embedding(req: IndexEmbeddingRequest):
    try:
        return RecommendationService.index_embeddings(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RuntimeError, OSError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"임베딩 인덱싱에 필요한 모델 사용 불가능: {type(e).__name__}",
        )


# 게시글 태그 저장 API
@router.post("/index-tags", response_model=IndexTagsResponse)
def index_tags(req: IndexTagsRequest):
    try:
        return RecommendationService.index_tags(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (RuntimeError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"태그 인덱싱에 필요한 모델 사용 불가능: {type(e).__name__}")


# 유사 사례 추천 API
@router.post("/recommendation", response_model=RecommendationResponse)
def recommend_posts(req: RecommendationRequest):
    try:
        return RecommendationService.recommend(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (RuntimeError, OSError) as e:
        raise HTTPException(
            status_code=503, detail=f"추천 처리에 필요한 모델 사용 불가능: {type(e).__name__}")
