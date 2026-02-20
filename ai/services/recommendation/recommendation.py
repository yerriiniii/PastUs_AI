import os

from ai.infra.upsert_db_provider import get_upsert_db
from ai.services.recommendation.embedding import encode_text
from ai.services.recommendation.final_rank import final_rank
from ai.services.recommendation.first_rank import first_rank
from ai.services.recommendation.mapper import map_to_RecommendationItem
from ai.services.recommendation.query import build_embeddings
from ai.schemas.recommendation.recommendation import (
    IndexEmbeddingRequest,
    IndexEmbeddingResponse,
    IndexTagsRequest,
    IndexTagsResponse,
    RecommendationRequest,
    RecommendationResponse,
)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name}는 float 값이어야 합니다: {raw}") from e


# 파라미터 (.env 우선, 미설정 시 기본값 사용)
TAG_SCORE_WEIGHT = _get_env_float("RECOMM_TAG_SCORE_WEIGHT", 0.15)  # 태그 유사도 가중치
MIN_EMBEDDING_SCORE = _get_env_float("RECOMM_MIN_EMBEDDING_SCORE", 0.4)  # 임베딩 유사도 최소 점수
MIN_FINAL_SCORE = _get_env_float("RECOMM_MIN_FINAL_SCORE", 0.4)  # 최종 유사도 최소 점수


class RecommendationService:

    # 임베딩 DB upsert 실행
    @staticmethod
    def index_embeddings(req: IndexEmbeddingRequest) -> IndexEmbeddingResponse:

        # 게시글에서 상황/결정 텍스트 추출
        situation_text = req.situation.strip()
        decision_text = req.choice.strip()

        # 추출한 텍스트를 각각 임베딩으로 변환해 저장
        get_upsert_db().upsert_post_embeddings(
            post_id=req.postId,
            situation_embedding=encode_text(situation_text),
            decision_embedding=encode_text(decision_text),
        )

        # 저장 결과 반환
        return IndexEmbeddingResponse(postId=req.postId, stored=True)


    # 태그 DB upsert 실행
    @staticmethod
    def index_tags(req: IndexTagsRequest) -> IndexTagsResponse:

        # 태그 ID/이름 모두 필요
        if not req.tagIds or not req.tagNames:
            raise ValueError("tagIds와 tagNames 모두 필요합니다.")

        # 게시글 태그 저장
        get_upsert_db().upsert_post_tags(
            post_id=req.postId,
            tag_ids=req.tagIds,
            tag_names=req.tagNames,
        )

        # 저장 결과 반환
        return IndexTagsResponse(
            postId=req.postId,
            tagIds=req.tagIds,
            tagNames=req.tagNames,
            stored=True,
        )


    # 최종 추천 실행
    @staticmethod
    def recommend(req: RecommendationRequest) -> RecommendationResponse:

        # 검색어가 비어 있으면 예외
        if not req.query.strip():
            raise ValueError("query는 비어 있을 수 없습니다.")

        # 임배딩 생성
        situation_vector, decision_vector, ws, wd = build_embeddings(query=req.query)

        # 임베딩 유사도 기반 1차 후보 랭킹
        first_candidates = first_rank(
            situation_vector=situation_vector,
            decision_vector=decision_vector,
            ws=ws,
            wd=wd,
            min_embedding_score=MIN_EMBEDDING_SCORE,
        )

        # 태그 유사도 기반 최종 재랭킹
        reranked = final_rank(
            query=req.query,
            first_candidates=first_candidates,
            tag_score_weight=TAG_SCORE_WEIGHT,
            min_final_score=MIN_FINAL_SCORE,
        )

        # 최종 추천 결과 반환
        return RecommendationResponse(
            query=req.query,
            results=[map_to_RecommendationItem(row) for row in reranked],
        )
