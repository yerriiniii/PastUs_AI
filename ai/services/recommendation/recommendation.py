from ai.infra.vector_store_provider import get_vector_store
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


class RecommendationService:

    @staticmethod
    def index_embeddings(req: IndexEmbeddingRequest) -> IndexEmbeddingResponse:
        # 게시글 본문에서 상황/결정 텍스트를 만들고 임베딩으로 변환한 뒤 저장한다.
        situation_text = f"{req.situation} {req.outcome} {req.reflection}".strip()
        decision_text = req.choice.strip() or req.situation.strip()

        get_vector_store().upsert_post_embeddings(
            post_id=req.postId,
            situation_embedding=encode_text(situation_text),
            decision_embedding=encode_text(decision_text),
        )
        return IndexEmbeddingResponse(postId=req.postId, stored=True)


    @staticmethod
    def index_tags(req: IndexTagsRequest) -> IndexTagsResponse:
        # 게시글 태그(tagId/tagName)를 AI 추천 DB에 동기화 저장한다.
        if not req.tagIds and not req.tagNames:
            raise ValueError("tagIds 또는 tagNames 중 하나는 필요합니다.")

        get_vector_store().upsert_post_tags(
            post_id=req.postId,
            tag_ids=req.tagIds,
            tag_names=req.tagNames,
        )
        return IndexTagsResponse(
            postId=req.postId,
            tagIds=req.tagIds,
            tagNames=req.tagNames,
            stored=True,
        )

    @staticmethod
    def recommend(req: RecommendationRequest) -> RecommendationResponse:
        # 쿼리 임베딩 + 태그 재랭킹 하이브리드 방식으로 유사 사례를 추천한다.
        if not req.query.strip():
            raise ValueError("query는 비어 있을 수 없습니다.")

        situation_vec, decision_vec, ws, wd = build_embeddings(
            query=req.query,
        )
        first_candidates = first_rank(
            situation_vector=situation_vec,
            decision_vector=decision_vec,
            ws=ws,
            wd=wd,
            min_embedding_score=req.minEmbeddingScore,
        )
        reranked = final_rank(
            query=req.query,
            first_candidates=first_candidates,
            tag_score_weight=req.tagWeight,
            min_final_score=req.minFinalScore,
        )

        return RecommendationResponse(
            query=req.query,
            results=[map_to_RecommendationItem(row) for row in reranked],
        )
