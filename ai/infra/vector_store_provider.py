from typing import Optional

from ai.services.recommendation.upsert_db import RecommendationUpsertDB

_vector_store: Optional[RecommendationUpsertDB] = None


# 벡터 저장소 인스턴스 생성 및 재사용
def get_vector_store() -> RecommendationUpsertDB:

    global _vector_store
    if _vector_store is None:
        _vector_store = RecommendationUpsertDB()
        
    return _vector_store
