from typing import Optional

from ai.services.recommendation.upsert_db import RecommendationUpsertDB

_upsert_db: Optional[RecommendationUpsertDB] = None


# Upsert DB 인스턴스 생성 및 재사용
def get_upsert_db() -> RecommendationUpsertDB:
    global _upsert_db
    if _upsert_db is None:
        _upsert_db = RecommendationUpsertDB()
    return _upsert_db
