from typing import Any, Dict

import numpy as np

from ai.schemas.recommendation.recommendation import RecommendationItem


# DB row를 RecommendationItem 응답 스키마로 변환
def map_to_RecommendationItem(row: Dict[str, Any]) -> RecommendationItem:

    return RecommendationItem(
        postId=row["post_id"],
        finalScore=float(row["final_score"] if np.isfinite(row["final_score"]) else 0.0),
    )
