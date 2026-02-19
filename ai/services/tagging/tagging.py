import json
import os

from ai.ml.tag_classification import TAGS, THRESHOLD, TagClassifier
from ai.schemas.tagging.tagging import TagRequest, TagResponse, TagResult
from ai.services.tagging.utils import merge_content 

BASE_MODEL_VERSION = os.getenv("TAG_MODEL_VERSION", "koelectra-v4")
MODEL_PATH = os.getenv("TAG_MODEL_PATH", "ai/model/tag_classification")


# 태그 이름 <-> 태그 아이디 매핑 표 생성
def build_tag_id_map():
    raw = os.getenv("TAG_ID_MAP_JSON")
    if raw:
        try:
            parsed = json.loads(raw)
            return {str(k): int(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"TAG_ID_MAP_JSON 설정이 올바르지 않습니다: {e}")

    return {name: idx for idx, name in enumerate(TAGS, start=1)}


TAG_ID_MAP = build_tag_id_map()

# 모델 로드
classifier = TagClassifier(MODEL_PATH)


class TagService:
    
    @staticmethod
    def classify_tags(req: TagRequest) -> TagResponse:

        # 쿼리 검증 시에는 situation만 사용하기 때문
        text_fields = [req.situation, req.choice, req.outcome, req.reflection]
        if not any(field.strip() for field in text_fields):
            raise ValueError("situation, choice, outcome, reflection 중 하나 이상은 필요합니다.")

        # 포맷 병합
        merged_text = merge_content(req)

        # 모델 예측
        raw_results = classifier.predict(merged_text)

        # TagResult 타입 맞추기 (tagId + tagName 동시 반환)
        tags = []
        for r in raw_results:
            tag_name = r["tag"]
            if tag_name not in TAG_ID_MAP:
                raise ValueError(f"태그 ID 매핑이 없습니다: {tag_name}")

            tags.append(
                TagResult(
                    tagId=TAG_ID_MAP[tag_name],
                    tagName=tag_name,
                    confidence=r["confidence"],
                )
            )

        # 최종 태깅 응답 반환
        return TagResponse(
            postId=req.postId,
            modelVersion=BASE_MODEL_VERSION,
            threshold=THRESHOLD,
            tags=tags
        )