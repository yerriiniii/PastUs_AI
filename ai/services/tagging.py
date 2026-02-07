from ai.ml.tag_classification import TagClassifier
from ai.schemas.tagging import TagRequest, TagResponse, TagResult
from ai.services.utils import merge_content 

BASE_MODEL_VERSION = "koelectra-v4"
THRESHOLD = 0.5
MODEL_PATH = "ai/model/tag_classification"

# 모델 로드
classifier = TagClassifier(MODEL_PATH)


class TagService:
    @staticmethod
    def tag_classification(req: TagRequest) -> TagResponse:
     
        # 포맷 병합
        merged_text = merge_content(req)

        # 모델 예측
        raw_results = classifier.predict(merged_text)

        # TagResult 타입 맞추기
        tags = [TagResult(**r) for r in raw_results]

        return TagResponse(
            postId=req.postId,
            modelVersion=BASE_MODEL_VERSION,
            threshold=THRESHOLD,
            tags=tags
        )