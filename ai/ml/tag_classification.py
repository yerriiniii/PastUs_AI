import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TAGS = [
    "군대", "휴학/복학", "성적", "전과", "연애",
    "장학금", "과제/팀플", "대학원", "졸업",
    "새내기", "자취", "편입", "N수/반수", "취업"
]

def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name}는 float 값이어야 합니다: {raw}") from e


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name}는 int 값이어야 합니다: {raw}") from e


THRESHOLD = _get_env_float("TAG_THRESHOLD", 0.5)
MAX_TAGS = _get_env_int("TAG_MAX_TAGS", 4)


class TagClassifier:
    
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path: str):
        return cls(model_path)

    def predict(self, text: str):
        # 입력 토큰화
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # 모델 예측
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # 확률 계산
        probs = torch.sigmoid(logits).squeeze().tolist()

        # 태그 선택
        results = []
        for tag, prob in zip(TAGS, probs):
            if prob >= THRESHOLD:
                results.append({
                    "tag": tag,
                    "confidence": round(prob, 4)
                })

        # 태그 정렬 후 반환
        return sorted(results, key=lambda x: x["confidence"], reverse=True)[:MAX_TAGS]