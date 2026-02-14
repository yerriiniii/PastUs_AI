import os
from typing import List

from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BM-K/KoSimCSE-roberta-multitask")
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)


# 단일 텍스트를 정규화 임베딩 벡터로 변환
def encode_text(text: str) -> List[float]:
    
    return embedding_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].tolist()
