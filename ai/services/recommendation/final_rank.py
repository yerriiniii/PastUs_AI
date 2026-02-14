from typing import Any, Dict, List, Set

from ai.schemas.tagging.tagging import TagRequest
from ai.services.tagging.tagging import TagService


# 검색어 쿼리 태깅
def classify_query_tags(query: str) -> List[int]:

    # 검색어 쿼리를 게시글 형태로 만들어 태그 분류 서비스에 요청
    query_tag_response = TagService.classify_tags(
        TagRequest(
            postId=-1,
            situation=query,
            choice="",
            outcome="",
            reflection="",
        )
    )

    query_tags: List[int] = []
    
    # 검색어 태깅 결과에서 tagId만 추출하여 리스트에 추가
    for query_tag in query_tag_response.tags:
        query_tags.append(query_tag.tagId)
    return query_tags


# 검색어 문장 태그와 1차 후보군 태그 사이의 자카드 유사도 계산
def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:  # 두 집합의 태그가 모두 비어있으면 0.0 반환
        return 0.0
    return len(a & b) / (len(a | b) + 1e-9)


# 태그 유사도를 바탕으로 후보군 최종 랭킹
def final_rank(
    query: str,
    first_candidates: List[Dict[str, Any]],
    tag_score_weight: float,
    min_final_score: float,
) -> List[Dict[str, Any]]:
    query_tags = classify_query_tags(query)  # 검색어 쿼리 태깅 함수 호출
    query_tags_set = set(query_tags)  # 검색어 문장 태그를 집합으로 변환
    reranked_candidates = []

    # 각 후보들에 대해 자카드 반복 호출 -> 태그 유사도 점수 계산
    for row in first_candidates:
        post_tags = set(row.get("tag_ids") or [])
        tag_score = _jaccard(query_tags_set, post_tags)
        row["tag_score"] = tag_score
        row["final_score"] = float(row["embedding_score"]) + tag_score_weight * tag_score
        if row["final_score"] >= min_final_score:
            reranked_candidates.append(row)

    # 재정렬된 후보들을 점수 기준으로 내림차순 정렬 후 반환
    reranked_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked_candidates
