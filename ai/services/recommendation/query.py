import re
from typing import List, Tuple

from ai.services.recommendation.embedding import encode_text


# 상황, 결정 문장 가중치
DEFAULT_W_SITUATION = 0.7
DEFAULT_W_DECISION = 0.3


# 입력 쿼리를 상황/결정 텍스트로 분리
def split_query(query: str) -> Tuple[str, str]:
    
    q = query.strip()
    decision_match = re.search(r"([가-힣]{2,})(하)?(을|를)?지", q)
    decision = ""
    situation = q

    if decision_match:
        decision = decision_match.group(0)
        situation = q.replace(decision, "").strip()

    situation = re.sub(
        r"(고민중|고민\s*중|생각중|생각\s*중|결정\s*중|중이다|중임)",
        "",
        situation,
    ).strip()
    
    return situation, decision # 상황, 결정 문장 반환 -> 튜플에 담음


# 임베딩 생성
def build_embeddings(query: str) -> Tuple[List[float], List[float], float, float]:
    
    query_situation, query_decision = split_query(query) # 쿼리 분리 함수 호출
    has_situation = bool(query_situation)
    has_decision = bool(query_decision)

    # 상황, 결정 문장 모두 없으면 에러 발생
    if not has_situation and not has_decision:
        raise ValueError("query에서 유효한 검색 문장을 추출할 수 없습니다.")
    # 상황, 결정 문장 모두 있으면 기본 가중치 사용
    if has_situation and has_decision:
        ws, wd = DEFAULT_W_SITUATION, DEFAULT_W_DECISION
    # 상황 문장만 있으면 상황 문장에 100% 가중치 사용
    elif has_situation:
        ws, wd = 1.0, 0.0
    # 결정 문장만 있으면 결정 문장에 100% 가중치 사용
    else:
        ws, wd = 0.0, 1.0

    situation_input = query_situation if has_situation else query
    decision_input = query_decision if has_decision else query
    
    # 임베딩 생성 함수 호출
    return encode_text(situation_input), encode_text(decision_input), ws, wd
