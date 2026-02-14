from typing import Any, Dict, List

from ai.infra.upsert_db_provider import get_upsert_db


# 코사인 유사도 기반 1차 후보 조회
def first_rank(
    situation_vector: List[float],
    decision_vector: List[float],
    ws: float,
    wd: float,
    min_embedding_score: float,
) -> List[Dict[str, Any]]:

    # SQL 쿼리
    sql_query = """
    with scored as (
        select
            "postId" as post_id,
            "tagId" as tag_ids,
            "tagName" as tag_names,
            (1 - ("situationEmbedding" <=> %(situation_vector)s::vector)) as situation_sim,
            (1 - ("decisionEmbedding" <=> %(decision_vector)s::vector)) as decision_sim,
            (
                %(ws)s * (1 - ("situationEmbedding" <=> %(situation_vector)s::vector)) +
                %(wd)s * (1 - ("decisionEmbedding" <=> %(decision_vector)s::vector))
            ) as embedding_score
        from ai_post_recommendations
    )
    select *
    from scored
    where embedding_score >= %(min_embedding_score)s
    order by embedding_score desc;
    """

    # DB 연결 -> SQL 실행 -> 결과 반환
    upsert_db = get_upsert_db()
    with upsert_db._connect_db() as conn, conn.cursor() as cur:
        cur.execute(
            sql_query,
            {
                "situation_vector": situation_vector,
                "decision_vector": decision_vector,
                "ws": ws,
                "wd": wd,
                "min_embedding_score": min_embedding_score,
            },
        )
        return cur.fetchall()
