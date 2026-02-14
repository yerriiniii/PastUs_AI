from typing import List

from ai.infra.db_connection import DBconnection


class RecommendationUpsertDB(DBconnection):

    # 게시글 임베딩 저장/갱신
    def upsert_post_embeddings(
        self,
        post_id: int,
        situation_embedding: List[float],
        decision_embedding: List[float],
    ) -> None:

        # SQL 쿼리
        sql_query = """
        insert into ai_post_recommendations (
            "postId", "situationEmbedding", "decisionEmbedding", "tagId", "tagName", "updatedAt"
        ) values (
            %(post_id)s, %(situation_embedding)s, %(decision_embedding)s, '{}'::bigint[], '{}'::text[], now()
        )
        on conflict ("postId") do update set
            "situationEmbedding" = excluded."situationEmbedding",
            "decisionEmbedding" = excluded."decisionEmbedding",
            "updatedAt" = now();
        """

        # DB 연결 -> SQL 실행 -> 커밋
        with self._connect_db() as conn, conn.cursor() as cur:
            cur.execute(
                sql_query,
                {
                    "post_id": post_id,
                    "situation_embedding": situation_embedding,
                    "decision_embedding": decision_embedding,
                },
            )
            conn.commit()


    # 게시글 태그 저장/갱신
    def upsert_post_tags(
        self,
        post_id: int,
        tag_ids: List[int],
        tag_names: List[str],
    ) -> None:

        # SQL 쿼리
        sql_query = """
        update ai_post_recommendations
        set "tagId" = %(tag_ids)s,
            "tagName" = %(tag_names)s,
            "updatedAt" = now()
        where "postId" = %(post_id)s;
        """

        # DB 연결 -> SQL 실행 -> 커밋
        with self._connect_db() as conn, conn.cursor() as cur:
            cur.execute(
                sql_query,
                {
                    "post_id": post_id,
                    "tag_ids": tag_ids,
                    "tag_names": tag_names,
                },
            )
            if cur.rowcount == 0:
                raise KeyError("post_id not found in ai_post_recommendations. index-embedding first.")
            conn.commit()
