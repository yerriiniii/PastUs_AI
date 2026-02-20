import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


class DBconnection:
    
    # DB 접속 정보 로딩
    def __init__(self):
        self.db_url = os.getenv("SUPABASE_DB_URL")
        if not self.db_url:
            raise ValueError("SUPABASE_DB_URL is not set.")

    # DB 커넥션 생성 및 벡터 확장 등록
    def _connect_db(self):
        conn = psycopg.connect(self.db_url, row_factory=dict_row)
        register_vector(conn)
        return conn
