import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import psycopg2
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

load_dotenv()

class PostgresService:
    def __init__(self, dsn: Optional[str] = None):
        """
        dsn opcional; si no, toma de env: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
        """
        self.dsn = dsn or self._build_dsn_from_env()

    def _build_dsn_from_env(self) -> str:
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        dbname = os.getenv("DB_NAME")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        if not all([host, port, dbname, user, password]):
            raise RuntimeError("Faltan variables de entorno para Postgres")
        return f"host={host} port={port} dbname={dbname} user={user} password={password}"

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(self.dsn)
            yield conn
        finally:
            if conn:
                conn.close()

    def insert_payload(self, payload: Dict[str, Any]) -> bool:
        """
        Ejemplo genérico: insertar en tabla 'registros' (ajustar según esquema).
        Se recomienda serializar payload a JSON y guardarlo.
        """
        try:
            import json
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO registros(payload_json, created_at) VALUES (%s, NOW()) RETURNING id",
                    (json.dumps(payload, ensure_ascii=False),)
                )
                inserted = cur.fetchone()
                conn.commit()
                logger.info("Payload insertado en Postgres id=%s", inserted and inserted[0])
            return True
        except Exception as e:
            logger.error("Error insertando payload en Postgres: %s", e, exc_info=True)
            return False