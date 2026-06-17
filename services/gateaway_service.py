import logging
import os
from typing import Optional
from dotenv import load_dotenv
from contextlib import contextmanager
# from services.system_service import clean_db

load_dotenv()

logger = logging.getLogger(__name__)

class ServiceGateaway:
    def __init__(self, dsn: Optional[str] = None):
        """Servicio encargado de gestionar, testear y obtener conexiones con otros servicios de manera local o remota"""
        self.dsn = dsn or self._build_dsn_from_env() # dsn opcional; si no, toma de env: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

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
    def get_local_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(self.dsn)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def test_local_connection(self) -> bool:
        try:
            with self.get_local_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except ConnectionRefusedError:
            logger.warning("Sin conexión a Postgres")
        return False
