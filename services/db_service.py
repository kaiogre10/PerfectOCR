import psycopg2
import logging
import os
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dotenv import load_dotenv
from contextlib import contextmanager
from psycopg2.extras import execute_values # type: ignore

logger = logging.getLogger(__name__)

load_dotenv()

class DataBaseService:
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

    def insert_payload(self, payload: List[Tuple[pd.DataFrame, Dict[str, str]]]) -> bool:
        """
        Inserta solo las filas de DetallesCompra para cada (df, metadata).
        Usa IDRegistro e IDProveedor tal cual vienen en metadata.
        Omite headers/fechas.
        Ajusta los nombres de columnas SQL si tu esquema difiere.
        """
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()

                for df, metadata in payload:
                    id_registro = metadata.get("id_registro")
                    id_proveedor = metadata.get("id_proveedor")

                    if id_registro is None:
                        logger.warning("Skipping payload item without IDRegistro")
                        continue

                    if df.empty:
                        logger.debug("No detalles para IDRegistro=%s", id_registro)
                        continue

                    # Detectar si existe columna opcional 'code_col'
                    has_code = "code_col" in df.columns

                    # Preparar tuples para bulk insert
                    if has_code:
                        records = [
                            (
                                id_registro,
                                id_proveedor,
                                row.get("c_col"),
                                row.get("text_col"),
                                row.get("pu_col"),
                                row.get("mtl_col"),
                                row.get("code_col"),
                            )
                            for _, row in df.iterrows()
                        ]
                        sql = (
                            "INSERT INTO detallescompra "
                            "(id_registro,cantidad, descripcion_original, precio_unitario, importe) "
                            "VALUES %s"
                        )
                    else:
                        records = [
                            (
                                id_registro,
                                # id_proveedor,
                                row.get("c_col"),
                                row.get("text_col"),
                                row.get("pu_col"),
                                row.get("mtl_col"),
                            )
                            for _, row in df.iterrows()
                        ]
                        sql = (
                            "INSERT INTO detallescompra "
                            "(id_registro, cantidad, descripcion_original, precio_unitario, importe)"
                            "VALUES %s"
                        )

                    # Ejecutar bulk insert
                    if records:
                        execute_values(cur, sql, records)

                conn.commit()
                logger.info("insert_payload: insertados %d documentos", len(payload))
            return True
        except Exception as e:
            logger.error("Error en insert_payload: %s", e, exc_info=True)
        return False