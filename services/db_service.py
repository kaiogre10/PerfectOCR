import psycopg2
import logging
import os
import pandas as pd # type: ignore
from typing import Optional, Dict, Tuple, List, Any
from dotenv import load_dotenv
from contextlib import contextmanager
from psycopg2.extras import execute_values # type: ignore

logger = logging.getLogger(__name__)

load_dotenv()

class DataBaseService:
    def __init__(self, dsn: Optional[str] = None):
        """dsn opcional; si no, toma de env: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS"""
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
    
    def test_connection(self) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            logger.warning("Sin conexión a Postgres")
        return False

    def insert_payload(self, payload: List[Tuple[pd.DataFrame, Dict[str, Any]]]) -> bool:
        """
        Inserta el payload procesado en una sola transaccion:
            1. clientes (ON CONFLICT DO NOTHING)
            2. proveedores (ON CONFLICT DO NOTHING)
            3. registros_compra (encabezados)
            4. detalles_compra (lineas de productos)
        """
        if not payload:
            logger.warning("Payload vacio, nada que insertar")
            return False

        df_con = pd.DataFrame()
        all_data: Dict[str, Dict[str, Any]] = {}

        for final_df in payload:
            df = final_df[0]
            global_data = final_df[1]
            id_registro = global_data.get("id_registro", "")
            if not id_registro:
                logger.warning("Item de payload sin id_registro, se omite")
                continue

            df_con = pd.concat([df_con, df], ignore_index=True)
            all_data[id_registro] = global_data

        if not all_data:
            logger.error("No hay registros validos en el payload")
            return False

        logger.info("TABLA FINAL:\n"f"{df_con.to_string(index=True)}"
            "\n"f"GLOBAL_DATA:\n"f"{all_data}")

        # Deduplicar maestros por su PK antes del bulk insert
        clientes_unicos: Dict[Any, Tuple[Any, Any, Any]] = {}
        proveedores_unicos: Dict[Any, Tuple[Any, Any, Any]] = {}

        for meta in all_data.values():
            id_cliente = meta.get("id_cliente")
            if id_cliente is not None and id_cliente not in clientes_unicos:
                clientes_unicos[id_cliente] = (
                    id_cliente,
                    meta.get("nombre_cliente"),
                    meta.get("giro"),
                )
            id_proveedor = meta.get("id_proveedor")
            if id_proveedor and id_proveedor not in proveedores_unicos:
                proveedores_unicos[id_proveedor] = (
                    id_proveedor,
                    meta.get("proveedor_norm"),
                    meta.get("rfc_prov"),
                )

        registros_records = [
            (
                meta.get("id_registro"),
                meta.get("id_cliente"),
                meta.get("folio_doc"),
                meta.get("date_doc"),
                meta.get("id_proveedor"),
                meta.get("total_doc"),
                meta.get("total_cal"),
                meta.get("total_art"),
                meta.get("art_cal"),
                meta.get("fecha_captura"),
            )
            for meta in all_data.values()
        ]

        detalles_cols = ["id_registro", "cantidad_art", "producto_norm", "precio_unitario", "costo_tran"]
        detalles_records: List[Tuple[Any, ...]] = [
            tuple(row)  # type: ignore[reportUnknownArgumentType]
            for row in df_con[detalles_cols].itertuples(index=False, name=None)  # type: ignore[reportUnknownVariableType]
        ]

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if clientes_unicos:
                        execute_values(
                            cur,
                            "INSERT INTO clientes (id_cliente, nombre_cliente, giro) "
                            "VALUES %s ON CONFLICT (id_cliente) DO NOTHING",
                            list(clientes_unicos.values()),
                        )
                    if proveedores_unicos:
                        execute_values(
                            cur,
                            "INSERT INTO proveedores (id_proveedor, proveedor_norm, rfc_prov) "
                            "VALUES %s ON CONFLICT (id_proveedor) DO NOTHING",
                            list(proveedores_unicos.values()),
                        )
                    if registros_records:
                        execute_values(
                            cur,
                            "INSERT INTO registros_compra "
                            "(id_registro, id_cliente, folio_doc, date_doc, id_proveedor, "
                            "total_doc, total_cal, total_art, art_cal, fecha_captura) "
                            "VALUES %s",
                            registros_records,
                        )
                    if detalles_records:
                        execute_values(
                            cur,
                            "INSERT INTO detalles_compra "
                            "(id_registro, cantidad_art, producto_norm, precio_unitario, costo_tran) "
                            "VALUES %s",
                            detalles_records,
                        )
                conn.commit()
            logger.info(
                "insert_payload: %d clientes, %d proveedores, %d registros, %d detalles",
                len(clientes_unicos), len(proveedores_unicos),
                len(registros_records), len(detalles_records),
            )
            return True
        except Exception as e:
            logger.error("Error en insert_payload: %s", e, exc_info=True)
        return False