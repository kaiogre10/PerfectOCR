import logging
import subprocess
import pandas as pd  # type: ignor
from typing import Dict, Tuple, List, Any
from psycopg2.extras import execute_values  # type: ignore
from services.gateaway_service import ServiceGateaway

gateaway_service = ServiceGateaway()

logger = logging.getLogger(__name__)

def start_postgres() -> bool:
    subprocess.run(
        ["sc", "start", "postgresql-x64-17"],
        check=False,
        capture_output=True
    )
    return gateaway_service.test_local_connection()
    

def stop_postgres() -> None:
    subprocess.run(
        ["sc", "stop", "postgresql-x64-17"],
        check=False,
        capture_output=True
    )

def insert_payload(payload: List[Tuple[pd.DataFrame, Dict[str, Any]]]):
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
        for row in df_con[detalles_cols].itertuples(index=False, name=None)
        # type: ignore[reportUnknownVariableType]
    ]

    try:
        with get_connection() as conn:
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

# def set_up_connectors(self, final_results: List[Tuple[int, int]]):
#     for i, _ in enumerate(final_results):
#         ptr, buff_size = final_results[i]
#         # try:
#         bytes_leidos = ctypes.string_at(ptr, buff_size)
#             # raise MemoryError("Error leyendo bytecode")
#         # except MemoryError as e:
#         #     logger.warning(f"Error leyendo bytecode: {e}", exc_info=True)
#         logger.info(f"BYTES_ALMACENADOS: '{bytes_leidos}'")
#     return None
