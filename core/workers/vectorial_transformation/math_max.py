# PerfectOCR/core/workers/vectorial_transformation/math_max.py
import pandas as pd # type: ignore
import logging
import numpy as np
import time
from itertools import permutations
from typing import Dict, Any, List, Tuple, Optional, cast
from decimal import Decimal, InvalidOperation
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from services.output_service import save_debug_table

logger = logging.getLogger(__name__)
ONE = Decimal('1.00')

class MatrixSolver(VectorizationAbstractWorker):
    """
    Resuelve inconsistencias matemáticas en una tabla estructurada usando
    clasificación semántica de polígonos, aritmética Decimal y votación global.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('math_max', {})
        self.total_mtl_tolerance = worker_config.get('total_mtl_abs_tolerance')
        tol = worker_config.get('row_relative_tolerance')
        self.arithmetic_tolerance = Decimal(str(tol)) if tol is not None else Decimal('0.15')
        self.output = config.get("math_max_corrected", False)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        try:
            start_time = time.time()
            if not manager.workflow:
                return False
            H = manager.workflow.H 
            table_matrix = cast(List[List[Dict[str, Any]]], context["table_matrix"])
            if not table_matrix:
                logger.error("No hay table_matrix en contexto para procesar")
                return False

            df = self._table_matrix_to_dataframe(table_matrix, H)
            if df.empty:
                logger.error("La table_matrix no contiene filas/columnas válidas")
                return False

            # logger.info("Tabla recibida para corrección matemática:\n" + df.to_string(index=True))

            corrected_df = self.solve(df, table_matrix)

            if df.equals(corrected_df):
                logger.info("No se corrigió tabla; se conserva versión original")
                return True
            
            if self.output:
                all_lines = manager.workflow.all_lines if manager.workflow else {}
                polygons = manager.workflow.polygons if manager.workflow else {}

                header_line_ids = [lid for lid, l in all_lines.items() if getattr(l, "header_line", False)]
                header_line_id = header_line_ids[0] if header_line_ids else None

                header_polygons = []
                if header_line_id and header_line_id in all_lines:
                    line_obj = all_lines[header_line_id]
                    polygon_ids = getattr(line_obj, "polygon_ids", [])
                    header_polygons = [polygons[pid] for pid in polygon_ids if pid in polygons]

                file_name: str = manager.workflow.metadata.image_name # type: ignore
                worker_name = context.get("worker_name") or "math_max"
                output_paths = context["output_paths"]
                save_debug_table(corrected_df, file_name, output_paths, worker_name, header_polygons)

            # logger.info("Tabla tras corrección matemática:\n" + corrected_df.to_string(index=True))
            # logger.info(f"Cambios:\n" + df.compare(corrected_df).to_string(index=True))
            final_semantic_types = list(corrected_df.columns)
            manager.save_structured_table(df=corrected_df, columns=list(corrected_df.columns), semantic_types=final_semantic_types)

            total_time = time.time() - start_time
            logger.info(f"Corrección matemática completada en {total_time:.6f}s, Se encontraron {len(corrected_df)} filas.")
            return True
        except Exception as e:
            logger.error(f"Error en MatrixSolver.vectorize: {e}", exc_info=True)
            return False
            
    def solve(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Fase 1: Identifica roles C, PU, MTL usando clasificación semántica
        de polígonos y votación global con aritmética Decimal.
        """
        df = df.copy()
        H = df.shape[1]
        # columns: List[str] = list(df.columns)
        
        # logger.info(f"TABLE MATRIX: {table_matrix[0]}")

        # --- PASO 0: Validación de Soledad ---
        aritmetic_df, dec_cols = self._separate_rows(df, table_matrix, H)
        df = self._find_hypotesis(df, aritmetic_df, dec_cols)
        logger.info("RENAMED:\n" + df.to_string(index=False))
        # correct_df = self._correct_df(df)
        return df

    def _move_polygon(self, df: pd.DataFrame, row: List[Dict[str, Any]], row_idx: int, src_col: int, target_col: int, pid: str, polygons_dict: Dict[str, Polygons]) -> None:
        """Mueve un polígono de src_col a target_col y actualiza df y row."""
        cell = row[src_col]
        target_cell = row[target_col]
        poly = polygons_dict.get(pid)
        if not poly or not poly.ocr_text:
            return
            
        displaced_text = poly.ocr_text.strip()
        if not displaced_text:
            return

        # Mover polígono a celda destino
        target_cell['polygon_ids'] = target_cell['polygon_ids'] + [pid]
        target_sc: List[int] = target_cell['semantic_clasification']
        target_cell['semantic_clasification'] = sorted(set(target_sc + list(poly.semantic_clasification)))
        old_target_text = (target_cell.get('text', '') or '').strip()
        
        # Mantener el orden geométrico estricto del texto dependiendo de la dirección original
        # Si la columna destino está a la derecha (target_col > src_col), 
        # el polígono que movemos estaba originalmente a la izquierda, así que va PRIMERO.
        if target_col > src_col:
            target_cell['text'] = f"{displaced_text} {old_target_text}".strip() if old_target_text else displaced_text
        # Si la columna destino está a la izquierda (target_col < src_col),
        # el polígono que movemos estaba originalmente a la derecha, así que va DESPUÉS.
        else:
            target_cell['text'] = f"{old_target_text} {displaced_text}".strip() if old_target_text else displaced_text

        # Actualizar celda fuente
        new_pids = [p for p in cell['polygon_ids'] if p != pid]
        cell['polygon_ids'] = new_pids
        remaining_sc: set[int] = set()
        for rpid in new_pids:
            rpoly = polygons_dict.get(rpid)
            if rpoly:
                remaining_sc.update(rpoly.semantic_clasification)
        cell['semantic_clasification'] = sorted(remaining_sc)
        cell['text'] = (cell.get('text', '') or '').replace(displaced_text, '').strip()

        # Sincronizar DataFrame
        df.iloc[row_idx, src_col] = cell['text']
        df.iloc[row_idx, target_col] = target_cell['text']
        logger.info(f"Soledad: '{displaced_text}' de col_{src_col} -> col_{target_col} (fila {row_idx})")

    def _table_matrix_to_dataframe(self, table_matrix: List[List[Dict[str, Any]]], H: int) -> pd.DataFrame:
        rows: List[List[str]] = []
        columns = [f"col_{i}" for i in range(H)]
        width = len(table_matrix[0])
        for row in table_matrix:
            row_values: List[str] = []
            for col_idx in range(width):
                text_val = ""
                if col_idx < len(row):
                    text_val = str(row[col_idx].get("text", "") or "")
                row_values.append(text_val)
            rows.append(row_values)
        return pd.DataFrame(rows, columns=columns)
        
    def _separate_rows(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]], H: int) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        try:
            R = len(table_matrix)
            cols = range(H)
            matrix_array = np.zeros((R, H), np.int8)
            matrix_decimal = matrix_array.copy()
            matrix_quantity = matrix_array.copy()
            elements_array = matrix_array.copy()
            textual_array = matrix_array.copy()
            
            for row_id, rows in enumerate(table_matrix):
                rows = table_matrix[row_id]
                for i in range(H):
                    sc_v = rows[i]["semantic_clasification"]
                    total_sc = len(sc_v)
                    
                    elements_array[row_id, i] = total_sc
                    
                    if not sc_v or not rows[i]["text"] or not rows[i]["polygon_ids"]:
                        elements_array[row_id, i] = 0
                        matrix_decimal[row_id, i] = 0
                        matrix_quantity[row_id, i] = 0
                        textual_array[row_id, i] = 0
                        continue
                        
                    elif all(s == 4 for s in sc_v):
                        matrix_decimal[row_id, i] = total_sc
                        textual_array[row_id, i] = 0
                        continue
                        
                    elif all(s == 5 for s in sc_v):
                        matrix_quantity[row_id, i] = total_sc
                        textual_array[row_id, i] = 0
                        continue
                    
                    else:
                        elements_array[row_id, i] = total_sc
                        textual_array[row_id, i] = total_sc
            
            # Matriz con unicamente las filas completas y al menos 1 un decimal y un numerico
            decimal_mask = (np.count_nonzero(matrix_decimal, axis=1) > 0) & (np.count_nonzero(matrix_quantity, axis=1) > 0)
            full_rows_mask = decimal_mask & (np.count_nonzero(elements_array, axis=1) >= H)
            full_idx = np.where(full_rows_mask)[0]
            # matrix_full_row = elements_array[full_idx]
            
            n_full = len(full_idx)
            # Columnas decimales: solo celdas con sc=4 en filas completas (submatriz tras full_idx)
            if n_full > 0:
                comple_cols_mask = ((np.count_nonzero(matrix_decimal[full_idx], axis=0) >= 1) | (np.count_nonzero(matrix_quantity[full_idx], axis=0) >= 1))
            else:
                return (df, [])
        
            decimal_cols = np.where(comple_cols_mask)[0]
        
            textual_array = textual_array[:, decimal_cols]
            text_mask = np.where(np.count_nonzero(textual_array, axis=1) == 0)[0]
            
            type_col: List[Tuple[str, str]] = []
            decimal_cols_str: List[str] = []
            for id in cols:
                if id in decimal_cols:
                    col = f"col_{id:01d}"
                    decimal_cols_str.append(col)
                    type_col.append((col, "decimal"))
                    
                else:
                    col = f"col_{id:01d}"
                    type_col.append((col, "textual"))
                    
            # logger.info(f"TYPOS: '{type_col}'")
            # full_rows_df = df.iloc[full_idx]
            aritmetic_df = df.loc[text_mask, decimal_cols_str]
            if aritmetic_df.empty: 
                return (df, type_col)
            else:
                # logger.info("FULL ROWS:\n"+ full_rows_df.to_string(index=False))
                # logger.info("FULL DECIMAL COLS:\n" + aritmetic_df.to_string(index=False))
                return (aritmetic_df, type_col)
            
        except Exception as e:
            logger.info(f"ERROR EN ASIGANCIÓN: '{e}'", exc_info=True)
        return (df.empty, type_col)

    def _find_hypotesis(self, df: pd.DataFrame, aritmetic_df: pd.DataFrame, dec_cols: List[Tuple[str, str]]):
        cols_name = [cols[0] for cols in dec_cols if cols[1] == "decimal"]
        # logger.info(f"COLS NAME: {cols_name}, CANTIDAD DE FILAS: {n_rows}")
        # logger.info("ARITMETIC:\n" + aritmetic_df.to_string(index=False))
        try:
            perm_df = aritmetic_df.map(lambda x: Decimal(x))
        except InvalidOperation as e:
            logger.error(f"ERROR CONVIRTIENDO VALORES DEL DF: '{e}'", exc_info=True)
            return None
        # logger.info("DECIMAL DF:\n" + perm_df.to_string(index=False))
        time_t = time.perf_counter()
        try:
            all_hypotesis = []
            array_votes = np.zeros(perm_df.shape, np.int8)
            for _, row in perm_df.iterrows():
                row_validated = []
                row_values = row.values
                n_cols = len(row_values)
                for c_idx, pu_idx, mtl_idx in permutations(range(n_cols), 3):
                    c_col = row_values[c_idx]
                    pu_col = row_values[pu_idx]
                    mtl_col = row_values[mtl_idx]
                    if mtl_col < pu_col:
                        continue
                    if c_col > mtl_col:
                        continue
                    if c_col > pu_col:
                        continue
                    
                    upper_tol = mtl_col + self.arithmetic_tolerance
                    lower_tol = mtl_col - self.arithmetic_tolerance
                    artimetic_mtl = c_col * pu_col
                    
                    if artimetic_mtl == mtl_col:
                        row_validated.append([c_col, pu_col, mtl_col])
                        array_votes[:, c_idx] = 1
                        array_votes[:, pu_idx] = 2
                        array_votes[:, mtl_idx] = 3
                        break
                    elif artimetic_mtl > lower_tol and artimetic_mtl < upper_tol:
                        row_validated.append([c_col, pu_col, mtl_col])
                        array_votes[:, c_idx] = 1
                        array_votes[:, pu_idx] = 2
                        array_votes[:, mtl_idx] = 3
                        break
                all_hypotesis.append(row_validated)
                    
            # logger.info(f"HIPOTESIS GANADORA: {all_hypotesis} \n"f"ARRAY:\n"f"{array_votes}")
        except ValueError as e:
            logger.error(f"ERROR PERMUTANDO: '{e}'", exc_info=True)
            return None
        logger.info(f"TIempo Permutando: {time.perf_counter() - time_t:.6f}'s")
        c_column, pu_column, mtl_column = [np.argmax(np.count_nonzero(array_votes==i, axis=0)) for i in (1, 2, 3)]
        logger.info(f"INDICES DE COLUMNA: C-PU-MTL: {c_column, pu_column, mtl_column}")
        for i, col in enumerate(cols_name):
            if i == c_column:
                df.rename(columns={col: "c_col"}, inplace=True)
            elif i == pu_column:
                df.rename(columns={col: "pu_col"}, inplace=True)
            elif i == mtl_column:
                df.rename(columns={col: "mtl_col"}, inplace=True)
            else:
                continue
        return df
        
    # def _correct_df(self, df: pd.DataFrame) -> pd.DataFrame:
        