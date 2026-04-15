# PerfectOCR/core/workers/vectorial_transformation/math_max.py
import pandas as pd # type: ignore
import logging
import numpy as np
import time
from itertools import permutations
from typing import Dict, Any, List, Tuple, cast
from decimal import Decimal, InvalidOperation
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from services.output_service import save_debug_table

logger = logging.getLogger(__name__)
ONE = Decimal('1.00')
ZERO = Decimal('0.00')

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
        start_time = time.perf_counter()
        try:
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

            logger.info("Tabla recibida para corrección matemática:\n" + df.to_string(index=True))

            corrected_df = self.solve(df, table_matrix)

            if corrected_df.empty:
                logger.error("SE DEVOLVIÓ DATA FRAME VACÍO")
                return False
            
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

            if not manager.save_structured_table(corrected_df):
                logger.info("No se pudo guardar data frame")
                return False

            logger.info(f"Corrección matemática completada en {time.perf_counter() -start_time:.6f}'s")
            return True
        except Exception as e:
            logger.error(f"Error en MatrixSolver.vectorize: '{e}'", exc_info=True)
        return False
            
    def solve(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Fase 1: Identifica roles C, PU, MTL usando clasificación semántica
        de polígonos y votación global con aritmética Decimal.
        """
        H = df.shape[1]

        # --- PASO 0: Validación de Soledad ---
        aritmetic_df, dec_cols = self.get_full_rows(df, table_matrix, H)
        if aritmetic_df.empty:
            logger.error("SIN COLUMNAS SUFICINETES PARA VALIDACIÓN ARITMETICA")
            return df.iloc[0:0]
            
        df = self._find_hypotesis(df, aritmetic_df, dec_cols)
        if df.empty:
            logger.error("DATAFRAME VACÍO")
            return df.iloc[0:0]

        # logger.info("RENAMED:\n" + df.to_string(index=True))
        df = self._correct_df(df, table_matrix, H, dec_cols)
        if df.empty:
            return df.iloc[0:0]
        return df
        
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
        
    def get_arrays_table(self, table_matrix: List[List[Dict[str, Any]]], H: int) -> np.ndarray[Any, np.dtype[np.int8]]:
        """"Devuelve [matrix_decimal, matrix_quantity, elements_array, textual_arrays]"""
        R = len(table_matrix)
        matrix_array = np.zeros((R, H), np.int8)
        matrix_decimal = matrix_array.copy()
        matrix_quantity = matrix_array.copy()
        elements_array = matrix_array.copy()
        textual_array = matrix_array.copy()
        
        for row_id, rows in enumerate(table_matrix):
            rows = table_matrix[row_id]
            for i in range(H):
                sc_v = rows[i]["semantic_clasification"]
                
                if not sc_v or not rows[i]["text"] or not rows[i]["polygon_ids"]:
                    elements_array[row_id, i] = 0
                    matrix_decimal[row_id, i] = 0
                    matrix_quantity[row_id, i] = 0
                    textual_array[row_id, i] = 0
                    continue
                
                total_sc = len(sc_v)
                elements_array[row_id, i] = total_sc
                    
                if any(s == 4 for s in sc_v):
                    matrix_decimal[row_id, i] = sum(1 for ch in sc_v if ch == 4)
                    
                if any(s == 5 for s in sc_v):
                    matrix_quantity[row_id, i] = sum(1 for ch in sc_v if ch == 5)
                    
                if any(s == 1 or s == 2 for s in sc_v):
                    textual_array[row_id, i] = sum(1 for ch in sc_v if ch in (1, 2))
                    
        table_arrays = np.stack([matrix_decimal, matrix_quantity, elements_array, textual_array], dtype=np.int8)
        # logger.info(f"ARRAYS TABLE: \n"f"{table_arrays}")
        return table_arrays

    def _find_hypotesis(self, df: pd.DataFrame, aritmetic_df: pd.DataFrame, dec_cols: List[Tuple[str, str]]) ->pd.DataFrame:
        cols_name = [cols[0] for cols in dec_cols if cols[1] == "decimal"]
        # logger.info(f"COLS NAME: {cols_name}, CANTIDAD DE FILAS: {n_rows}")
        # logger.info("ARITMETIC:\n" + aritmetic_df.to_string(index=False))
        try:
            perm_df = aritmetic_df.map(lambda x: Decimal(x))
        except InvalidOperation as e:
            logger.error(f"ERROR CONVIRTIENDO VALORES DEL DF: '{e}'", exc_info=True)
            return df.iloc[0:0]
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
            return df.iloc[0:0]
            
        logger.info(f"TIempo Permutando: {time.perf_counter() - time_t:.6f}'s")
        c_column, pu_column, mtl_column = [np.argmax(np.count_nonzero(array_votes==i, axis=0)) for i in (1, 2, 3)]
        # logger.info(f"INDICES DE COLUMNA: C-PU-MTL: {c_column, pu_column, mtl_column}")
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
    
    def get_full_rows(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]], H: int) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        try:
            matrix_decimal, matrix_quantity, elements_array, textual_array = self.get_arrays_table(table_matrix, H)
            # Matriz con unicamente las filas completas y al menos 1 un decimal y un numerico
            # decimal_mask = (np.count_nonzero(matrix_decimal, axis=1) > 0) & (np.count_nonzero(matrix_quantity, axis=1) > 0)
            full_rows_mask = np.count_nonzero(elements_array, axis=1) >= H
            full_idx = np.where(full_rows_mask)[0]
            
            full_dec = matrix_decimal + matrix_quantity
            # full_dec = full_dec[full_idx]
            # logger.info("FULL_ID:\n"f"{np.column_stack([full_idx, elements_array[full_idx]])}")
            # logger.info("FULL_DEC:\n"f"{np.column_stack([full_idx, full_dec[full_idx]])}")
            full_n = full_idx.size
            if full_n > 0:
                comple_cols_mask = ((np.count_nonzero(full_dec[full_idx], axis=0)) > full_n / 2) | (np.count_nonzero(textual_array[full_idx]== full_n, axis=0))
            else:
                return (df, [])
            
            decimal_cols = np.where(comple_cols_mask)[0]
            
            full_dec = full_dec[:, decimal_cols]
            
            # logger.info("FULL_COL:\n"f"{np.column_stack([np.arange(len(table_matrix)), full_dec])}")
            
            rows_mask = np.count_nonzero(full_dec, axis=1) >= (decimal_cols.shape[0])
            dec_rows = np.where(rows_mask)[0]
            
            # logger.info(f"DECIMAL_COLS: {decimal_cols} SHAPE: {decimal_cols.shape[0]} | DEC_ROWS: {dec_rows} | ROWS_MASK: {rows_mask}")
            
            type_col: List[Tuple[str, str]] = []
            decimal_cols_str: List[str] = []
            for id in range(H):
                if id in decimal_cols:
                    col = f"col_{id:01d}"
                    decimal_cols_str.append(col)
                    type_col.append((col, "decimal"))
                    
                else:
                    col = f"col_{id:01d}"
                    type_col.append((col, "textual"))
                    
            # logger.info(f"TYPOS: '{type_col}'")
            # full_rows_df = df.iloc[full_idx]
            aritmetic_df = df.loc[dec_rows, decimal_cols_str]
            if aritmetic_df.empty: 
                return (df, type_col)
            else:
                # logger.info("FULL ROWS:\n"+ full_rows_df.to_string(index=True))
                # logger.info("FULL DECIMAL COLS:\n" + aritmetic_df.to_string(index=True))
                return (aritmetic_df, type_col)
            
        except Exception as e:
            logger.info(f"ERROR EN ASIGANCIÓN: '{e}'", exc_info=True)
        return (df.iloc[0:0], [])
        
    def _correct_df(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]], H: int, dec_cols: List[Tuple[str, str]]) -> pd.DataFrame:
        targets = ["c_col", "pu_col", "mtl_col"]
        idx_map = np.array([(df.columns.get_loc(name) if name in df.columns else None) for name in targets], np.uint8)
        if idx_map.size==0:
            return df.iloc[0:0]
            
        cols_name: List[Tuple[int, str]] = []
        for i, col_name in enumerate(dec_cols):
            cols_name.append((i, col_name[1]))
        
        matrix_decimal, matrix_quantity, elements_array, _ = self.get_arrays_table(table_matrix, H)
        cols = np.arange(len(table_matrix))
        non_complete_qty = np.setdiff1d(cols, np.nonzero(matrix_quantity[:, idx_map])[0], True)
        non_complete_dec = np.setdiff1d(cols, np.nonzero(matrix_decimal[:, idx_map])[0], True)
        
        # full_dec = matrix_decimal + matrix_quantity
        # logger.info("FULL_ID:\n"f"{np.column_stack([cols, full_dec])}")
        
        if np.all([non_complete_qty.size, non_complete_dec.size]) > 0:
            non_mask = np.sort(np.union1d(non_complete_dec, non_complete_qty))
        elif non_complete_qty.size == 0:
            non_mask = non_complete_dec
        elif non_complete_dec.size ==0:
            non_mask = non_complete_qty
        else:
            logger.warning("TABLA COMPLETA")
            return df
        
        # incomplete = np.count_nonzero(elements_array, axis=1) < H
        # non_dec = np.count_nonzero(full_dec, axis=1) < 1
        # mask = non_dec & incomplete
        # incomplete_mask = np.where(mask)[0]
        # df_doubles = df.iloc[incomplete_mask]
        # logger.info(f"FILAS INCOMPLETAS DECIMALES: {incomplete_mask}")
        # df_tocorrect = df.loc[non_mask, targets]
        # logger.info("INCOMPLETE ROWS:\n"+ df_doubles.to_string(index=True))
        
        text_cols = [cols[0] for cols in dec_cols if cols[1] == "textual"]
        # logger.info("TEXTUAL COLS:\n"+ df.loc[:, text_cols].to_string(index=True))
        
        textual_indices: List[int] = [df.columns.get_loc(name) for name in text_cols if name in df.columns]
        # logger.info(f"TEXTUAL: {textual_indices}, \n"f"TYPO: {type(textual_indices)}")
        
        if not textual_indices:
            logger.warning("No hay columnas textuales para mover el texto.")
            return df
        zeros_repl = np.zeros((matrix_decimal.shape), np.uint8)
        for row_idx in non_mask:
            for target_col_idx in idx_map:
                if target_col_idx is None:
                    continue
                
                # Leemos la matriz semántica directamente en la misma coordenada
                # Si es > 0 en decimal o en quantity, el array nos dice que es un número.
                if matrix_decimal[row_idx, target_col_idx] > 0 or matrix_quantity[row_idx, target_col_idx] > 0:
                    continue  # El array dice que está bien, no lo movemos.
                    
                cell_text = df.iat[row_idx, target_col_idx]
                if pd.isna(cell_text) or not str(cell_text).strip() or str(cell_text).strip() == str(ZERO):
                    continue
                    
                extracted_text = str(cell_text).strip()
                
                # 1. Limpiar la celda original (poner zero)
                df.iat[row_idx, target_col_idx] = ZERO
                zeros_repl[row_idx, target_col_idx] = 1
                
                # 2. Encontrar la columna textual más cercana
                closest_textual_idx = textual_indices[0]
                min_distance = abs(closest_textual_idx - target_col_idx)
                
                for txt_idx in textual_indices[1:]:
                    dist = abs(txt_idx - target_col_idx)
                    if dist < min_distance:
                        min_distance = dist
                        closest_textual_idx = txt_idx
                
                # 3. Concatenar todo el texto a la columna textual
                current_text_target = df.iat[row_idx, closest_textual_idx]
                if pd.isna(current_text_target):
                    current_text_target = ""
                else:
                    current_text_target = str(current_text_target).strip()
                    
                # Respetar el orden original izquierda-derecha al concatenar
                if target_col_idx < closest_textual_idx:
                    new_text = f"{extracted_text} {current_text_target}".strip()
                else:
                    new_text = f"{current_text_target} {extracted_text}".strip()
                    
                df.iat[row_idx, closest_textual_idx] = new_text

        logger.info("CORRECT:\n"+ df.to_string(index=True))
        # logger.info(f"ZEROS: \n"f"{zeros_repl}")
        if np.count_nonzero(zeros_repl) == 0:
            logger.info(f"Tabla completa acomodada correctamente")
            return df
        df = self._complete_decimal_vals(df, zeros_repl)
        if df.empty:
            return df.iloc[0:0]
        return df
        
    def _complete_decimal_vals(self, df: pd.DataFrame, zeros_repl: np.ndarray[Any, np.dtype[np.uint8]]) -> pd.DataFrame:
        rows_id, cols_id = np.nonzero(zeros_repl)
        coords = np.column_stack((rows_id, cols_id))
        
        # logger.info("TO COMPLETE:\n"+ df.to_string(index=True))
        logger.info("ZEROS: \n"f"{zeros_repl} \n"f"CCORDS: \n"f"{coords}")
        
        c_idx = df.columns.get_loc("c_col") if "c_col" in df.columns else None
        pu_idx = df.columns.get_loc("pu_col") if "pu_col" in df.columns else None
        mtl_idx = df.columns.get_loc("mtl_col") if "mtl_col" in df.columns else None
        
        if c_idx is None or pu_idx is None or mtl_idx is None:
            logger.error("No se encontraron las columnas c_col, pu_col o mtl_col para calcular los valores.")
            return df
            
        for coord in coords:
            r = coord[0]
            c = coord[1]
            
            try:
                val_c = Decimal(str(df.iat[r, c_idx])) if c != c_idx else ZERO
                val_pu = Decimal(str(df.iat[r, pu_idx])) if c != pu_idx else ZERO
                val_mtl = Decimal(str(df.iat[r, mtl_idx])) if c != mtl_idx else ZERO
                
                result = ZERO
                
                if c == c_idx:
                    if val_pu != ZERO and val_mtl != ZERO:
                        result = val_mtl / val_pu
                elif c == pu_idx:
                    if val_c != ZERO and val_mtl != ZERO:
                        result = val_mtl / val_c
                elif c == mtl_idx:
                    if val_c != ZERO and val_pu != ZERO:
                        result = val_c * val_pu
                else:
                    continue  # El cero no está en las columnas objetivo
                    
                df.iat[r, c] = f"{result.quantize(Decimal('0.01'))}"
                
            except (InvalidOperation, ZeroDivisionError, ValueError) as err:
                logger.error(f"Fallo cálculo aritmético en fila {r}, col {c}: {err}")
                continue
        # logger.info("CORRECTED:\n"+ df.to_string(index=True))
        return df