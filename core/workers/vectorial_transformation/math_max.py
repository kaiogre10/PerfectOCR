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
from core.domain.data_models import Polygons
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

            df, df_copy = self._table_matrix_to_dataframe(table_matrix, H)
            if df.empty:
                logger.error("La table_matrix no contiene filas/columnas válidas")
                return False

            # logger.debug(
            #     "Tablas recibidas para corrección matemática:\n"
            #     f"DataFrame original:\n{df.to_string(index=True)}\n\n"
            #     f"DataFrame copia:\n{df_copy.to_string(index=True)}"
            # )
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            cut_polygons = self.map_polygons_ids(polygons, df_copy)
            context["cut_polygons"] = cut_polygons
            context["df_copy"] = df_copy
       
            corrected_df = self.solve(df, H, context)

            if corrected_df.empty:
                logger.error("SE DEVOLVIÓ DATA FRAME VACÍO")
                return False
                
            elif not manager.save_structured_table(corrected_df):
                logger.error("No se pudo guardar data frame")
                return False

            logger.info(f"Corrección matemática completada en {time.perf_counter() -start_time:.6f}'s")
            
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
                save_debug_table(corrected_df, file_name, worker_name, header_polygons)
            return True
        except Exception as e:
            logger.error(f"Error en MatrixSolver.vectorize: '{e}'", exc_info=True)
        return False
            
    def solve(self, df: pd.DataFrame, H: int, context: Dict[str, Any]) -> pd.DataFrame:
        """
        Fase 1: Identifica roles C, PU, MTL usando clasificación semántica
        de polígonos y votación global con aritmética Decimal.
        """
        aritmetic_df, dec_cols = self.get_decimal_df(df, H, context)
        dec_rows_ids = aritmetic_df.index.to_numpy()
   
        # logger.info("ARITHMETIC IDS:\n"f"{dec_rows_ids}")
        if aritmetic_df.empty:
            logger.error("SIN COLUMNAS SUFICINETES PARA VALIDACIÓN ARITMETICA")
            return df.iloc[0:0]
            
        df = self.find_hypotesis(df, aritmetic_df, dec_cols)
        if df.empty:
            logger.error("DATAFRAME VACÍO")
            return df.iloc[0:0]

        logger.info("RENAMED:\n" + df.to_string(index=True))
        df = self.correct_df(df, H, dec_cols, dec_rows_ids, context)
        if df.empty:
            return df.iloc[0:0]
        return df
        
    def _table_matrix_to_dataframe(self, table_matrix: List[List[Dict[str, Any]]], H: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        columns = [f"col_{i}" for i in range(H)]
        width = len(table_matrix[0])
        
        rows: List[List[str]] = []
        rows_copy: List[List[List[str]]] = []
        
        for row in table_matrix:
            row_values: List[str] = []
            row_copy_values: List[List[str]] = []
            
            for col_idx in range(width):
                text_val = ""
                poly_ids: List[str] = []
                if col_idx < len(row):
                    text_val = str(row[col_idx].get("text", "") or "")
                    poly_ids = list(row[col_idx].get("polygon_ids", []))
                    
                row_values.append(text_val)
                row_copy_values.append(poly_ids)
                
            rows.append(row_values)
            rows_copy.append(row_copy_values)
        df_main = pd.DataFrame(rows, columns=columns)
        df_map = pd.DataFrame(rows_copy, columns=columns)
        return (df_main, df_map)
    
    def map_polygons_ids(self, polygons: Dict[str, Polygons], df_copy: pd.DataFrame) -> Dict[str, Any]:
        poly_ids: List[str] = []
        for cell in df_copy.to_numpy().ravel():
            if isinstance(cell, list):
                poly_ids.extend(cell)
            elif isinstance(cell, str):
                poly_ids.append(cell)
        
        cut_polygons: Dict[str, Any] = {}
        for poly_id in poly_ids:
            if poly_id in polygons:
                cut_polygons[poly_id] = {
                    "text": polygons[poly_id].ocr_text or "",
                    "semantic_clasification": polygons[poly_id].semantic_clasification,
                }
                
        return cut_polygons
        
    def get_arrays_table(self, H: int, context: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """"Devuelve [matrix_decimal, matrix_quantity, elements_array, textual_arrays]"""
        df_copy: pd.DataFrame = context["df_copy"]
        cut_polygons: Dict[str, Dict[str, Any]] = context["cut_polygons"]
        # logger.info("DataFrame copia:\n"f"{df_copy.to_string(index=True)}")
        R = df_copy.shape[0]
        
        matrix_array = np.zeros((R, H), np.uint8)
        matrix_decimal = matrix_array.copy()
        matrix_quantity = matrix_array.copy()
        elements_array = matrix_array.copy()
        textual_array = matrix_array.copy()
        code_array = matrix_array.copy()
        
        for row_id in range(R):
            for i in range(H):
                cell_poly_ids = []
                if i < df_copy.shape[1]:
                    cell = df_copy.iat[row_id, i]
                    if isinstance(cell, list):
                        cell_poly_ids = cell
                    elif isinstance(cell, str) and cell:
                        cell_poly_ids = [cell]

                if not cell_poly_ids:
                    elements_array[row_id, i] = 0
                    matrix_decimal[row_id, i] = 0
                    matrix_quantity[row_id, i] = 0
                    textual_array[row_id, i] = 0
                    code_array[row_id, i] = 0
                    continue

                sc_v: List[int] = []
                has_text = False
                for poly_id in cell_poly_ids:
                    poly_data = cut_polygons.get(poly_id)
                    if not poly_data:
                        continue
                    poly_sc = poly_data["semantic_clasification"]
                    if isinstance(poly_sc, list):
                        sc_v.extend(poly_sc)
                    if poly_data.get("text"):
                        has_text = True

                if not sc_v or not has_text:
                    elements_array[row_id, i] = 0
                    matrix_decimal[row_id, i] = 0
                    matrix_quantity[row_id, i] = 0
                    textual_array[row_id, i] = 0
                    code_array[row_id, i] = 0
                    continue

                total_sc = len(sc_v)
                elements_array[row_id, i] = total_sc
                    
                if any(s == 4 for s in sc_v):
                    matrix_decimal[row_id, i] = sum(1 for ch in sc_v if ch == 4)
                    
                if any(s == 5 for s in sc_v):
                    matrix_quantity[row_id, i] = sum(1 for ch in sc_v if ch == 5)
                    
                if any(s in (1, 2) for s in sc_v):
                    textual_array[row_id, i] = sum(1 for ch in sc_v if ch in (1, 2))
                
                if any(s == 3 for s in sc_v):
                    code_array[row_id, i] = sum(1 for ch in sc_v if ch == 3)
                                   
        table_arrays = np.stack([matrix_decimal, matrix_quantity, elements_array, textual_array, code_array], dtype=np.uint8)
        # logger.info(f"ARRAYS TABLE: \n"f"{table_arrays}")
        return table_arrays

    def find_hypotesis(self, df: pd.DataFrame, aritmetic_df: pd.DataFrame, dec_cols: List[Tuple[str, str]]) ->pd.DataFrame:
        cols_name = [cols[0] for cols in dec_cols if cols[1] == "decimal"]
        # logger.info(f"COLS NAME: {cols_name}")
        # logger.info("ARITMETIC:\n" + aritmetic_df.to_string(index=True))
        try:
            perm_df = aritmetic_df.map(lambda x: Decimal(x))
        except InvalidOperation as e:
            logger.debug(f"ERROR CONVIRTIENDO VALORES DEL DF: '{e}'", exc_info=True)
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
            
        # logger.info(f"TIempo Permutando: {time.perf_counter() - time_t:.6f}'s")
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
    
    def get_decimal_df(self, df: pd.DataFrame, H: int, context: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        try:
            arrays_table = self.get_arrays_table(H, context)
            matrix_decimal, matrix_quantity, elements_array = arrays_table[0], arrays_table[1], arrays_table[2]
            textual_array = arrays_table[3] + arrays_table[4]
            full_rows_mask = np.count_nonzero(elements_array, axis=1) >= H
            full_idx = np.where(full_rows_mask)[0]                          # índices originales sin celdas vacías
            # full_array = elements_array[full_idx]
            
            full_dec = matrix_decimal + matrix_quantity                         # Array fusionado que contiene los numericos y cuantitativos: "DECIMAL"
            full_dec = full_dec[full_idx]                                       # Array Decimal sin celdas faltantes
            # logger.info("\n"f"{full_dec}")
            # logger.info("FULL_DEC:\n"f"{np.column_stack([full_idx, full_dec])}")
            
            full_dec_mask = np.count_nonzero(full_dec==1, axis=1, keepdims=True)    # Mascara booleana donde hay unicamente un elemento decimal por celda con el mismo shape que el array decimal reducido
            full_idx_dec = np.where(full_dec_mask>= 3)[0]                           # índices del array anterior (No del array original) donde hay suficientes elementos decimales
            full_dec_idx = full_idx[full_idx_dec]                                   # índices originales con filas completas y suficiente numero de decimales
            # logger.info("\n"f"{full_idx}\n"f"{full_idx_dec}\n"f"{full_dec_idx}")
            # full_rows_dfs = df.iloc[full_dec_idx]
            # logger.info("FULL DEC ROWS:\n"+ full_rows_dfs.to_string(index=True))
            
            # logger.info("FULL_DEC_RAW:\n"f"{np.column_stack([full_idx, full_dec[full_idx]])}")
            # logger.info("FULL_ID:\n"f"{np.column_stack([full_idx, elements_array[full_idx]])}")
            
            n_full_rows = full_idx_dec.size
            if n_full_rows > 0:
                textual_mask = (np.count_nonzero(textual_array[full_dec_idx] > 0, axis=0) > n_full_rows // 2)
                textual_cols = np.where(textual_mask==False)[0]                                             # índices originales sin columnas textuales
                full_dec = full_dec[full_idx_dec]
                # logger.info(f"{np.column_stack([full_idx_dec, full_dec])}")
                non_textual_cols = full_dec[:, textual_cols]                                                # Array con columnas textuales/codigo filtradas
                # logger.info("\n"f"{textual_array[full_dec_idx]}, {textual_cols}")
                # logger.info("\n"f"{non_textual_cols}")
                complete_row_idx = np.sum(non_textual_cols, axis=1, keepdims=True)
                complete_dec_idx = np.where(complete_row_idx==textual_cols.size)[0]                         # Índices filas filtradas decimales
                non_textual_cols = non_textual_cols[complete_dec_idx]                                       # Array decimal completamente decimal 
                full_dec_idx = full_dec_idx[complete_dec_idx]                                               # índices originales de filas completas
                # logger.info("\n"f"{non_textual_cols}, {full_dec_idx}")
            else:
                return (df, [])
            
            type_col: List[Tuple[str, str]] = []
            decimal_cols_str: List[str] = []
            for id in range(H):
                if id in textual_cols:
                    col = f"col_{id:01d}"
                    decimal_cols_str.append(col)
                    type_col.append((col, "decimal"))
                    
                else:
                    col = f"col_{id:01d}"
                    type_col.append((col, "textual"))
                    
            # logger.info(f"TYPOS: '{type_col}'")
            aritmetic_df = df.loc[full_dec_idx, decimal_cols_str]
            if aritmetic_df.empty: 
                return (df, type_col)
            else:
                # logger.info("FULL DECIMAL COLS:\n" + aritmetic_df.to_string(index=True))
                return (aritmetic_df, type_col)
            
        except Exception as e:
            logger.warning(f"ERROR EN ASIGANCIÓN: '{e}'", exc_info=True)
        return (df.iloc[0:0], [])
        
    def correct_df(self, df: pd.DataFrame, H: int, dec_cols: List[Tuple[str, str]], dec_rows_ids: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any]) -> pd.DataFrame:
        targets = ["c_col", "pu_col", "mtl_col"]
        idx_map = np.sort(np.array([(df.columns.get_loc(name) if name in df.columns else None) for name in targets], np.uint8))
        # logger.info(f"{idx_map}")
        if idx_map.size==0:
            return df.iloc[0:0]

        cols_name: List[Tuple[int, str]] = []
        for i, col_name in enumerate(dec_cols):
            cols_name.append((i, col_name[1]))
        
        tables_array = self.get_arrays_table(H, context)
        elements_array, textual_array, code_array = tables_array[2], tables_array[3], tables_array[4]
        cols = np.arange(elements_array.shape[1], dtype=np.uint8)
        rows = elements_array.shape[0]
        rows_ids = np.arange(rows, dtype=np.uint8)
        if cols.size == 4:
            descriptive_idx = np.setdiff1d(cols, idx_map, assume_unique=True)
        else:
            non_dec_cols_idx = np.setdiff1d(cols, idx_map, assume_unique=True)          # índices de columnas que no son decimales ni válidas para la validación
            textual_array_temp = textual_array[:, non_dec_cols_idx]
            # logger.info("COLUMNAS TEXTUALES:\n"f"{np.row_stack([non_dec_cols_idx, textual_array])}")
            
            textual_cols_id = np.argmax(np.sum(textual_array_temp, axis=0, dtype=np.uint8))  # índice relativo de columna descriptiva
            descriptive_idx = np.atleast_1d(non_dec_cols_idx[textual_cols_id])          # índice real de columna descriptiva principal
            
            leftovers_idx = np.delete(non_dec_cols_idx, textual_cols_id)                # índices de columnas restantes después de extraer la columna descriptiva principal y las decimales
            pot_code_col = code_array[:, leftovers_idx]
            code_rows = np.count_nonzero(pot_code_col)
            if code_rows > (rows // 2):
                code_col_idx = leftovers_idx                                            # índice real de la única columna code
                logger.info(f"{code_col_idx}")
                logger.info("\n"f"{np.column_stack([rows_ids, elements_array[:, code_col_idx], code_array[:, code_col_idx], textual_array[:, code_col_idx]])}")
            else:
                code_col_idx = None
            
            # if code_col_idx is not None:
            #     mapped_cols = np.sort(np.concatenate([descriptive_idx, idx_map, code_col_idx]))
            # else:
            #     mapped_cols = np.sort(np.concatenate([descriptive_idx, idx_map]))
        # logger.info("MAPPED\n"f"{mapped_cols}")
        # logger.info("TEXTUAL COLS NEW:\n"+ df.iloc[:, descriptive_idx].to_string(index=True))
        incomplete_rows_id = np.setdiff1d(rows_ids, dec_rows_ids)        # índices originales de filas a corregir/completar        
        textual_array = textual_array[incomplete_rows_id]
        # logger.info(f"TEXTUAL ARRAY FILTRADO:\n"f"{idx_map}\n"f"{np.column_stack([incomplete_rows_id, textual_array[:, idx_map]])}")
        
        text_in_dec = np.nonzero(textual_array[:, idx_map])
        fil_cols = text_in_dec[1]
        fil_rows = text_in_dec[0]
        
        fil_cols = idx_map[fil_cols]
        fil_rows = incomplete_rows_id[fil_rows]
        
        # logger.info("UBICACIONES INFILTRADOS ARRAY:\n"f"{text_in_dec}\n"f"{fil_rows}, {fil_cols}")
        # logger.info("INFLITRADOS DF:\n"f"{df.iloc[fil_rows, fil_cols]}")
        df_copy: pd.DataFrame = context["df_copy"]
        # logger.info("COPY:\n" + df_copy.to_string(index=True))
        
        values = df.values
        copy_values = df_copy.values
      
        for r, c in zip(fil_rows, fil_cols):
            if c == descriptive_idx:
                continue
            val = values[r, c]
            
            vals = copy_values[r, c]
            dest_vals = copy_values[r, descriptive_idx[0]]
            
            if val == "":
                continue
            
            if c < descriptive_idx:
                values[r, descriptive_idx] = val + " " + values[r, descriptive_idx]             # Original
                copy_values[r, descriptive_idx[0]] = vals + dest_vals                           # Valores espejo
            else:
                values[r, descriptive_idx] = values[r, descriptive_idx] + " " + val             # Original
                copy_values[r, descriptive_idx[0]] = dest_vals + vals                           # Espejo
                
            values[r, c] = ""
            copy_values[r, c] = []
            
        logger.info("CORRECTED:\n" + df.to_string(index=True))
        # logger.info("COPY CORR:\n" + df_copy.to_string(index=True))
        
        context["df_copy"] = df_copy
        
        tables_array = self.get_arrays_table(H, context)
        matrix_decimal, matrix_quantity = tables_array[0], tables_array[1]
        full_dec = matrix_decimal + matrix_quantity
        
        full_dec = full_dec[incomplete_rows_id]                                 # Array decimal a corregir
        elements_array = elements_array[incomplete_rows_id]                     # Array Global a corregir
        
        semi_mask = np.count_nonzero(full_dec==1, axis=1, keepdims=True)        # cantidad de celdas con valor exactamente 1 por fila
        unique_mask = (np.all(full_dec <= 1, axis=1, keepdims=True))
        rows_incomplete = np.where((semi_mask >= 2) & unique_mask)[0]           # índice relativo con filas con un elemento decimal por celda y al menos 2 decimales
        logger.info("DEC\n"f"{full_dec[rows_incomplete]}")
        logger.info("\n"f"{rows_incomplete}")
        
        correct_rows_df = df.iloc[incomplete_rows_id]
        logger.info("TO CORRECT:\n" + correct_rows_df.to_string(index=True))
        
        sum_cells = np.count_nonzero(elements_array, axis=1, keepdims=True)
        empty_cells = np.where(sum_cells < cols.size)[0]                        # índices relativos de las filas que presentan celdas vacías
        incomplete_dec = full_dec[empty_cells]                                  # Filas con celdas vacías y con al menos 2 decimales para operar
        logger.info("\n"f"{incomplete_dec}")
        
        text_cols = [cols[0] for cols in dec_cols if cols[1] == "textual"]
        textual_indices: List[int] = [df.columns.get_loc(name) for name in text_cols if name in df.columns]
        
        non_mask = incomplete_rows_id[rows_incomplete]
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

        # logger.info("CORRECT:\n"+ df.to_string(index=True))
        # logger.info(f"ZEROS: \n"f"{zeros_repl}")
        if np.count_nonzero(zeros_repl) == 0:
            # logger.info(f"Tabla completa acomodada correctamente")
            return df
        df = self._complete_decimal_vals(df, zeros_repl)
        if df.empty:
            return df.iloc[0:0]
        return df
        
    def _complete_decimal_vals(self, df: pd.DataFrame, zeros_repl: np.ndarray[Any, np.dtype[np.uint8]]) -> pd.DataFrame:
        rows_id, cols_id = np.nonzero(zeros_repl)
        coords = np.column_stack((rows_id, cols_id))
        
        # logger.info("TO COMPLETE:\n"+ df.to_string(index=True))
      #  logger.info("ZEROS: \n"f"{zeros_repl} \n"f"CCORDS: \n"f"{coords}")
        
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