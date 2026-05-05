# PerfectOCR/core/workers/vectorial_transformation/math_max.py
import pandas as pd # type: ignore
import logging
import numpy as np
import time
from itertools import permutations
from typing import Dict, Any, List, Tuple, cast, FrozenSet
from decimal import Decimal, InvalidOperation
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.text_utils import validate_quant_chars
from core.domain.data_models import Polygons
from services.output_service import save_debug_table

logger = logging.getLogger(__name__)

ONE = Decimal('1.00')
ZERO = Decimal('0.00')
DEC_COLS_NAME: FrozenSet[str] = frozenset({"c_col", "pu_col", "mtl_col"})

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
       
            corrected_df = self.solve(df, context)

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
            
    def solve(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """
        Fase 1: Identifica roles C, PU, MTL usando clasificación semántica
        de polígonos y votación global con aritmética Decimal.
        """
        aritmetic_df= self.get_decimal_df(df, context)
        dec_rows_ids = aritmetic_df.index.to_numpy()
   
        # logger.info("ARITHMETIC IDS:\n"f"{dec_rows_ids}")
        if aritmetic_df.empty:
            logger.error("SIN COLUMNAS SUFICINETES PARA VALIDACIÓN ARITMETICA")
            return df.iloc[0:0]
            
        df = self.find_hypotesis(df, aritmetic_df)
        df_copy: pd.DataFrame = context["df_copy"]
        
        df_copy.columns = df.columns
        context["df_copy"] = df_copy
        if df.empty:
            logger.error("DATAFRAME VACÍO")
            return df.iloc[0:0]

        # logger.info("RENAMED:\n" + df.to_string(index=True))
        df = self.correct_df(df, dec_rows_ids, context)
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
                    poly_ids = row[col_idx]["polygon_ids"]
                    
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
        
        cut_polygons: Dict[str, Polygons] = {}
        for poly_id in poly_ids:
            if poly_id in polygons:
                cut_polygons[poly_id] = {
                    "text": polygons[poly_id].ocr_text or "",
                    "semantic_clasification": polygons[poly_id].semantic_clasification,
                }                
        return cut_polygons
        
    def get_arrays_table(self, context: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """"Devuelve [cuantiative_array, numeric_array, elements_array, textual_array, code_array]"""
        df_copy: pd.DataFrame = context["df_copy"]
        cut_polygons: Dict[str, Dict[str, Any]] = context["cut_polygons"]
        R = df_copy.shape[0]
        H = df_copy.shape[1]
        
        matrix_array = np.zeros((R, H), np.uint8)
        cuantiative_array = matrix_array.copy()
        numeric_array = matrix_array.copy()
        elements_array = matrix_array.copy()
        textual_array = matrix_array.copy()
        code_array = matrix_array.copy()
        
        for row_id in range(R):
            for i in range(H):
                cell_poly_ids = df_copy.iat[row_id, i] if i < df_copy.shape[1] else []

                if not cell_poly_ids:
                    elements_array[row_id, i] = 0
                    cuantiative_array[row_id, i] = 0
                    numeric_array[row_id, i] = 0
                    textual_array[row_id, i] = 0
                    code_array[row_id, i] = 0
                    continue

                sc_v: List[int] = []
                has_text = False
                for poly_id in cell_poly_ids:
                    poly_data = cut_polygons.get(poly_id)
                    if not poly_data:
                        continue
                    sc_v.extend(poly_data["semantic_clasification"])
                    if poly_data.get("text"):
                        has_text = True

                if not sc_v or not has_text:
                    elements_array[row_id, i] = 0
                    cuantiative_array[row_id, i] = 0
                    numeric_array[row_id, i] = 0
                    textual_array[row_id, i] = 0
                    code_array[row_id, i] = 0
                    continue

                total_sc = len(sc_v)
                elements_array[row_id, i] = total_sc
                    
                if any(s == 4 for s in sc_v):
                    cuantiative_array[row_id, i] = sum(1 for ch in sc_v if ch == 4)
                    
                if any(s == 5 for s in sc_v):
                    numeric_array[row_id, i] = sum(1 for ch in sc_v if ch == 5)
                    
                if any(s in (1, 2) for s in sc_v):
                    textual_array[row_id, i] = sum(1 for ch in sc_v if ch in (1, 2))
                
                if any(s == 3 for s in sc_v):
                    code_array[row_id, i] = sum(1 for ch in sc_v if ch == 3)
                                   
        table_arrays = np.stack([cuantiative_array, numeric_array, elements_array, textual_array, code_array], dtype=np.uint8)
        # logger.info(f"ARRAYS TABLE: \n"f"{table_arrays}")
        return table_arrays

    def find_hypotesis(self, df: pd.DataFrame, aritmetic_df: pd.DataFrame) ->pd.DataFrame:
        cols_name = list(aritmetic_df.columns)
        # logger.info(f"COLS NAME: {cols_name}")
        # logger.info("ARITMETIC:\n" + aritmetic_df.to_string(index=True))
        try:
            perm_df = aritmetic_df.map(lambda x: Decimal(x))
        except InvalidOperation as e:
            logger.error(f"ERROR CONVIRTIENDO VALORES DEL DF: '{e}'", exc_info=True)
            return df.iloc[0:0]
        # logger.info("DECIMAL DF:\n" + perm_df.to_string(index=False))
        # time_t = time.perf_counter()
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
            
        if all(name in df.columns for name in DEC_COLS_NAME):
            if df.shape[1] == 4:
                for col in df.columns:
                    if col not in DEC_COLS_NAME:
                        df.rename(columns={col: "text_col"}, inplace=True)
                        break
            elif len(cols_name) == 4:
                used_idx = {c_column, pu_column, mtl_column}
                for idx, orig in enumerate(cols_name):
                    if idx not in used_idx:
                        df.rename(columns={orig: "noise_um_col"}, inplace=True)
                        break
        return df
    
    def get_decimal_df(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        H = df.shape[1]
        arrays_table = self.get_arrays_table(context)
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
            return df
        
        decimal_cols_str: List[str] = []
        for id in range(H):
            if id in textual_cols:
                col = f"col_{id:01d}"
                decimal_cols_str.append(col)
            else:
                col = f"col_{id:01d}"
                
        aritmetic_df = df.loc[full_dec_idx, decimal_cols_str]
        if aritmetic_df.empty: 
            return df
        else:
            # logger.info("FULL DECIMAL COLS:\n" + aritmetic_df.to_string(index=True))
            return aritmetic_df
        
    def correct_df(self, df: pd.DataFrame, dec_rows_ids: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any]) -> pd.DataFrame:
        idx_map = np.sort(np.array([(df.columns.get_loc(name) if name in df.columns else None) for name in DEC_COLS_NAME], np.uint8))
        # logger.info(f"{idx_map}")
        if idx_map.size==0:
            return df.iloc[0:0]
        
        h = df.shape[1]
        cols_idx = np.arange(h, dtype=np.uint8)
        rows = df.shape[0]
        tables_array = self.get_arrays_table(context)
        textual_array, code_array = tables_array[3], tables_array[4]
        rows_ids = np.arange(rows, dtype=np.uint8)
        if df.shape[1] == 4 and idx_map.size == 3:
            descriptive_idx = np.setdiff1d(cols_idx, idx_map, assume_unique=True)
            
        else:
            non_dec_cols_idx = np.setdiff1d(cols_idx, idx_map, assume_unique=True)          # índices de columnas que no son decimales ni válidas para la validación
            textual_array_temp = textual_array[:, non_dec_cols_idx]
            # logger.info("COLUMNAS TEXTUALES:\n"f"{np.row_stack([non_dec_cols_idx, textual_array])}")
            
            textual_cols_id = np.argmax(np.sum(textual_array_temp, axis=0, dtype=np.uint8))  # índice relativo de columna descriptiva
            descriptive_idx = np.atleast_1d(non_dec_cols_idx[textual_cols_id])          # índice real de columna descriptiva principal
            
            # leftovers_idx = np.delete(non_dec_cols_idx, textual_cols_id)                # índices de columnas restantes después de extraer la columna descriptiva principal y las decimales
            # pot_code_col = code_array[:, leftovers_idx]
            # code_rows = np.count_nonzero(pot_code_col)
            # if code_rows > (rows // 2):
            #     code_col_idx = leftovers_idx                                            # índice real de la única columna code
            #     # logger.info(f"{code_col_idx}")
            #     # logger.info("\n"f"{np.column_stack([rows_ids, elements_array[:, code_col_idx], code_array[:, code_col_idx], textual_array[:, code_col_idx]])}")
            # else:
            #     code_col_idx = None
        # logger.info("MAPPED\n"f"{mapped_cols}")
        
        df, df_copy = self.isolate_decimals(df, descriptive_idx, context)
        context["df_copy"] = df_copy
        
        df, df_copy = self.separate_decimals(df, descriptive_idx, context)
        context["df_copy"] = df_copy
        tables_array = self.get_arrays_table(context)
        textual_array = tables_array[3]
        
        incomplete_rows_id = np.setdiff1d(rows_ids, dec_rows_ids)        # índices originales de filas a corregir/completar
        textual_array = textual_array[incomplete_rows_id]
        # logger.info(f"TEXTUAL ARRAY FILTRADO:\n"f"{idx_map}\n"f"{np.column_stack([incomplete_rows_id, textual_array[:, idx_map]])}")
        # logger.info("SEPARATED:\n" + df_copy.to_string(index=True))
        
        text_in_dec = np.nonzero(textual_array[:, idx_map])
        fil_cols = text_in_dec[1]
        fil_rows = text_in_dec[0]
        
        fil_cols = idx_map[fil_cols]
        fil_rows = incomplete_rows_id[fil_rows]
        
        # logger.info("UBICACIONES INFILTRADOS ARRAY:\n"f"{text_in_dec}\n"f"{fil_rows}, {fil_cols}")
        # logger.info("INFLITRADOS DF:\n"f"{df.iloc[fil_rows, fil_cols]}")

        dest_idx = int(descriptive_idx[0]) if hasattr(descriptive_idx, "__len__") else int(descriptive_idx)

        for r, c in zip(fil_rows, fil_cols):
            r = int(r) 
            c = int(c)
            if c == dest_idx:
                continue

            val = df.iat[r, c]
            vals = df_copy.iat[r, c]
            dest_vals = df_copy.iat[r, dest_idx]

            if val == "" or val is None:
                continue

            if c < dest_idx:
                df.iat[r, dest_idx] = str(val) + " " + str(df.iat[r, dest_idx])
                df_copy.iat[r, dest_idx] = vals + dest_vals
            else:
                df.iat[r, dest_idx] = str(df.iat[r, dest_idx]) + " " + str(val)
                df_copy.iat[r, dest_idx] = dest_vals + vals

            df.iat[r, c] = ""
            df_copy.iat[r, c] = []
            
        # logger.info("CORRECT TEXT:\n" + df.to_string(index=True))
        # logger.info("COPY CORR:\n" + df_copy.to_string(index=True))
        context["df_copy"] = df_copy
        
        df = self.complete_rows(df, context, incomplete_rows_id, idx_map)
        # logger.info("CORRECTED:\n" + df.to_string(index=True))
        return df
        
    def complete_rows(self, df: pd.DataFrame, context: Dict[str, Any], incomplete_rows_id: np.ndarray[Any, np.dtype[np.uint8]], idx_map: np.ndarray[Any, np.dtype[np.uint8]]) -> pd.DataFrame:
        tables_array = self.get_arrays_table(context)
        matrix_decimal, matrix_quantity, elements_array = tables_array[0], tables_array[1], tables_array[2]
        full_dec = matrix_decimal + matrix_quantity
        
        full_dec = full_dec[incomplete_rows_id]                                 # Array decimal a corregir
        elements_array = elements_array[incomplete_rows_id]                     # Array Global a corregir
        
        # semi_mask = np.count_nonzero(full_dec==1, axis=1, keepdims=True)        # cantidad de celdas con valor exactamente 1 por fila
        # unique_mask = (np.all(full_dec <= 1, axis=1, keepdims=True))
        # rows_incomplete = np.where((semi_mask >= 2) & unique_mask)[0]           # índice relativo con filas con un elemento decimal por celda y al menos 2 decimales
        # logger.info("DEC\n"f"{full_dec[rows_incomplete]}")
        # logger.info("\n"f"{rows_incomplete}")
        
        # sum_cells = np.count_nonzero(elements_array, axis=1, keepdims=True)
        # empty_cells = np.where(sum_cells < H)[0]                        # índices relativos de las filas que presentan celdas vacías
        # incomplete_dec = full_dec[empty_cells]                                  # Filas con celdas vacías y con al menos 2 decimales para operar
        # logger.info("\n"f"{incomplete_dec[:, idx_map]}")
        
        # correct_rows_df = df.iloc[incomplete_rows_id, idx_map]
        # logger.info("TO CORRECT:\n" + correct_rows_df.to_string(index=True))
        
        c_idx = df.columns.get_loc("c_col") if "c_col" in df.columns else None
        pu_idx = df.columns.get_loc("pu_col") if "pu_col" in df.columns else None
        mtl_idx = df.columns.get_loc("mtl_col") if "mtl_col" in df.columns else None

        for r in incomplete_rows_id:
            raw_c = df.iat[r, c_idx].strip()
            raw_pu = df.iat[r, pu_idx].strip()
            raw_mtl = df.iat[r, mtl_idx].strip()

            missing_c = (raw_c == "")
            missing_pu = (raw_pu == "")
            missing_mtl = (raw_mtl == "")

            # Debe haber exactamente una vacía por fila
            if (missing_c + missing_pu + missing_mtl) != 1:
                continue

            val_c = Decimal(raw_c) if not missing_c else ZERO
            val_pu = Decimal(raw_pu) if not missing_pu else ZERO
            val_mtl = Decimal(raw_mtl) if not missing_mtl else ZERO

            if missing_c:
                result = val_mtl / val_pu
                df.iat[r, c_idx] = str(result)
            elif missing_pu:
                result = val_mtl / val_c
                df.iat[r, pu_idx] = str(result)
            else:  # missing_mtl
                result = val_c * val_pu
                df.iat[r, mtl_idx] = str(result)
        return df
        
    def isolate_decimals(self, df: pd.DataFrame, descriptive_idx: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tables_array = self.get_arrays_table(context)
        matrix_decimal, matrix_quantity = tables_array[0], tables_array[1]
        full_dec = matrix_decimal + matrix_quantity
        cols_idx = np.arange(df.shape[1])
        # rows_idx = np.arange(df.shape[0])
        # total_decimals = np.count_nonzero(full_dec, axis=1)
        
        descript_num = full_dec[:, descriptive_idx]          # Array decimal con las columnas textuales
        rows_to_com = np.where(descript_num)[0]              # índices absolutos de Filas con decimales en donde van textuales
        relative_idx = np.arange(rows_to_com.size)
        
        two_decimals = full_dec[rows_to_com]
        two_cols_ids = np.count_nonzero(two_decimals, axis=0) == two_decimals.shape[0]
        idx = cols_idx[two_cols_ids]
        
        invaded_df = df.iloc[rows_to_com, idx]
        # logger.info("\n"f"{invaded_df.to_string(index=True)}")
        # logger.info("INVADED2:\n"f"{df.iloc[rows_to_com].to_string(index=True)}")
        
        # incomplete_rows = elements_array[rows_to_com]
        
        # double_mask = np.count_nonzero(two_decimals, axis=1) == 2   # índices con Filas con exactamente2 decimales
        
        # incomplete_rows = incomplete_rows[double_mask]              # Elements array filtrado con las filas con 2 decimales
        # logger.info(f"MASK2: {rows_to_com}, {relative_idx}")
        invaded_df = invaded_df.map(lambda x: Decimal(x))
        df_copy: pd.DataFrame = context["df_copy"]
        
        for r in relative_idx:
            real_idx = rows_to_com[r]
            # logger.info(f"VALS: {vals}")
            val_n = invaded_df.iat[r, 0]
            val_m = invaded_df.iat[r, 1]
            src_a = invaded_df.columns[0]
            src_b = invaded_df.columns[1]
            poly_m = list(df_copy.at[real_idx, src_a])  # copia defensiva
            poly_n = list(df_copy.at[real_idx, src_b])
            # empty_col = invaded_df2.iloc[r].index[invaded_df2.iloc[r].eq("")][0]
            val_a = max(val_m, val_n)
            val_b = min(val_m, val_n)
            quotient = (val_a / val_b)
            if val_m == val_a:
                poly_a_mtl, poly_b_pu = poly_m, poly_n
            else:
                poly_a_mtl, poly_b_pu = poly_n, poly_m
            # logger.info(f"A: {val_a}, B: {val_b}, COCIENTE: {quotient}")
            if (val_a / val_b) % Decimal("1") == ZERO:
                if val_a > quotient and val_b >= quotient and val_a != quotient:
                    df.at[real_idx, src_a] = ""
                    df.at[real_idx, src_b] = ""
                    
                    df_copy.at[real_idx, src_a] = []
                    df_copy.at[real_idx, src_b] = []
                    
                    df.at[real_idx, "mtl_col"] = str(val_a)
                    df.at[real_idx, "pu_col"] = str(val_b)
                    
                    df_copy.at[real_idx, "mtl_col"] = poly_a_mtl
                    df_copy.at[real_idx, "pu_col"] = poly_b_pu
                    # logger.info(f"VALOR FALTANTE ES C_COL")    
                else:
                    continue
                    # logger.info(f"VALOR FALTANTE ES: '{empty_col}'")
            else:
                continue
                # logger.info(f"NO ENCONTRADO: {quotient}")
                
        # logger.info("CLEAN:\n" + df.to_string(index=True))
        
        return (df, df_copy)
        
    def unmix_cells(self, df: pd.DataFrame, context: Dict[str, Any]):
        cols_idx = np.arange(df.shape[1])
        tables_array = self.get_arrays_table(context)
        cuantiative_array, elements_array, textual_array = tables_array[0], tables_array[2], tables_array[3]
        
        rows_double = np.where(cuantiative_array==2)[0]
        double_cuant = cuantiative_array[rows_double]
        double_element = elements_array[rows_double]
        # logger.info("CUANTITATIVE:\n"f"{np.column_stack([rows_double, double_cuant])}")
        
        # logger.info("ELEMENTS:\n"f"{np.column_stack([rows_double, double_element])}")
        alone_doubles_rel = np.where(double_element == double_cuant)[0]
        mixed_sc_ids = np.setdiff1d(rows_double, rows_double[alone_doubles_rel], assume_unique=True)        # Ids de filas absolutos con texto mezclado
        mixed_cells = np.nonzero(cuantiative_array[mixed_sc_ids])
        mixed_cols = mixed_cells[1]
        mixed_text_array = textual_array[mixed_sc_ids]
        
        # mixed_text = mixed_text_array[:, mixed_cells[1]]
        # logger.info("MIXED DECIMAL CELLS:\n"f"{mixed_cells}")
        # logger.info("TEXTUAL:\n"f"{np.column_stack([mixed_sc_ids, mixed_text_array])}")
        non_empty_text = np.where(mixed_text_array)[1]
        # mixed_df = df.iloc[mixed_sc_ids, mixed_cols]
        # logger.info("MIXED_DF:\n" + mixed_df.to_string(index=True))
        pot_dest_id = mixed_cols - 1
        if pot_dest_id in non_empty_text:
            dest_id = int(pot_dest_id)
        else:
            pot_dest_id = mixed_cols + 1
            if pot_dest_id in non_empty_text:
                dest_id = int(pot_dest_id)
        df_copy: pd.DataFrame = context["df_copy"]
        for row in mixed_sc_ids:
            for col in cols_idx:
                mr = int(row)
                mc = int(col)
                poly_val = df_copy.iat[mr, mc]
                dest_vals = df_copy.iat[mr, dest_id]
                # logger.info(f"{dest_vals}")
                mixed_vals = df.iat[mr, mc]
                if mc == mixed_cols:
                    mixed_vals_list = mixed_vals.split(" ")
                    for i, va in enumerate(mixed_vals_list):
                        if va.isalpha() or not validate_quant_chars(va):
                            v = va
                            
                            mixed_vals_list.remove(va)
                            poly_to_move = poly_val.pop(i)
                            # logger.info(f"TEXTO POP: '{poly_to_move}'")
                            # logger.info(f"TEXTO: '{poly_val}'")
                            if mc == dest_id:
                                continue
                            elif mc < dest_id:
                                df.iat[mr, dest_id] = str(v) + " " + str(df.iat[mr, dest_id])
                                df_copy.iat[mr, dest_id] = [poly_to_move] + dest_vals
                            else:
                                df.iat[mr, dest_id] = str(df.iat[mr, dest_id]) + " " + str(v)
                                df_copy.iat[mr, dest_id] = dest_vals + [poly_to_move]
                                
                            df.iat[mr, mc] = " ".join(mixed_vals_list).strip()
                            df_copy.iat[mr, mc] = poly_val
                            
        # logger.info("UNMIXED:\n" + df.to_string(index=True))
        return (df, df_copy)
        
    def separate_decimals(self, df: pd.DataFrame, text_idx: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        df, df_copy = self.unmix_cells(df, context)
        
        context["df_copy"] = df_copy
        
        tables_array = self.get_arrays_table(context)
        cuantiative_array, elements_array, textual_array = tables_array[0], tables_array[2], tables_array[3]
        cols_idx = np.arange(df.shape[1])
        # rows_idx = np.arange(df.shape[0])
        cols_decimal_list = [(df.columns.get_loc(name) if name in df.columns else None) for name in DEC_COLS_NAME]
        idx_decimal = np.sort(np.array(cols_decimal_list, np.uint8))
        logger.info("COMPLETE:\n" + df.to_string(index=True))
        
        rows_double = np.where(cuantiative_array==2)[0]
        double_cuant = cuantiative_array[rows_double]
        double_element = elements_array[rows_double]
        double_text = textual_array[rows_double]
        # logger.info("CUANTITATIVE:\n"f"{np.column_stack([rows_double, double_cuant])}")
        
        # logger.info("ELEMENTS:\n"f"{np.column_stack([rows_double, double_element])}")
        alone_doubles_rel = np.where(double_element == double_cuant)[0]
        logger.info("INTERSECT\n"f"{alone_doubles_rel}")        
        interfered_text = double_text[alone_doubles_rel]

        intersect_cols_text_idx = np.intersect1d(np.where(interfered_text)[1], idx_decimal)
        logger.info(f"INTERSECT COLS: {intersect_cols_text_idx}")
        
        double_cols = int(np.where(double_cuant[alone_doubles_rel])[1])
        # logger.info("DOUBLE DECIMAL COLS:\n"f"{double_cols}")
        
        # double_df = df.iloc[rows_double[alone_doubles_rel]]
        # logger.info("DOUBLES:\n" + double_df.to_string(index=True))
        
        df_copy: pd.DataFrame = context["df_copy"]                            
        dest_idx = int(text_idx[0])
        
        if double_cols == cols_idx[-1]:
            closest_idx = double_cols - 1
        else:
            close_idx1 = double_cols - 1
            close_idx2 = double_cols + 1
            # Determine which of close_idx1 or close_idx2 is present in cols_decimal_list
            if close_idx1 in cols_decimal_list:
                closest_idx = close_idx1
            elif close_idx2 in cols_decimal_list:
                closest_idx = close_idx2
            else:
                closest_idx = None  # or raise an error, depending on desired behavior
        # logger.info(f"closest: {closest_idx}")        
        for r in rows_double[alone_doubles_rel]:
            for c in intersect_cols_text_idx:
                r = int(r)
                c = int(c)
                if c == dest_idx:
                    continue
                
                val = df.iat[r, c]
                poly_val = df_copy.iat[r, c]
                dest_vals = df_copy.iat[r, dest_idx]
                
                if val == "" or val is None:
                    continue
                
                if c < dest_idx:
                    df.iat[r, dest_idx] = str(val) + " " + str(df.iat[r, dest_idx])
                    df_copy.iat[r, dest_idx] = poly_val + dest_vals
                else:
                    df.iat[r, dest_idx] = str(df.iat[r, dest_idx]) + " " + str(val)
                    df_copy.iat[r, dest_idx] = dest_vals + poly_val
                    
                df.iat[r, c] = ""
                df_copy.iat[r, c] = []
                
            for dc in cols_idx:
                dc = int(dc)
                dr = int(r)
                # logger.info(f"COLUMN: {dc}")
                dec_val = df.iat[dr, dc]
                poly_vals = df_copy.iat[dr, dc]
                dest_polys = df_copy.iat[dr, closest_idx]
                # logger.info(f"{dest_polys}")
                
                if dec_val == "" or dec_val is None:
                    continue
                
                if dc == double_cols:
                    split_decimals = dec_val.split(" ", 1)
                    # logger.info(f"vals: {vals}")
                    if closest_idx == dc:
                        continue
                    elif closest_idx < dc:
                        df.iat[dr, closest_idx] = str(split_decimals[0])
                        df.iat[dr, dc] = str(split_decimals[1])
                        
                        df_copy.iat[dr, closest_idx] = dest_polys + [poly_vals[0]]
                        df_copy.iat[dr, dc] = [poly_vals[1]] + dest_polys
                        
                    else:
                        df.iat[dr, closest_idx] = str(split_decimals[1])
                        df.iat[dr, dc] = str(split_decimals[0])
                        
                        df_copy.iat[dr, closest_idx] = dest_polys + [poly_vals[1]]
                        df_copy.iat[dr, dc] = [poly_vals[0]] + dest_polys
                    
        logger.info("SEPARATED:\n" + df.to_string(index=True))
        # logger.info("SEPARATED copy 2:\n" + df_copy.to_string(index=True))
        return (df, df_copy)