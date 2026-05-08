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
from core.utils.text_utils import validate_quant_chars, format_cuant
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
        tol: str = worker_config.get('row_relative_tolerance', "")
        self.arithmetic_tolerance = Decimal(tol) if tol else Decimal('0.15')
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

            logger.info(f"DataFrame recibido:\n{df.to_string(index=True)}")

            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            cut_polygons = self.map_polygons_ids(polygons, df_copy)
            context["cut_polygons"] = cut_polygons
            context["df_copy"] = df_copy
       
            corrected_df = self.solve(df, context)

            if corrected_df.empty:
                logger.error("SE DEVOLVIÓ DATA FRAME VACÍO")
                return False
                
            if not self.validate_vals(corrected_df, polygons):
                # logger.info("No pasó validación global")
                return False
                
            # logger.info("Validación global pasada")
            if not manager.save_structured_table(corrected_df):
                logger.error("No se pudo guardar data frame")
                return False

            logger.debug(f"Corrección matemática completada en {time.perf_counter() -start_time:.6f}'s")
            
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
        rows_idx = np.arange(df.shape[0], dtype=np.uint8)
        cols_idx = np.arange(df.shape[1], dtype=np.uint8)
        context["rows_idx"] = rows_idx
        context["cols_idx"] = cols_idx
        aritmetic_df = self.get_decimal_df(df, context)
        if aritmetic_df.equals(df):
            # logger.info("SE DEVOLVIÓ DF ORIGINAL")
            return df.iloc[0:0]
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
        
        cut_polygons: Dict[str, Any] = {}
        for poly_id in poly_ids:
            if poly_id in polygons:
                cut_polygons[poly_id] = {
                    "text": polygons[poly_id].ocr_text or "",
                    "semantic_clasification": polygons[poly_id].semantic_clasification,
                }                
        return cut_polygons
        
    def get_arrays_table(self, context: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """"Devuelve 'cuantiative_array[0], numeric_array[1], elements_array[2], textual_array[3], code_array[4]'"""
        df_copy: pd.DataFrame = context["df_copy"]
        cut_polygons: Dict[str, Dict[str, Any]] = context["cut_polygons"]
        r = df_copy.shape[0]
        h = df_copy.shape[1]
        rows_idx = context["rows_idx"]
        cols_idx = context["cols_idx"]

        matrix_array = np.zeros((r, h), np.uint8)
        cuantiative_array = matrix_array.copy()
        numeric_array = matrix_array.copy()
        elements_array = matrix_array.copy()
        textual_array = matrix_array.copy()
        code_array = matrix_array.copy()
        
        for row_id in rows_idx:
            for i in cols_idx:
                cell_poly_ids: List[str] = df_copy.iat[row_id, i] if i < h else []

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
                    
                if any(s in (1, 2) for s in sc_v):
                    textual_array[row_id, i] = sum(1 for ch in sc_v if ch in (1, 2))
                
                if any(s == 4 for s in sc_v):
                    cuantiative_array[row_id, i] = sum(1 for ch in sc_v if ch == 4)
                    
                if any(s == 5 for s in sc_v):
                    numeric_array[row_id, i] = sum(1 for ch in sc_v if ch == 5)
                
                if any(s == 3 for s in sc_v):
                    code_array[row_id, i] = sum(1 for ch in sc_v if ch == 3)
                                   
        table_arrays = np.stack([cuantiative_array, numeric_array, elements_array, textual_array, code_array], dtype=np.uint8)
        # logger.info(f"ARRAYS TABLE: \n"f"{table_arrays}")
        return table_arrays

    def find_hypotesis(self, df: pd.DataFrame, aritmetic_df: pd.DataFrame) ->pd.DataFrame:
        cols_name = list(aritmetic_df.columns)
        # logger.info(f"COLS NAME: {cols_name}")
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
                    
                    upper_tol_mtl = mtl_col + self.arithmetic_tolerance
                    lower_tol_mtl = mtl_col - self.arithmetic_tolerance
                    upper_tol_pu = pu_col + self.arithmetic_tolerance
                    lower_tol_pu = pu_col - self.arithmetic_tolerance
                    upper_tol_c = c_col + self.arithmetic_tolerance
                    lower_tol_c = c_col - self.arithmetic_tolerance
                    
                    artimetic_mtl = c_col * pu_col
                    artimetic_pu = mtl_col / c_col 
                    artimetic_c = mtl_col / pu_col
                    
                    if artimetic_mtl == mtl_col and artimetic_c == c_col and artimetic_pu == pu_col:
                        row_validated.append([c_col, pu_col, mtl_col])
                        array_votes[:, c_idx] = 1
                        array_votes[:, pu_idx] = 2
                        array_votes[:, mtl_idx] = 3
                        break
                                        
                    elif (lower_tol_mtl < artimetic_mtl < upper_tol_mtl) and \
                         (lower_tol_c < artimetic_c < upper_tol_c) and \
                         (lower_tol_pu < artimetic_pu < upper_tol_pu):
                        row_validated.append([c_col, pu_col, mtl_col])
                        array_votes[:, c_idx] = 1
                        array_votes[:, pu_idx] = 2
                        array_votes[:, mtl_idx] = 3
                        break
                    else:
                        continue
                    
                all_hypotesis.append(row_validated)
                    
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
        cols_idx = context["cols_idx"]
        arrays_table = self.get_arrays_table(context)
        matrix_decimal, matrix_quantity, elements_array, textual_array = arrays_table[0], arrays_table[1], arrays_table[2], arrays_table[3]
        full_rows_mask = np.count_nonzero(elements_array, axis=1)
        full_idx = np.where(full_rows_mask==cols_idx.size)[0]                   # índices columnas originales sin celdas vacías
        
        full_dec_com = matrix_decimal + matrix_quantity                             # Array fusionado que contiene los numericos y cuantitativos: "DECIMAL"
        full_dec = full_dec_com[full_idx]                                           # Array Decimal sin celdas faltantes
        
        full_dec_mask = np.count_nonzero(full_dec==1, axis=1)                   # Mascara booleana donde hay unicamente un elemento decimal por celda con el mismo shape que el array decimal reducido
        full_idx_dec = np.where(full_dec_mask>= 3)[0]                           # índices del array anterior (No del array original) donde hay suficientes elementos decimales
        full_dec_idx = full_idx[full_idx_dec]                                   # índices originales con filas completas y suficiente numero de decimales
        
        textual_mask = np.count_nonzero(textual_array[full_dec_idx], axis=1, keepdims=True)
        unique_mask = np.count_nonzero(elements_array[full_dec_idx], axis=1, keepdims=True)
        dec_mask = np.count_nonzero(full_dec[full_dec_idx], axis=1, keepdims=True)
        sums = textual_mask + dec_mask
        full_dec_idx_rel = np.where(unique_mask==sums)[0]                                       # índices relativos donde hay elementos decimales de sobra
        # logger.info("IDX:\n"f"{full_dec_idx_rel}, {full_dec_idx}")
        full_dec_idx = full_dec_idx[full_dec_idx_rel]                                           # índices absolutos filtrados con elementos decimales únicos
        # logger.info("\n" + df.iloc[full_dec_idx].to_string(index=True))

        n_full_rows = full_dec_idx.size
        if n_full_rows > 0:
            decimal_counts = np.count_nonzero(full_dec_com[full_dec_idx], axis=0)
            # logger.info("Counts:\n"f"{decimal_counts}")
            decimal_mask = (decimal_counts > n_full_rows // 2) | (decimal_counts == n_full_rows)
            decimal_cols = np.where(decimal_mask)[0]
            textual_cols = np.setdiff1d(cols_idx, decimal_cols, assume_unique=True)                                             # índices originales sin columnas textuales
            temp_dec = full_dec_com[full_dec_idx]
            temp_dec = temp_dec[:, decimal_cols]                                                # Array temporal decimal con los índices completos filtrados
            complete_rows_mask = np.count_nonzero(temp_dec, axis=1) == decimal_cols.size
            complete_idx = np.where(complete_rows_mask)[0]                                           # índices relativos de filas con non_zero
            full_dec_idx = full_dec_idx[complete_idx]                                           # índices absolutos de filas con non_zero
        else:
            logger.info("No hay filas completas en array, devolviendo df original")
            return df
        
        decimal_cols_str: List[str] = []
        for id in cols_idx:
            if id in textual_cols:
                col = f"col_{id:01d}"
                decimal_cols_str.append(col)
            else:
                col = f"col_{id:01d}"
                
        aritmetic_df = df.iloc[full_dec_idx, decimal_cols]
        if aritmetic_df.empty:
            logger.info("No hay filas completas, devolviendo df original")
            return df
        else:
            # logger.info("FULL DECIMAL COLS:\n" + aritmetic_df.to_string(index=True))
            return aritmetic_df
        
    def correct_df(self, df: pd.DataFrame, dec_rows_ids: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any]) -> pd.DataFrame:
        idx_map = np.sort(np.array([(df.columns.get_loc(name) if name in df.columns else None) for name in DEC_COLS_NAME], np.uint8))
        # logger.info(f"{idx_map}")
        if idx_map.size==0:
            return df.iloc[0:0]
            
        rows_ids = context["rows_idx"]
        cols_idx = context["cols_idx"]
        tables_array = self.get_arrays_table(context)
        textual_array, code_array = tables_array[3], tables_array[4]
        
        if df.shape[1] == 4 and idx_map.size == 3:
            descriptive_idx = np.setdiff1d(cols_idx, idx_map, assume_unique=True)
            
        else:
            non_dec_cols_idx = np.setdiff1d(cols_idx, idx_map, assume_unique=True)          # índices de columnas que no son decimales ni válidas para la validación
            textual_array_temp = textual_array[:, non_dec_cols_idx]
            
            textual_cols_id = np.argmax(np.sum(textual_array_temp, axis=0, dtype=np.uint8))  # índice relativo de columna descriptiva
            descriptive_idx = np.atleast_1d(non_dec_cols_idx[textual_cols_id])         # índice real de columna descriptiva principal
            orig_col_name = df.columns[int(descriptive_idx[0])]
            df.rename(columns={orig_col_name: "text_col"}, inplace=True)
            
            # leftovers_idx = np.delete(non_dec_cols_idx, textual_cols_id)                # índices de columnas restantes después de extraer la columna descriptiva principal y las decimales
            # pot_code_col = code_array[:, leftovers_idx]
            # code_rows = np.count_nonzero(pot_code_col)
            # if code_rows > (rows // 2):
            #     code_col_idx = leftovers_idx                                            # índice real de la única columna code
            #     # logger.info(f"{code_col_idx}")
            #     # logger.info("\n"f"{np.column_stack([rows_ids, elements_array[:, code_col_idx], code_array[:, code_col_idx], textual_array[:, code_col_idx]])}")
            # else:
            #     code_col_idx = None
        
        df, df_copy = self.isolate_decimals(df, descriptive_idx, context)
        context["df_copy"] = df_copy
        
        df, df_copy = self.separate_decimals(df, descriptive_idx, context)
        context["df_copy"] = df_copy
        tables_array = self.get_arrays_table(context)
        textual_array = tables_array[3]
        
        incomplete_rows_id = np.setdiff1d(rows_ids, dec_rows_ids)        # índices originales de filas a corregir/completar
        textual_array = textual_array[incomplete_rows_id]
        
        text_in_dec = np.nonzero(textual_array[:, idx_map])
        fil_cols = text_in_dec[1]
        fil_rows = text_in_dec[0]
        
        fil_cols = idx_map[fil_cols]
        fil_rows = incomplete_rows_id[fil_rows]

        dest_idx = int(descriptive_idx[0])

        for r, c in zip(fil_rows, fil_cols):
            r = int(r) 
            c = int(c)
            if c == dest_idx:
                continue

            val: str = df.iat[r, c]
            vals: List[str] = df_copy.iat[r, c]
            dest_vals = df_copy.iat[r, dest_idx]

            if val == "" or not val:
                continue

            if c < dest_idx:
                df.iat[r, dest_idx] = str(val) + " " + str(df.iat[r, dest_idx])
                df_copy.iat[r, dest_idx] = vals + dest_vals
            else:
                df.iat[r, dest_idx] = str(df.iat[r, dest_idx]) + " " + str(val)
                df_copy.iat[r, dest_idx] = dest_vals + vals

            df.iat[r, c] = ""
            df_copy.iat[r, c] = []
            
        #logger.info("CORRECT TEXT:\n" + df.to_string(index=True))
        # logger.info("COPY CORR:\n" + df_copy.to_string(index=True))
        context["df_copy"] = df_copy
        
        df = self.complete_rows(df, context, incomplete_rows_id)
        # logger.info("CORRECTED:\n" + df.to_string(index=True))
        
        df = self.simplify_rows(df, context)
        return df
        
    def complete_rows(self, df: pd.DataFrame, context: Dict[str, Any], incomplete_rows_id: np.ndarray[Any, np.dtype[np.uint8]]) -> pd.DataFrame:
        tables_array = self.get_arrays_table(context)
        matrix_decimal, matrix_quantity, elements_array = tables_array[0], tables_array[1], tables_array[2]
        full_dec = matrix_decimal + matrix_quantity
        
        full_dec = full_dec[incomplete_rows_id]                                 # Array decimal a corregir
        elements_array = elements_array[incomplete_rows_id]                     # Array Global a corregir
        
        c_idx = df.columns.get_loc("c_col") if "c_col" in df.columns else None
        pu_idx = df.columns.get_loc("pu_col") if "pu_col" in df.columns else None
        mtl_idx = df.columns.get_loc("mtl_col") if "mtl_col" in df.columns else None

        for r in incomplete_rows_id:
            raw_c: str = df.iat[r, c_idx].strip()
            raw_pu: str = df.iat[r, pu_idx].strip()
            raw_mtl: str = df.iat[r, mtl_idx].strip()

            missing_c = (raw_c == "")
            missing_pu = (raw_pu == "")
            missing_mtl = (raw_mtl == "")

            # Debe haber exactamente una vacía por fila
            if (missing_c + missing_pu + missing_mtl) != 1:
                continue
            try:
                val_c = Decimal(raw_c) if not missing_c else ZERO
                val_pu = Decimal(raw_pu) if not missing_pu else ZERO
                val_mtl = Decimal(raw_mtl) if not missing_mtl else ZERO
            except InvalidOperation as e:
                logger.warning(f"ERROR COMPLETANDO: '{e}'", exc_info=True)

            if missing_c:
                result = val_mtl / val_pu
                df.iat[r, c_idx] = str(result)
            elif missing_pu:
                result = val_mtl / val_c
                df.iat[r, pu_idx] = str(result)
            else: 
                result = val_c * val_pu
                df.iat[r, mtl_idx] = str(result)
        return df
        
    def isolate_decimals(self, df: pd.DataFrame, descriptive_idx: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aísla los valores decimales que invaden las columnas descriptivas (de texto)
        y determina a qué columna matemática corresponden utilizando comparaciones aritméticas.
        """
        tables_array = self.get_arrays_table(context)
        full_dec = tables_array[0]
        cols_idx = context["cols_idx"]
        
        descript_num = full_dec[:, descriptive_idx]          # Array decimal con las columnas textuales
        rows_to_com = np.where(descript_num)[0]              # índices absolutos de Filas con decimales en donde van textuales
        relative_idx = np.arange(rows_to_com.size)
        
        two_decimals = full_dec[rows_to_com]
        two_cols_ids = np.count_nonzero(two_decimals, axis=0) == two_decimals.shape[0]
        idx = cols_idx[two_cols_ids]
        
        invaded_df = df.iloc[rows_to_com, idx]
        # logger.info("\n"f"{invaded_df.to_string(index=True)}")
        # incomplete_rows = elements_array[rows_to_com]
        invaded_df = invaded_df.map(lambda x: Decimal(x))
        df_copy: pd.DataFrame = context["df_copy"]
        
        for r in relative_idx:
            real_idx = rows_to_com[r]
            val_n: str = invaded_df.iat[r, 0]
            val_m: str = invaded_df.iat[r, 1]
            src_a = invaded_df.columns[0]
            src_b = invaded_df.columns[1]
            poly_m = list(df_copy.at[real_idx, src_a]) 
            poly_n = list(df_copy.at[real_idx, src_b])
            val_a = max(val_m, val_n)
            val_b = min(val_m, val_n)
            quotient = (val_a / val_b)
            if val_m == val_a:
                poly_a_mtl, poly_b_pu = poly_m, poly_n
            else:
                poly_a_mtl, poly_b_pu = poly_n, poly_m
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
                else:
                    continue
            else:
                continue

        #logger.info("ISOLATED:\n" + df.to_string(index=True))
        return (df, df_copy)
        
    def unmix_cells(self, df: pd.DataFrame, context: Dict[str, Any]):
        """Separa celdas con descriptivo y cuantitativo mezclado"""
        cols_idx = context["cols_idx"]
        tables_array = self.get_arrays_table(context)
        cuantiative_array, elements_array, textual_array = tables_array[0], tables_array[2], tables_array[3]
        
        rows_double = np.where(cuantiative_array>=2)[0]
        double_cuant = cuantiative_array[rows_double]
        double_element = elements_array[rows_double]

        alone_doubles_rel = np.where(double_element == double_cuant)[0]
        mixed_sc_ids = np.setdiff1d(rows_double, rows_double[alone_doubles_rel], assume_unique=True)        # Ids de filas absolutos con texto mezclado
        
        df_copy: pd.DataFrame = context["df_copy"]
        for row in mixed_sc_ids:
            mr = int(row)
            
            row_mixed_cols = np.nonzero(cuantiative_array[mr])[0]
            if len(row_mixed_cols) == 0:
                continue
            cur_mixed_col = row_mixed_cols[0]
            
            row_non_empty_text = np.nonzero(textual_array[mr])[0]
            
            pot_dest_id = cur_mixed_col - 1
            if pot_dest_id in row_non_empty_text:
                dest_id = int(pot_dest_id)
            else:
                dest_id = int(cur_mixed_col + 1)
                
            for col in cols_idx:
                mc = int(col)
                poly_val: List[str] = df_copy.iat[mr, mc]
                dest_vals = df_copy.iat[mr, dest_id]
                # logger.info(f"{dest_vals}")
                mixed_vals: str = df.iat[mr, mc]
                if mc == cur_mixed_col:
                    mixed_vals_list = mixed_vals.split(" ")
                    for i, va in enumerate(mixed_vals_list):
                        if va.isalpha() or not validate_quant_chars(va):
                            v = va
                            mixed_vals_list.remove(va)
                            poly_to_move = poly_val.pop(i)
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
        """"Deja lista la tabla redistribuyendo los cuanitativos y textuales para completarla posteriormente"""
        df, df_copy = self.unmix_cells(df, context)
        context["df_copy"] = df_copy
        tables_array = self.get_arrays_table(context)
        cuantiative_array, elements_array, textual_array = tables_array[0], tables_array[2], tables_array[3]
        cols_idx = context["cols_idx"]
        # rows_idx = np.arange(df.shape[0])
        cols_decimal_list = [(df.columns.get_loc(name) if name in df.columns else None) for name in DEC_COLS_NAME]
        idx_decimal = np.sort(np.array(cols_decimal_list, np.uint8))
        # logger.info("COMPLETE:\n" + df.to_string(index=True))
        
        rows_double = np.where(cuantiative_array==2)[0]
        double_cuant = cuantiative_array[rows_double]
        double_element = elements_array[rows_double]
        double_text = textual_array[rows_double]
        # logger.info("CUANTITATIVE:\n"f"{np.column_stack([rows_double, double_cuant])}")
        
        # logger.info("ELEMENTS:\n"f"{np.column_stack([rows_double, double_element])}")
        alone_doubles_rel = np.where(double_element == double_cuant)[0]                         # Índices relativos de Columnas decimales donde hay texto
        # logger.info("INTERSECT\n"f"{alone_doubles_rel}")
        # interfered_text = double_text[alone_doubles_rel]

        # intersect_cols_text_idx = np.intersect1d(np.where(interfered_text)[1], idx_decimal)     # Índices absolutos de Columnas decimales donde hay texto
        # logger.info(f"INTERSECT COLS: {intersect_cols_text_idx}")
        
        # double_cols = np.where(double_cuant[alone_doubles_rel])[1]                              # índices de columnas donde hay más de un cuantitativo
        # logger.info("DOUBLE DECIMAL COLS:\n"f"{double_cols}")
        
        df_copy: pd.DataFrame = context["df_copy"]                 
        dest_idx = int(text_idx[0])
        # logger.info(f"closest: {closest_idx}")
        
        for rr in alone_doubles_rel:
            r = rows_double[rr]
            
            # Find the text columns specifically for this row
            row_text_cols = np.where(double_text[rr])[0]
            intersect_cols_text_idx = np.intersect1d(row_text_cols, idx_decimal)
            
            # Find the double decimal column for this specific row
            row_double_cols = np.where(double_cuant[rr])[0]
            if len(row_double_cols) == 0:
                continue
            cur_double_col = row_double_cols[0]
            
            for c in intersect_cols_text_idx:
                r = int(r)
                c = int(c)
                if c == dest_idx:
                    continue

                val: str = df.iat[r, c]
                poly_val: List[str] = df_copy.iat[r, c]
                dest_vals = df_copy.iat[r, dest_idx]
                
                if val == "" or not val:
                    continue
                
                if cur_double_col == cols_idx[-1]:
                    closest_idx = cur_double_col - 1
                else:
                    close_idx1 = cur_double_col - 1
                    close_idx2 = cur_double_col + 1
                    # Determine which of close_idx1 or close_idx2 is present in cols_decimal_list
                    if close_idx1 in cols_decimal_list:
                        closest_idx = close_idx1
                    else: 
                        # close_idx2 in cols_decimal_list:
                        closest_idx = close_idx2
                        
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
                dec_val: str = df.iat[dr, dc]
                poly_vals: List[str] = df_copy.iat[dr, dc]
                dest_polys = df_copy.iat[dr, closest_idx]
                # logger.info(f"{dest_polys}")
                
                if dec_val == "" or not dec_val:
                    continue
                
                if dc == cur_double_col:
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

        #logger.info("SEPARATED:\n" + df.to_string(index=True))
        return (df, df_copy)
    
    def simplify_rows(self, df: pd.DataFrame, context: Dict[str, Any]):
        arrays_table = self.get_arrays_table(context)
        elements_array = arrays_table[2]
        unique_cells = np.count_nonzero(elements_array, axis=1) == 1
        empty_rows_idx = np.where(unique_cells)[0]
        
        for r in empty_rows_idx:
            c_idx = np.nonzero(elements_array[r])[0]
            if len(c_idx) == 0:
                continue
            c = int(c_idx[0])
            
            r_target = int(r) - 1
            if r_target < 0:
                continue
                
            val: str = df.iat[r, c]
            
            if val != "" and val:
                target_val: str = df.iat[r_target, c]
                if target_val != "" and target_val:
                    df.iat[r_target, c] = str(target_val).strip() + str(val).strip()
                else:
                    df.iat[r_target, c] = str(val).strip()
                    
                df.iat[r, c] = ""
        
        df = df.drop(empty_rows_idx, axis=0)
        df = df.reset_index(drop=True)
        # logger.info("REINDEX:\n" + df.to_string(index=True))
        return df
        
    def validate_vals(self, corrected_df: pd.DataFrame, polygons: Dict[str, Polygons]) -> bool:
        mtl_col = corrected_df["mtl_col"]
        c_col = corrected_df["c_col"]
        
        mtl_col_dec = mtl_col.map(lambda x: Decimal(x.strip()))
        c_col_dec = c_col.map(lambda x: Decimal(x.strip()))
        
        total = sum(mtl_col_dec)
        total_prod = sum(c_col_dec)

        logger.debug(f"{total}, {total_prod}")

        monetary_vals: List[Decimal] = []
        for _, poly_data in polygons.items():
            
            kf = poly_data.key_field or None
            if kf is None:
                continue
            
            text = poly_data.ocr_text or ""
            if 1 in kf and validate_quant_chars(text):
                formated_total = format_cuant(text)
                # logger.info(f"{poly_id}: {text}, {formated_total}")
                total_dec = Decimal(formated_total)
                monetary_vals.append(total_dec)
                continue
                
            if 2 in kf and validate_quant_chars(text):
                formated_products = format_cuant(text)
                # logger.info(f"{poly_id}: {text}, {formated_products}")
                total_produc = Decimal(formated_products)
                monetary_vals.append(total_produc)
            else:
                continue
        
        logger.debug(f"{monetary_vals}")
        if not monetary_vals:
            return True
            
        elif len(monetary_vals) == 2:
            if monetary_vals[0] == total and monetary_vals[1] == total_prod:
                return True
            elif monetary_vals[1] == total and monetary_vals[0] == total_prod:
                return True
            else:
                return False  
                
        elif len(monetary_vals) == 1:
            if monetary_vals[0] == total or monetary_vals[0] == total_prod:
                return True
        else:
            return False