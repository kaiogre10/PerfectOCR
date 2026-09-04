# PerfectOCR/core/workers/vectorial_transformation/math_max.py
import pandas as pd # type: ignore
import logging
import numpy as np
import time
from itertools import permutations
from typing import Dict, Any, List, Tuple
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from domain.abstract_worker import VectorizationAbstractWorker
from domain.data_formatter import DataFormatter
from utils.compiled_utils import validate_quant_chars
from utils.math_utils import check_full_df
from core.assets.assets import ONE_DEC, ZERO_DEC, ROW_TOL, SC_RANGE
from services.output_service import save_debug_table
from domain.class_models import SemantiClass, DataKeys, DataMathDict

_row_tol = ROW_TOL
_one = ONE_DEC
_zero = ZERO_DEC
_total_sc = SC_RANGE[1]

logger = logging.getLogger(__name__)

class MatrixSolver(VectorizationAbstractWorker):
    """
    Resuelve inconsistencias matemáticas en una tabla estructurada usando
    clasificación semántica de polígonos, aritmética Decimal y votación global.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get('math_max', {})
        self.dec_cols_name = worker_config["dec_cols_name"]
        self.output = config.get("math_max_corrected")

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter):
        start_time = time.perf_counter()
        try:
            if not manager.workflow:
                return False
            df = manager.workflow.table_data.df_table if manager.workflow.table_data is not None else pd.DataFrame(dtype=str)
            if df is None or df.empty:
                logger.error("No hay table_matrix en contexto para procesar")
                return False

            corrected_df = self.solve(df, context)
            del context
            if corrected_df.empty:
                logger.error("SE DEVOLVIÓ DATA FRAME VACÍO")
                return False

            if manager.save_final_output(corrected_df, {}):
                logger.info(f"DataFrame RECONSTRUIDO en {time.perf_counter() - start_time:.6f}'s:\n{corrected_df.to_string(index=False)}")
                if self.output:
                    file_name = manager.workflow.metadata.image_name if manager.workflow.metadata else ""
                    save_debug_table(corrected_df, file_name, None, None)
                return True

        except Exception as e:
            logger.error(f"Error en MatrixSolver.vectorize: '{e}'", exc_info=True)
        return False

    def solve(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        df, context = self.get_decimal_df(df, context)
        if not context or df.empty:
            logger.error(f"DATA FRAME CON DATOS INSUFICIENTES O MUCHO RUIDO")
            return pd.DataFrame()
        
        arithmetical_cols = context[DataMathDict.ARITH_COLS_IDS.value]
        arithmetical_rows = context[DataMathDict.DEC_ROWS_IDS.value]

        if context[DataMathDict.COMPLETE.value]:
            df, context = self.find_hypotesis(df, context)
            dec_cols = context[DataMathDict.DEC_COLS.value]
            text_col_temp = context[DataMathDict.TEXT_COL_TEMP.value]
            cols_idx = context[DataMathDict.COLS_IDX.value]
            left_cols = np.setdiff1d(cols_idx, dec_cols)

            if text_col_temp in left_cols:
                text_col_idx = text_col_temp
            elif left_cols.size == 1:
                text_col_idx = left_cols[0]
            else:
                text_col_idx = context[DataMathDict.TEXT_COL.value]

            cols_list: List[str] = list(df.columns)
            text_col: str = cols_list[int(text_col_idx[0])]
            df.rename(columns={text_col: DataKeys.producto_norm.value}, inplace = True)
            if check_full_df(df):
                logger.debug("DF PERFECTO")
                return df
        else:
            if df.shape == (arithmetical_rows.size, arithmetical_cols.size):
                logger.debug(f"DF INCOMPLETO")
                df_art, context_art = self.solve_incomplete(df, context)
                if df_art.empty:
                    del context_art
                    del context
                    return pd.DataFrame()
                else:
                    df = df_art
                    context = context_art

        df, context = self.find_hypotesis(df, context)
        
        del context[DataMathDict.ARITH_COLS_IDS.value]
        del context[DataMathDict.TEXT_COL_TEMP.value]

        if df.empty:
            logger.warning("SIN HIPOTESIS VÁLIDA")
            return pd.DataFrame()

        # logger.info("RENAMED:\n" + df.to_string(index=True))
        df = self.correct_df(df, context)
        if not check_full_df(df):
            return pd.DataFrame()
        else:
            return df

    def get_arrays_table(self, context: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """"Devuelve los arrays 'total[0], textual[1], umd[2], code[3], cuantitative[4], numeric[5]'"""
        df_copy = context[DataMathDict.DF_COPY.value]
        cut_polygons: Dict[str, Dict[str, Any]] = context[DataMathDict.CUT_POLYGONS.value]
        r = df_copy.shape[0]
        h = df_copy.shape[1]

        matrix_array = np.zeros(shape=(_total_sc, r, h), dtype=np.uint8)
        for row_id in range(r):
            for i in range(h):
                cell_poly_ids: List[str] = df_copy.iat[row_id, i] if i < h else []

                if not cell_poly_ids:
                    continue

                sc_v: List[int] = []
                has_text = False
                for poly_id in cell_poly_ids:
                    poly_data = cut_polygons.get(poly_id)
                    if not poly_data:
                        continue
                        
                    sc_v.extend(poly_data[DataMathDict.SEMANTIC_CLASIFICATION.value])
                    if poly_data.get(DataMathDict.TEXT.value):
                        has_text = True

                if not sc_v or not has_text:
                    continue
                
                matrix_array[0, row_id, i] = len(sc_v)
                
                sc = np.asarray(sc_v, dtype=np.uint8)
                counts = np.bincount(sc, minlength=_total_sc)
                matrix_array[1:, row_id, i] = counts[1:]

        # logger.info(f"ARRAYS TABLE: \n"f"{matrix_array}")
        return matrix_array

    def find_hypotesis(self, df: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Determina qué columnas cumplen roles C, PU y MTL por votación global.
        Evalúa permutaciones de tríos por fila de manera aritmética,
        valida igualdades exactas o dentro de tolerancia relativa configurada y
        acumula votos por posición de columna para etiquetar los roles finales.
        """
        arithmetical_cols = context[DataMathDict.ARITH_COLS_IDS.value]
        arithmetical_rows = context[DataMathDict.DEC_ROWS_IDS.value]

        sliced_df: pd.DataFrame = df.iloc[arithmetical_rows, arithmetical_cols]
        try:
            aritmetic_df = sliced_df.map(lambda x: Decimal(x)) # type: ignore
        except InvalidOperation as e:
            logger.debug(f"ERROR CONVIRTIENDO VALORES DEL DF: '{e}'", exc_info=True)
            return (pd.DataFrame(), {})
        try:
            n_arith_cols = aritmetic_df.shape[1]
            array_votes = np.zeros(aritmetic_df.shape, dtype=np.int8, order='F')

            for _, row_values in enumerate(aritmetic_df.values):
                for c_idx, pu_idx, mtl_idx in permutations(range(n_arith_cols), 3):
                    c_col = row_values[c_idx]
                    pu_col = row_values[pu_idx]
                    mtl_col = row_values[mtl_idx]
                    
                    if mtl_col < pu_col:
                        continue
                    if c_col > mtl_col:
                        continue

                    artimetic_mtl = c_col * pu_col
                    artimetic_pu = mtl_col / c_col
                    artimetic_c = mtl_col / pu_col

                    if artimetic_mtl == mtl_col and artimetic_c == c_col and artimetic_pu == pu_col:
                        array_votes[:, c_idx] = 1
                        array_votes[:, pu_idx] = 2
                        array_votes[:, mtl_idx] = 3
                        break
                    
                    elif (abs(mtl_col - artimetic_mtl) < _row_tol) and (abs(c_col - artimetic_c) < _row_tol) and (abs(pu_col - artimetic_pu) < _row_tol):
                        array_votes[:, c_idx] = 1
                        array_votes[:, pu_idx] = 2
                        array_votes[:, mtl_idx] = 3
                        break
                    else:
                        continue

        except TypeError as e:
            logger.warning(f"ERROR PERMUTANDO: '{e}'", exc_info=True)
            return (pd.DataFrame(), {})
        
        c_column, pu_column, mtl_column = [np.argmax(np.count_nonzero(array_votes==i, axis=0)) for i in (1, 2, 3)]

        cols_name: List[str] = list(aritmetic_df.columns)
        for i, col in enumerate(cols_name):
            if i == c_column:
                df.rename(columns={col: DataKeys.cantidad_art.value}, inplace=True)
            elif i == pu_column:
                df.rename(columns={col: DataKeys.precio_unitario.value}, inplace=True)
            elif i == mtl_column:
                df.rename(columns={col: DataKeys.costo_tran.value}, inplace=True)
            else:
                continue

        cols_idx = context[DataMathDict.COLS_IDX.value]
        num_cols = cols_idx.size
        if num_cols == 4:
            text_col = context[DataMathDict.TEXT_COL_TEMP.value]
            col_name_to_rename = df.columns[text_col][0]
            # text_cols = df.columns.get_loc(col_name_to_rename)

            df.rename(columns={col_name_to_rename: DataKeys.producto_norm.value}, inplace=True)
            context[DataMathDict.TEXT_COL.value] = text_col
            dec_cols = arithmetical_cols
            
        elif 4 < num_cols:
            dec_cols = np.sort(np.asarray([(df.columns.get_loc(name) if name in df.columns else None) for name in self.dec_cols_name], np.uint8)) 
            context[DataMathDict.TEXT_COL.value] = np.empty(0, dtype=np.uint8)
            
        else:
            logger.error(f"NO SE ENCONTRARON TODAS LAS COLUMNAS DECIMALES: {[df.columns.get_loc(name) for name in df.columns]}")
            return pd.DataFrame(), {}
        
        context[DataMathDict.DEC_COLS.value] = dec_cols
        df_copy: pd.DataFrame = context[DataMathDict.DF_COPY.value]
        df_copy.columns = df.columns
        context[DataMathDict.DF_COPY.value] = df_copy

        return (df, context)

    def get_decimal_df(self, df: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Selecciona filas/columnas con densidad decimal suficiente para inferencia."""
        cols_idx = context[DataMathDict.COLS_IDX.value]
        arrays_table = self.get_arrays_table(context)
        elements_array, textual_array, umd_array, matrix_decimal, matrix_quantity = arrays_table[0], arrays_table[SemantiClass.DESCRIPTIVE], arrays_table[SemantiClass.UMD], arrays_table[SemantiClass.QUANTITATIVE], arrays_table[SemantiClass.NUMERIC]
        full_rows_mask = np.count_nonzero(elements_array, axis=1)
        full_idx = np.where(full_rows_mask==cols_idx.size)[0]               # índices columnas originales sin celdas vacías

        full_dec_com = matrix_decimal + matrix_quantity                     # Array fusionado que contiene los numericos y "DECIMALES"
        #logger.info("DEC:\n"f"{full_dec_com}")
        full_dec = full_dec_com[full_idx]                                   # Array Decimal sin celdas faltantes

        if full_dec.size < 1 and full_idx.size < 1:
            min_decs = np.count_nonzero(full_dec_com, axis=1) > 1
            if any(min_decs):
                context[DataMathDict.ARITH_COLS_IDS.value] = cols_idx
                context[DataMathDict.DEC_ROWS_IDS.value] = context[DataMathDict.ROWS_IDX.value]
                context[DataMathDict.TEXT_COL_TEMP.value] = []
                return df, context
            else:
                return (pd.DataFrame(), {})

        sums = np.sum(full_dec==1, axis=1, dtype=np.uint8)
        max_decs = np.max(sums)
        #logger.info(f"SUMS/MAX: {sums} / {max_decs}")

        full_dec_mask = np.count_nonzero(full_dec==1, axis=1) == max_decs   # Mascara booleana donde hay unicamente un elemento decimal por celda con el mismo shape que el array decimal reducido
        full_idx_dec = np.where(full_dec_mask)[0]                           # índices del array anterior (No del array original) donde hay suficientes elementos decimales
        full_dec_idx = full_idx[full_idx_dec]                               # índices originales con filas completas y suficiente numero de decimales

        masks = np.count_nonzero(arrays_table[:4, full_dec_idx], axis=2, keepdims=True)
        
        dec_mask = np.count_nonzero(full_dec_com[full_dec_idx], axis=1, keepdims=True)
        unique_mask, textual_mask, umd_mask, code_mask = masks
        
        sums = textual_mask + dec_mask + code_mask + umd_mask
        full_dec_idx_rel = np.where(unique_mask==sums)[0]               # índices relativos donde hay elementos decimales de sobra
        if full_dec_idx_rel.size < 1:
            full_dec_idx = full_dec_idx
        else:
            ##logger.info("IDX:\n"f"{full_dec_idx_rel}, {full_dec_idx}")
            full_dec_idx = full_dec_idx[full_dec_idx_rel]                   # índices absolutos filtrados con elementos decimales únicos

        # logger.debug(f"{full_dec_idx}")
        #logger.info("ROWS DEC DF:\n" + df.iloc[full_dec_idx].to_string(index=True))

        n_full_rows = full_dec_idx.size
        if 0 < n_full_rows:
            textual_array = umd_array + textual_array
            textual_counts = np.sum(textual_array[full_dec_idx], axis=0, dtype=np.uint8)
            textual_cols = np.argmax(textual_counts, keepdims=True)
            temp_dec = full_dec_com[full_dec_idx]

            if cols_idx.size == 4:
                decimal_cols = np.setdiff1d(cols_idx, textual_cols, assume_unique=True)
                # #  ##logger.info("COLS DEC DF:\n" + df.iloc[full_dec_idx, decimal_cols].to_string(index=True))
            else:
                decimal_counts = np.count_nonzero(temp_dec, axis=0)
                decimal_mask = (decimal_counts > n_full_rows // 2) | (decimal_counts == n_full_rows)
                decimal_idx = np.where(decimal_mask)[0]
                decimal_cols = np.setdiff1d(decimal_idx, textual_cols, assume_unique=True)          # índices originales sin columnas textuales

            temp_dec = temp_dec[:, decimal_cols]                                                # Array temporal decimal con los índices completos filtrados
            complete_rows_mask = np.count_nonzero(temp_dec, axis=1) == decimal_cols.size
            complete_idx = np.where(complete_rows_mask)[0]                                      # índices relativos de filas con non_zero
            full_dec_idx = full_dec_idx[complete_idx]                                           # índices absolutos de filas con non_zero
        else:
            context[DataMathDict.ARITH_COLS_IDS.value] = np.empty(0)
            context[DataMathDict.DEC_ROWS_IDS.value] = context[DataMathDict.ROWS_IDX.value]
            logger.debug("No hay filas con suficientes datos, devolviendo df original")
            return (df, context)

        invalid_indexes = (full_dec_idx.size < 1) | (decimal_cols.size < 1)
        if invalid_indexes:
            logger.debug("No hay filas completas para aritmetic, devolviendo df original")
            context[DataMathDict.ARITH_COLS_IDS.value] = cols_idx
            context[DataMathDict.DEC_ROWS_IDS.value] = context[DataMathDict.ROWS_IDX.value]
            context[DataMathDict.TEXT_COL_TEMP.value] = []
            return (df, context)
        else:
            #logger.info(f"ROWS FULL: {full_dec_idx} , {full_dec_idx.size} | SHAPE DF: {df.shape[0]}")
            #logger.info("DECIMALDF:\n" + df.iloc[full_dec_idx, decimal_cols].to_string(index=True))
            context[DataMathDict.TEXT_COL_TEMP.value] = textual_cols
            context[DataMathDict.ARITH_COLS_IDS.value] = decimal_cols
            context[DataMathDict.DEC_ROWS_IDS.value] = full_dec_idx
            context[DataMathDict.COMPLETE.value] = (full_dec_idx.size == df.shape[0])
            return (df, context)

    def correct_df(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """
        Corrige la tabla tras detectar columnas decimales y descriptivas.
        Estima la columna descriptiva principal, aísla contenido decimal/textual
        mezclado, redistribuye valores entre celdas, completa filas incompletas
        con reglas aritméticas y elimina filas residuales de ruido.
        """
        idx_map = context[DataMathDict.DEC_COLS.value]                       # índices de columnas decimales
        dec_rows_ids = context[DataMathDict.DEC_ROWS_IDS.value]              # Filas completas con suficientes elementos para operar
        rows_ids = context[DataMathDict.ROWS_IDX.value]
        cols_idx = context[DataMathDict.COLS_IDX.value]
        desc_idx = context[DataMathDict.TEXT_COL.value]

        tables_array = self.get_arrays_table(context)
        textual_array = tables_array[SemantiClass.DESCRIPTIVE] + tables_array[SemantiClass.UMD]

        if desc_idx.size < 1 or not desc_idx:
            non_dec_cols_idx = np.setdiff1d(cols_idx, idx_map, assume_unique=True)          # índices de columnas que no son decimales ni válidas para la validación
            textual_array_temp = textual_array[:, non_dec_cols_idx]

            textual_cols_id = np.argmax(np.sum(textual_array_temp, axis=0, dtype=np.uint8))  # índice relativo de columna descriptiva
            descriptive_idx = np.atleast_1d(non_dec_cols_idx[textual_cols_id])         # índice real de columna descriptiva principal

            orig_col_name = df.columns[int(descriptive_idx[0])]
            df.rename(columns={orig_col_name: DataKeys.producto_norm.value}, inplace=True)
        else:
            descriptive_idx = desc_idx

            # leftovers_idx = np.delete(non_dec_cols_idx, textual_cols_id)                # índices de columnas restantes después de extraer la columna descriptiva principal y las decimales
            # pot_code_col = code_array[:, leftovers_idx]
            # code_rows = np.count_nonzero(pot_code_col)
            # if code_rows > (rows // 2):
            #     code_col_idx = leftovers_idx                                            # índice real de la única columna code
            #     # logger.debug(f"{code_col_idx}")
            #     # logger.debug("\n"f"{np.column_stack([rows_ids, elements_array[:, code_col_idx], code_array[:, code_col_idx], textual_array[:, code_col_idx]])}")
            # else:
            #     code_col_idx = None
        context[DataMathDict.TEXT_COL] = descriptive_idx

        # Aislar decimales para poder operar
        df, df_copy = self.isolate_decimals(df, context)
        context[DataMathDict.DF_COPY.value] = df_copy

        # Separar decimales malagrupados
        df, df_copy = self.separate_decimals(df, context)
        context[DataMathDict.DF_COPY.value] = df_copy
        tables_array = self.get_arrays_table(context)
        textual_array = tables_array[SemantiClass.DESCRIPTIVE] + tables_array[SemantiClass.UMD]

        incomplete_rows_id = np.asarray(np.setdiff1d(rows_ids, dec_rows_ids), dtype=np.uint8)        # índices originales de filas a corregir/completar
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

            val: str = str(df.iat[r, c])
            vals: List[str] = df_copy.iat[r, c]
            dest_vals = df_copy.iat[r, dest_idx]

            if val == "" or not val:
                continue

            if c < dest_idx:
                df.iat[r, dest_idx] = str(val + " " + str(df.iat[r, dest_idx]))
                df_copy.iat[r, dest_idx] = vals + dest_vals
            else:
                df.iat[r, dest_idx] = str((df.iat[r, dest_idx]) + " " + val)
                df_copy.iat[r, dest_idx] = dest_vals + vals

            df.iat[r, c] = ""
            df_copy.iat[r, c] = []

        # logger.info("CORRECT TEXT:\n" + df.to_string(index=True))

        context[DataMathDict.DF_COPY.value] = df_copy

        df = self.complete_rows(df, incomplete_rows_id)
        # logger.info("CORRECTED:\n" + df.to_string(index=True))

        df = self.simplify_rows(df, context)
        return df

    def complete_rows(self, df: pd.DataFrame, incomplete_rows_id: np.ndarray[Any, np.dtype[np.uint8]]) -> pd.DataFrame:
        """
        Completa una celda decimal faltante por fila usando identidad C*PU=MTL.
        Solo actúa en filas donde exactamente una de las tres columnas
        matemáticas está vacía; calcula el valor faltante con división o
        multiplicación según corresponda.
        """
        c_idx = df.columns.get_loc(DataKeys.cantidad_art.value) if DataKeys.cantidad_art.value in df.columns else None # type: ignore
        pu_idx = df.columns.get_loc(DataKeys.precio_unitario.value) if DataKeys.precio_unitario.value in df.columns else None # type: ignore
        mtl_idx = df.columns.get_loc(DataKeys.costo_tran.value) if DataKeys.costo_tran.value in df.columns else None # type: ignore

        for _, r in enumerate(incomplete_rows_id):
            raw_c: str = str(df.iat[r, c_idx]) # type: ignore
            raw_pu: str = str(df.iat[r, pu_idx]) # type: ignore
            raw_mtl: str = str(df.iat[r, mtl_idx]) # type: ignore

            missing_c = (raw_c == "")
            missing_pu = (raw_pu == "")
            missing_mtl = (raw_mtl == "")

            # Debe haber exactamente una vacía por fila
            if (missing_c + missing_pu + missing_mtl) != 1:
                continue
            try:
                val_c = Decimal(raw_c) if not missing_c else _zero
                val_pu = Decimal(raw_pu) if not missing_pu else _zero
                val_mtl = Decimal(raw_mtl) if not missing_mtl else _zero
            except InvalidOperation as e:
                logger.warning(f"ERROR COMPLETANDO: '{e}'", exc_info=True)

            if missing_c:
                result = val_mtl / val_pu
                df.iat[r, c_idx] = str(result.to_integral_value(rounding=ROUND_HALF_UP))
            elif missing_pu:
                result = val_mtl / val_c
                df.iat[r, pu_idx] = str(result)
            else:
                result = val_c * val_pu
                df.iat[r, mtl_idx] = str(result)

        #  ##logger.info("COMPLETED:\n" + df.to_string(index=True))
        return df

    def isolate_decimals(self, df: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aísla los valores decimales que invaden las columnas descriptivas (y viceversa)
        y determina a qué columna matemática corresponden utilizando comparaciones aritméticas.
        Esta rutina detecta filas donde la columna textual contiene datos
        decimales, identifica pares numéricos invadidos, y reubica los valores
        a `costo_tran` y `precio_unitario` cuando la relación entre ambos es
        consistente con un patrón entero válido.
        """
        tables_array = self.get_arrays_table(context)
        full_dec = tables_array[SemantiClass.QUANTITATIVE]
        descriptive_idx = context[DataMathDict.TEXT_COL.value]
        cols_idx = context[DataMathDict.COLS_IDX.value]

        descript_num = full_dec[:, descriptive_idx]          # Array decimal con las columnas textuales
        rows_to_com = np.where(descript_num)[0]              # índices absolutos de Filas con decimales en donde van textuales
        relative_idx = np.arange(rows_to_com.size)

        two_decimals = full_dec[rows_to_com]
        two_cols_ids = np.count_nonzero(two_decimals, axis=0) == two_decimals.shape[0]
        idx = cols_idx[two_cols_ids]

        invaded_df: pd.DataFrame = df.iloc[rows_to_com, idx]
        invaded_df = invaded_df.map(lambda x: Decimal(x)) # type: ignore
        df_copy: pd.DataFrame = context[DataMathDict.DF_COPY.value]

        #  ##logger.info("INVDED:\n" + df.iloc[:, idx].to_string(index=True))

        for _, r in enumerate(relative_idx):
            real_idx = rows_to_com[r]
            val_n = invaded_df.iat[r, 0]
            val_m = invaded_df.iat[r, 1]
            src_a = invaded_df.columns[0]
            src_b = invaded_df.columns[1]
            poly_m: List[str] = list(df_copy.at[real_idx, src_a])
            poly_n: List[str]  = list(df_copy.at[real_idx, src_b])
            val_a = max(val_m, val_n)
            val_b = min(val_m, val_n)
            quotient = (val_a / val_b)
            if val_m == val_a:
                poly_a_mtl, poly_b_pu = poly_m, poly_n
            else:
                poly_a_mtl, poly_b_pu = poly_n, poly_m
            if (val_a / val_b) % _one == _zero:
                if val_a > quotient and val_b >= quotient and val_a != quotient:
                    df.at[real_idx, src_a] = ""
                    df.at[real_idx, src_b] = ""

                    df_copy.at[real_idx, src_a] = []
                    df_copy.at[real_idx, src_b] = []

                    df.at[real_idx, DataKeys.costo_tran.value] = str(val_a)
                    df.at[real_idx, DataKeys.precio_unitario.value] = str(val_b)

                    df_copy.at[real_idx, DataKeys.costo_tran.value] = poly_a_mtl
                    df_copy.at[real_idx, DataKeys.precio_unitario.value] = poly_b_pu
                else:
                    continue
            else:
                continue

        #logger.info("ISOLATED:\n" + df.to_string(index=True))
        return (df, df_copy)

    def clean_cells(self, df: pd.DataFrame, context: Dict[str, Any]):
        rows_idx = context[DataMathDict.ROWS_IDX.value]
        dec_rows_ids = context[DataMathDict.DEC_ROWS_IDS.value]
        not_aritmetic = np.setdiff1d(rows_idx, dec_rows_ids, assume_unique=True)

        # #logger.info(f"NOT ARITMETHIC: {not_aritmetic}, {rows_idx}")

        tables_array = self.get_arrays_table(context)
        elements_array, cuantitative_array = tables_array[0], tables_array[SemantiClass.QUANTITATIVE]

        shadow_dec = np.unique(cuantitative_array[dec_rows_ids], axis=0)[0]

        #logger.info(f"SHADOW: {shadow_dec}")

        fake_dec = np.where(np.all(shadow_dec == cuantitative_array[not_aritmetic], axis=1))[0]
        fake_dec = not_aritmetic[fake_dec]                                                          # índices donde hay match con el shadow

        # logger.info(f"FAKE_DEC: {fake_dec}")

        double_idx = np.setdiff1d(fake_dec, not_aritmetic, assume_unique=True)

        # logger.info(f"DOUBLE: {double_idx}")

        fake_mask = (cuantitative_array[fake_dec] != elements_array[fake_dec]) & (cuantitative_array[fake_dec] > 0) & (elements_array[fake_dec] > 0)
        fake_unique_idx = np.argwhere(fake_mask)
        # logger.info(f"fake_unique_idx: {fake_mask}")

        double_mask = (cuantitative_array[double_idx] != elements_array[double_idx]) & (cuantitative_array[double_idx] > 0) & (elements_array[double_idx] > 0)
        double_idx_final = np.argwhere(double_mask)
        # logger.info(f"double_idx_final: {double_idx_final}")

        double_idx_abs = double_idx[double_idx_final[:, 0]]
        fake_unique_abs = fake_dec[fake_unique_idx[:, 0]]

        fake_size = fake_unique_idx.size
        double_size = double_idx_final.size
        if fake_size < 1 and double_size < 1:
            # logger.info("SIN CELDAS DOBLES O MEZCLADAS")
            df_copy = context[DataMathDict.DF_COPY.value]
            return (df, df_copy)

        if fake_size < 1 and double_size > 0:
            concat = double_idx_abs
            fake_arr = double_idx_final

        elif fake_size > 0 and double_size < 1:
            concat = fake_unique_abs
            fake_arr = fake_unique_idx
        else:
            concat = np.concatenate((fake_unique_abs, double_idx_abs), axis=0)
            fake_arr = np.concatenate((fake_unique_idx, double_idx_final), axis=0)

        fake_array = np.column_stack([concat, fake_arr[:, 1]])

        # #logger.info("FAKE ARRAY:\n"f"{fake_array}")

        fake_rows = fake_array[:, 0]
        fake_cols = fake_array[:, 1]
        # #logger.info("TO CLEAN:\n" + df.iloc[fake_rows].to_string(index=True))

        df_copy: pd.DataFrame = context[DataMathDict.DF_COPY.value]
        for r, c in zip(fake_rows, fake_cols):
            r = int(r)
            c = int(c)
            values: str = str(df.iat[r, c])
            split_values: List[str] = values.split(" ")
            vals: List[str] = df_copy.iat[r, c]
            # if split_values[0].isalpha() or split_values[0].isdecimal() or not validate_quant_chars(split_values[0]):
            #     split_values.remove(split_values[0])
            #     vals.remove(vals[0])

            if split_values[-1].isalpha() or split_values[-1].isdecimal() or not validate_quant_chars(split_values[-1]):
                split_values.remove(split_values[-1])
                vals.remove(vals[-1])

            df.iat[r, c] = " ".join(split_values)
            df_copy.iat[r, c] = vals

        #logger.info("CLEAN:\n" + df.to_string(index=True))
        return df, df_copy

    def unmix_cells(self, df: pd.DataFrame, context: Dict[str, Any]):
        """
        Separa tokens textuales y cuantitativos cuando vienen mezclados.
        Detecta celdas con patrones semánticos ambiguos, divide tokens por espacios, valida fragmentos cuantitativos y mueve texto residual hacia
        la celda descriptiva más probable preservando mapeo de polígonos
        """
        cols_idx = context[DataMathDict.COLS_IDX.value]
        df, df_copy = self.clean_cells(df, context)
        context[DataMathDict.DF_COPY.value] = df_copy

        tables_array = self.get_arrays_table(context)
        elements_array, cuantiative_array = tables_array[0], tables_array[SemantiClass.QUANTITATIVE]
        textual_array = tables_array[SemantiClass.DESCRIPTIVE] + tables_array[SemantiClass.UMD]
        diff_array = elements_array - cuantiative_array

        # logger.debug("CUANT\n"f"{np.column_stack([rows_idx, cuantiative_array])}")
        # logger.debug("DIFF\n"f"{np.column_stack([rows_idx, diff_array])}")

        dec_cells = np.transpose(np.nonzero(cuantiative_array))
        # logger.debug("\n"f"{dec_cells}")
        mask = diff_array[dec_cells[:, 0], dec_cells[:, 1]] > 0
        dec_cells = dec_cells[mask]
        # logger.debug(f"{dec_cells[:, 0]}")

        rows_double = np.where(cuantiative_array>=2)[0]
        double_cuant = cuantiative_array[rows_double]
        double_element = elements_array[rows_double]

        alone_doubles_rel = np.where(double_element == double_cuant)[0]
        rows_double = rows_double[alone_doubles_rel]
        mixed_sc_ids = np.concatenate((rows_double, dec_cells[:, 0]))
        # logger.debug("\n"f"{rows_double}, {alone_doubles_rel}, {mixed_sc_ids}")
        #  ##logger.info("DOU:\n" + df.iloc[mixed_sc_ids].to_string(index=True))
        for _, mr in enumerate(mixed_sc_ids):

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

                mixed_vals: str = str(df.iat[mr, mc])
                if mc == cur_mixed_col:
                    mixed_vals_list = mixed_vals.split(" ")
                    new_text: List[str] = []
                    new_polys_ids: List[str] = []

                    current_text: List[str] = []
                    current_polys: List[str] = []

                    for i, v in enumerate(mixed_vals_list):
                        poly_id: str = poly_val[i] if i < len(poly_val) else ""

                        if v.isalpha() or not validate_quant_chars(v):
                            new_text.append(v)
                            if poly_id is not None:
                                new_polys_ids.append(poly_id)

                            if mc == dest_id:
                                continue
                            elif mc < dest_id:
                                df.iat[mr, dest_id] = (" ".join(new_text) + " " + str(df.iat[mr, dest_id]))
                                df_copy.iat[mr, dest_id] = dest_vals + new_polys_ids
                                continue
                            else:
                                df.iat[mr, dest_id] = (str(df.iat[mr, dest_id]) + " " + " ".join(new_text))
                                df_copy.iat[mr, dest_id] = new_polys_ids + dest_vals
                        else:
                            current_text.append(v)
                            if poly_id is not None:
                                current_polys.append(poly_id)

                            df.iat[mr, mc] = " ".join(current_text)
                            df_copy.iat[mr, mc] = current_polys

        #logger.info("UNMIXED:\n" + df.to_string(index=True))
        return (df, df_copy)

    def separate_decimals(self, df: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Redistribuye cuantitativos dobles y texto para preparar completado final.
        Primero desmezcla celdas ambiguas y luego, en filas con
        doble cuantitativo, mueve fragmentos textuales hacia la columna
        descriptiva y divide pares numéricos en columnas decimales vecinas.
        """
        text_idx = context[DataMathDict.TEXT_COL.value]
        df, df_copy = self.unmix_cells(df, context)
        context[DataMathDict.DF_COPY.value] = df_copy

        tables_array = self.get_arrays_table(context)
        elements_array, cuantiative_array = tables_array[0], tables_array[SemantiClass.QUANTITATIVE]
        textual_array = tables_array[SemantiClass.DESCRIPTIVE] + tables_array[SemantiClass.UMD]
        cols_idx = context[DataMathDict.COLS_IDX.value]

        cols_decimal_list = [(df.columns.get_loc(name) if name in df.columns else None) for name in self.dec_cols_name]
        idx_decimal = np.sort(np.asarray(cols_decimal_list, dtype=np.uint8))

        rows_double = np.where(cuantiative_array==2)[0]
        double_cuant = cuantiative_array[rows_double]
        double_element = elements_array[rows_double]
        double_text = textual_array[rows_double]

        alone_doubles_rel = np.where(double_element == double_cuant)[0]                         # Índices relativos de Columnas decimales donde hay texto

        df_copy: pd.DataFrame = context[DataMathDict.DF_COPY.value]
        dest_idx = int(text_idx[0])
        # logger.debug(f"closest: {closest_idx}")

        for _, rr in enumerate(alone_doubles_rel):
            r = rows_double[rr]

            row_text_cols = np.where(double_text[rr])[0]
            intersect_cols_text_idx = np.asarray(np.intersect1d(row_text_cols, idx_decimal), dtype=np.uint8)

            row_double_cols = np.where(double_cuant[rr])[0]
            if len(row_double_cols) == 0:
                continue
            cur_double_col = int(row_double_cols[0])

            for c in intersect_cols_text_idx:
                if c == dest_idx:
                    continue

                val = str(df.iat[r, c])
                if val == "" or not val:
                    continue

                poly_val: List[str] = df_copy.iat[r, c]
                dest_vals = df_copy.iat[r, dest_idx]

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
                    df.iat[r, dest_idx] = val + " " + str(df.iat[r, dest_idx])
                    df_copy.iat[r, dest_idx] = poly_val + dest_vals
                
                else:
                    df.iat[r, dest_idx] = str(df.iat[r, dest_idx]) + " " + val
                    df_copy.iat[r, dest_idx] = dest_vals + poly_val

                    df.iat[r, c] = ""
                    df_copy.iat[r, c] = []

            for dc in cols_idx:
                # logger.debug(f"COLUMN: {dc}")
                dec_val: str = str(df.iat[r, dc])
                poly_vals: List[str] = df_copy.iat[r, dc]
                dest_polys = df_copy.iat[r, closest_idx]

                if dec_val == "" or not dec_val:
                    continue

                if dc == cur_double_col:
                    split_decimals = dec_val.split(" ", 1)
                    # logger.debug(f"vals: {vals}")
                    if closest_idx == dc:
                        continue
                        
                    elif closest_idx < dc:
                        df.iat[r, closest_idx] = str(split_decimals[0])
                        df.iat[r, dc] = str(split_decimals[1])

                        df_copy.iat[r, closest_idx] = dest_polys + [poly_vals[0]]
                        df_copy.iat[r, dc] = [poly_vals[1]] + dest_polys

                    else:
                        df.iat[r, closest_idx] = str(split_decimals[1])
                        df.iat[r, dc] = str(split_decimals[0])

                        df_copy.iat[r, closest_idx] = dest_polys + [poly_vals[1]]
                        df_copy.iat[r, dc] = [poly_vals[0]] + dest_polys

        # logger.info("SEPARATED:\n" + df.to_string(index=True))
        return (df, df_copy)

    def simplify_rows(self, df: pd.DataFrame, context: Dict[str, Any]):
        """Elimina filas dobles"""
        arrays_table = self.get_arrays_table(context)
        elements_array = arrays_table[0]
        unique_cells = np.count_nonzero(elements_array, axis=1) == 1
        if unique_cells.size < 1:
            logger.info(f"DF COMPLETO")
            return df

        c_target = df.columns.get_loc(DataKeys.producto_norm.value)
        empty_rows_idx = np.where(unique_cells)[0]
        # logger.info("DIRTY:\n" + df.to_string(index=True))
        for _, r in enumerate(empty_rows_idx[::-1]):
            c_idx = np.nonzero(elements_array[r])[0]
            if len(c_idx) == 0:
                continue

            c = int(c_idx[0])
            r_target = int(r) - 1
            if r_target < 0:
                continue

            val = str(df.iat[r, c])
            if val != "" and val:
                target_val = str(df.iat[r_target, c_target])

                if target_val != "" and target_val:
                    df.iat[r_target, c_target] = str(target_val) + " " + val
                else:
                    df.iat[r_target, c_target] = val

                df.iat[r, c] = ""

        df = df.drop(empty_rows_idx, axis=0)
        # logger.info("REINDEX:\n" + df.to_string(index=True))
        return df

    def solve_incomplete(self, df: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # logger.info("INCOMPLETO RECIBIDO:\n" + df.to_string(index=True))
        cols_idx = context[DataMathDict.COLS_IDX.value]
        arrays_table = self.get_arrays_table(context)
        decimal_array = arrays_table[SemantiClass.QUANTITATIVE]
        semi_decimal_rows = np.count_nonzero(decimal_array, axis=1, keepdims=True)
        top_dec_counts = np.max(semi_decimal_rows)
        non_empty_rows_idx = np.where(semi_decimal_rows==top_dec_counts)[0]

        if non_empty_rows_idx.size < 1:
            logger.warning("DF SIN DATOS SUFICIENTES, IMPOSIBLE PROCESAR")
            return (pd.DataFrame(), {})

        max_decimal_array = decimal_array[non_empty_rows_idx]
        decimal_coords = np.transpose(np.nonzero(max_decimal_array))
        decimal_rows = decimal_coords[:, 0]

        decimal_u, decimal_i = np.unique(decimal_rows, return_inverse=True) # Array de indices únicos de filas
        decimal_u = non_empty_rows_idx[decimal_u]                           # Obtención de los índices absolutos

        decimal_cols = decimal_coords[:, 1]
        decimal_rows = decimal_u[decimal_i]                                 # Mapeo de índices asbolutos

        decimal_coords_abs = np.column_stack([decimal_rows, decimal_cols])
        # #logger.info("ABS_CORRDS:\n"f"{decimal_coords_abs}")

        potencial_val = np.zeros(df.shape, dtype=np.uint8)
        for _, r in enumerate(non_empty_rows_idx):
            dec_cells = decimal_coords_abs[decimal_coords_abs[:, 0] == r, 1]

            val_a = Decimal(df.iat[r, dec_cells[0]])
            val_b = Decimal(df.iat[r, dec_cells[1]])

#            logger.debug(f"ROW {r} VALUES:\n"f"A: '{val_a}', B: '{val_b}'")
            a_int = val_a.to_integral_value()

            if val_b == val_a:
                if val_a == a_int:              # Ambos enteros
                    potencial_val[r, 0] = 1.0
                    continue
                value_a = val_a
                value_b = val_b

            elif val_b > val_a:
                value_a = val_a
                value_b = val_b
            else:
                value_a = val_b
                value_b = val_a

            quotient1 = value_b / value_a                                                   # Cociente
 #           logger.debug(f"COCIENTE 1: {quotient1}")
            quotient2 = quotient1.to_integral_value(rounding=ROUND_HALF_UP)                 # Redondear por si acaso
  #          logger.debug(f"COCIENTE 2: {quotient2}")
            quotient_diff = _zero if quotient1 != quotient2 else abs(quotient2 - quotient1)  # Diferencia absoluta del redondeo y valor real
            if quotient_diff == _zero or quotient_diff < _row_tol:          # Debajo de umbral
                potencial_val[r, 0] = int(quotient2)
            continue                                                                        # Momentaneamente, después agregaré los casos para pu y mtl

  #      logger.info("SEMI_COMP:\n" + df.iloc[non_empty_rows_idx].to_string(index=True))
        # logger.debug("SEMI_ARRAY:\n"f"{np.column_stack([non_empty_rows_idx, max_decimal_array])}")
        # logger.debug("\n"f"{np.column_stack([non_empty_rows_idx, potencial_val[non_empty_rows_idx]])}")

        shadows_dec, shadows_counts = np.unique(max_decimal_array, axis=0, return_counts=True)
        major_shadow_idx = np.argmax(shadows_counts, keepdims=True)
        major_shadow = shadows_dec[major_shadow_idx]                                            # Patrón decimal más frecuente
        major_rows_idx = np.where(np.all(major_shadow == max_decimal_array, axis=1))[0]         # índices donde hay match con el shadow
        major_rows_idx = non_empty_rows_idx[major_rows_idx]

        total_inserts = major_rows_idx.size
        if total_inserts < 1:
            logger.error("NO SE HALLARON FILAS APTAS, IMPOSIBLE PROCESAR")
            return (pd.DataFrame(), {})
        
       # logger.debug(f"MAJOR SHADOW: {major_shadow}")
        dec_cols_idx = np.where(major_shadow)[1]
        non_dec_idx = np.setdiff1d(cols_idx, dec_cols_idx, assume_unique=True)
        text_col = np.argmax(non_dec_idx, keepdims=True)
        text_col = non_dec_idx[text_col]

        dest_idx = int(text_col[0])
        # #logger.info(f"DEC IDX: {non_dec_idx}, {dec_cols_idx}, {dest_idx}, ROWS: {major_rows_idx}")
      #  logger.info("SHAWDOW_DF:\n" + df.iloc[major_rows_idx].to_string(index=True))

        cut_polygons = context[DataMathDict.CUT_POLYGONS.value]
        max_idx = (cut_polygons[DataMathDict.MAX_IDX.value] + 1)
        new_inserts = max_idx + total_inserts

        artificial_polys: List[str] = []
        for idx in range(max_idx, new_inserts):
            new_poly_id = f"poly_{idx:04d}"
            cut_polygons[new_poly_id] = {}
            artificial_polys.append(new_poly_id)

        df_copy: pd.DataFrame = context[DataMathDict.DF_COPY.value]
        # logger.info("DF COPY ORIGINAL:\n" + df_copy.to_string(index=True))

        for i, r in enumerate(major_rows_idx):
            r = int(r)
            for c in non_dec_idx:
                c = int(c)

                if c == dest_idx:
                    continue

                val = str(df.iat[r, c]).strip()
                vals: List[str] = df_copy.iat[r, c]
                dest_vals = df_copy.iat[r, dest_idx]

                if val == "" or not val:
                    continue

                if c < dest_idx:
                    df.iat[r, dest_idx] = (val + " " + str(df.iat[r, dest_idx])).strip()
                    df_copy.iat[r, dest_idx] = vals + dest_vals
                else:
                    df.iat[r, dest_idx] = str(df.iat[r, dest_idx] + " " + val).strip()
                    df_copy.iat[r, dest_idx] = dest_vals + vals

                artifial_data = str(potencial_val[r, c]).strip()
                df.iat[r, c] = artifial_data
                current_poly = artificial_polys[i]

                cut_polygons[current_poly] = {
                    DataMathDict.TEXT.value: artifial_data,
                    DataMathDict.SEMANTIC_CLASIFICATION.value: [SemantiClass.NUMERIC]
                }
                df_copy.iat[r, c] = [current_poly]

        dec_cols = np.array(np.setdiff1d(cols_idx, text_col, assume_unique=True), np.int_)
        context[DataMathDict.DF_COPY.value] = df_copy
        context[DataMathDict.CUT_POLYGONS.value] = cut_polygons
        context[DataMathDict.DEC_ROWS_IDS.value] = major_rows_idx
        context[DataMathDict.DEC_COLS.value] = dec_cols
        context[DataMathDict.ARITH_COLS_IDS.value] = dec_cols
        context[DataMathDict.TEXT_COL_TEMP.value] = text_col
        #logger.info("DF INVENTADO:\n" + df.to_string(index=True))
        #logger.info("DF COPY INVENTADO:\n" + df_copy.to_string(index=True))
        return (df, context)
