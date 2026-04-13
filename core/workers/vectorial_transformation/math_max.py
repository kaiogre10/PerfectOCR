# PerfectOCR/core/workers/vectorial_transformation/math_max.py
import pandas as pd # type: ignore
import logging
from itertools import permutations
import numpy as np
import time
from itertools import permutations
from typing import Dict, Any, List, Tuple, Optional, cast
from decimal import Decimal, InvalidOperation
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from services.output_service import save_debug_table
from core.utils.text_utils import clean_cuant

logger = logging.getLogger(__name__)

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
        self.arithmetic_tolerance = Decimal(str(tol)) if tol is not None else Decimal('0.01')
        self.output = config.get("math_max_corrected", False)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        try:
            start_time = time.time()

            table_matrix = cast(List[List[Dict[str, Any]]], context["table_matrix"])
            if not table_matrix:
                logger.error("No hay table_matrix en contexto para procesar")
                return False

            table_columns = cast(List[str], context["table_columns"])
            if not table_columns:
                inferred_width = len(table_matrix[0]) if table_matrix else 0
                table_columns = [f"col_{i}" for i in range(inferred_width)]

            df = self._table_matrix_to_dataframe(table_matrix, table_columns)
            if df.empty:
                logger.error("La table_matrix no contiene filas/columnas válidas")
                return False

            logger.info("Tabla recibida para corrección matemática:\n" + df.to_string(index=True))

            polygons_dict: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            corrected_df, final_semantic_types = self.solve(df, table_matrix, polygons_dict)

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

            manager.save_structured_table(df=corrected_df, columns=list(corrected_df.columns), semantic_types=final_semantic_types)

            total_time = time.time() - start_time
            logger.info(f"Corrección matemática completada en {total_time:.6f}s, Se encontraron {len(corrected_df)} filas.")
            return True
        except Exception as e:
            logger.error(f"Error en MatrixSolver.vectorize: {e}", exc_info=True)
            return False
            
    def solve(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]], polygons_dict: Dict[str, Polygons]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fase 1: Identifica roles C, PU, MTL usando clasificación semántica
        de polígonos y votación global con aritmética Decimal.
        """
        df = df.copy()
        columns: List[str] = list(df.columns)
        H = len(columns)
        
        # logger.info(f"TABLE MATRIX: {table_matrix[0]}")

        # --- PASO 0: Validación de Soledad ---
        self._validate_cells(df, table_matrix, H)
        df, table_matrix = self._enforce_solitude(df, table_matrix, polygons_dict, H)

        # --- FASE 1: Votación basada en clasificación semántica ---
        complete_rows = self._get_complete_rows(table_matrix, H)
        logger.info(f"Filas completas: {sorted(complete_rows)}")

        row_quant_map, qualifying_rows = self._identify_quant_cells(table_matrix, complete_rows, H)
        logger.info(f"Filas cualificadas (>=3 cuantitativos válidos): {qualifying_rows}")

        basic_types = self._infer_column_types(table_matrix, H)
        logger.info(f"Tipos semánticos: {basic_types}")

        if not qualifying_rows:
            logger.info("No hay filas con >= 3 cuantitativos válidos; no se corrige.")
            return df, basic_types

        hypothesis = self._vote_hypothesis(row_quant_map, qualifying_rows)

        if hypothesis is None:
            logger.error("No se encontró hipótesis válida; no se corrige.")
            return df, basic_types

        c_col, pu_col, mtl_col = hypothesis
        logger.info(f"Roles Globales: C='{columns[c_col]}', PU='{columns[pu_col]}', MTL='{columns[mtl_col]}'")

        final_semantic_types = basic_types[:]
        final_semantic_types[c_col] = "cuantitativo, c"
        final_semantic_types[pu_col] = "cuantitativo, pu"
        final_semantic_types[mtl_col] = "cuantitativo, mtl"

        # --- FASE 2: Reconstrucción Aritmética ---
        ZERO = Decimal('0')
        for row_idx in complete_rows:
            row_cells = table_matrix[row_idx]
            
            try:
                c_val = Decimal(clean_cuant(str(row_cells[c_col].get('text', '') or '')))
            except (InvalidOperation, ValueError):
                c_val = None
                
            try:
                pu_val = Decimal(clean_cuant(str(row_cells[pu_col].get('text', '') or '')))
            except (InvalidOperation, ValueError):
                pu_val = None
                
            try:
                mtl_val = Decimal(clean_cuant(str(row_cells[mtl_col].get('text', '') or '')))
            except (InvalidOperation, ValueError):
                mtl_val = None

            if c_val is not None and pu_val is not None and c_val > ZERO and pu_val > ZERO:
                expected_mtl = c_val * pu_val
                if mtl_val is None or expected_mtl != mtl_val:
                    new_mtl_str = self._format_num(expected_mtl)
                    row_cells[mtl_col]['text'] = new_mtl_str
                    df.iloc[row_idx, mtl_col] = new_mtl_str
                    logger.info(f"Fila {row_idx}: Corrección MTL -> {new_mtl_str}")
            elif c_val is not None and mtl_val is not None and c_val > ZERO:
                expected_pu = mtl_val / c_val
                if pu_val is None or expected_pu != pu_val:
                    new_pu_str = self._format_num(expected_pu)
                    row_cells[pu_col]['text'] = new_pu_str
                    df.iloc[row_idx, pu_col] = new_pu_str
                    logger.info(f"Fila {row_idx}: Corrección PU -> {new_pu_str}")

        return df, final_semantic_types

    def _validate_cells(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]], H: int):
        R = len(table_matrix)
        matrix_array = np.zeros((R, H), np.int8)
        matrix_decimal = matrix_array.copy()
        matrix_quantity = matrix_array.copy()
        elements_array = matrix_array.copy()
        textual_array = matrix_array.copy()
        
        for row_id, rows in enumerate(table_matrix):
            # logger.info(f"ROWS 0: '{table_matrix[row_id][0]}'")
            rows = table_matrix[row_id]
            for i in range(H):
                # logger.info(f"ROWS: '{rows[i]}'")
                sc_v = rows[i]["semantic_clasification"]
                elements_array[row_id, i] = len(rows[i]["polygon_ids"])
                text = rows[i]["text"]
                
                if not text:
                    elements_array[row_id, i] = 0
                    
                elif all(s == 4 for s in sc_v):
                    matrix_decimal[row_id, i] = len(sc_v)
                    
                elif all(s == 5 for s in sc_v):
                    matrix_quantity[row_id, i] = len(sc_v)
                
                else:
                    textual_array[row_id, i] = -1
                #     cols_vot.append(("textual", i))
        
        decimal_mask = (np.count_nonzero(matrix_decimal, axis=1) > 0) & (np.count_nonzero(matrix_quantity, axis=1) > 0)
        
        full_rows_mask = (np.count_nonzero(elements_array, axis=1) == H) & decimal_mask
        
        full_idx = np.where(full_rows_mask)[0]
        matrix_decimal = elements_array[full_idx]
        
        decimal_votes = np.count_nonzero(matrix_decimal, axis=0)
        quantityt_votes = np.count_nonzero(matrix_quantity, axis=0)
        
        # dec_idx = np.where(decimal_rows)[0]
        # matrix_decimal = matrix_decimal[dec_idx]
        
        logger.info(f"{full_idx}")
        # logger.info(f"DECIMAL: \n"f"{matrix_decimal}")
        # df = df.iloc[full_idx]
        logger.info("FULL ROWS:\n" + df.iloc[full_idx].to_string(index=True))
        
        
        
        
        decimal_cols = np.where(decimal_votes > 0)[0]
        # quantity_cols = np.where(quantityt_votes > 0)[0]
        
        leftsquantity = 3 - decimal_cols.shape[0]
        
        idx = np.argsort(quantityt_votes)[-leftsquantity:][::-1]
        
        logger.info(f"COLUMNAS CUANTITATIVAS: {decimal_cols}, COMPLEMENTARIAS: {idx}")
        
        # decimal_matrix_final = matrix_decimal + matrix_quantity
        # decimal_matrix_final = matrix_decimal[non_idx]
        # logger.info(f"MATRIZ FINAL: {decimal_matrix_final}")
        
        # Construimos la lista de tipos semánticos columna a columna acorde a los índices de decimal_cols y final_quantity_col
        types_s = []
        for col_idx in range(H):
            if col_idx in decimal_cols:
                types_s.append("decimal")
            elif col_idx in idx:
                types_s.append("decimal")
            else:
                types_s.append("textual")
 
        logger.info(f"TYPOS: '{types_s}'")

    def _enforce_solitude(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]], polygons_dict: Dict[str, Polygons], H: int) -> Tuple[pd.DataFrame, List[List[Dict[str, Any]]]]:
        """Si una celda contiene 2+ polígonos con sc=2, desplaza el excedente a celda adyacente."""
        for row_idx, row in enumerate(table_matrix):
            for col_idx in range(min(len(row), H)):
                cell = row[col_idx]
                pids: List[str] = cell['polygon_ids']
                if len(pids) < 2:
                    continue

                quant_pids: List[str] = [
                    pid for pid in pids
                    if pid in polygons_dict and 4 in polygons_dict[pid].semantic_clasification
                ]

                if len(quant_pids) < 2:
                    continue

                # Intentar preservar orden geométrico
                # Si hay 2, intentamos mover el derecho hacia la derecha, o el izquierdo hacia la izquierda
                placed = False

                # 1. Intentar mover el último hacia la derecha
                right_pid = quant_pids[-1]
                target_col = col_idx + 1
                if target_col < H:
                    target_cell = row[target_col]
                    target_pids = target_cell.get('polygon_ids', [])
                    target_has_quant = any(
                        tpid in polygons_dict and 4 in polygons_dict[tpid].semantic_clasification
                        for tpid in target_pids
                    )
                    if not target_has_quant:
                        self._move_polygon(df, row, row_idx, col_idx, target_col, right_pid, polygons_dict)
                        placed = True

                # 2. Si no se pudo, intentar mover el primero hacia la izquierda
                if not placed:
                    left_pid = quant_pids[0]
                    target_col = col_idx - 1
                    if target_col >= 0:
                        target_cell = row[target_col]
                        target_pids = target_cell.get('polygon_ids', [])
                        target_has_quant = any(
                            tpid in polygons_dict and 4 in polygons_dict[tpid].semantic_clasification
                            for tpid in target_pids
                        )
                        if not target_has_quant:
                            self._move_polygon(df, row, row_idx, col_idx, target_col, left_pid, polygons_dict)
                            placed = True

                if not placed:
                    logger.warning(f"No se pudo aislar cuantitativos en fila {row_idx}, col_{col_idx} sin violar orden")

        return df, table_matrix

    def _move_polygon(
        self, df: pd.DataFrame, row: List[Dict[str, Any]], row_idx: int, src_col: int,
        target_col: int, pid: str, polygons_dict: Dict[str, Polygons]
    ) -> None:
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

    # ── Fase 1: Identificación de cuantitativos y votación ───────────────

    def _get_complete_rows(self, table_matrix: List[List[Dict[str, Any]]], H: int) -> set[int]:
        """Filas donde todas las H columnas tienen texto no vacío."""
        complete: set[int] = set()
        for row_idx, row in enumerate(table_matrix):
            if len(row) < H:
                continue
            is_complete = all(
                str(row[col_idx].get('text', '') or '').strip() != ''
                for col_idx in range(H)
            )
            if is_complete:
                complete.add(row_idx)
        return complete

    def _identify_quant_cells(self, table_matrix: List[List[Dict[str, Any]]], complete_rows: set[int], H: int) -> Tuple[List[Dict[int, Decimal]], List[int]]:
        """
        Solo en filas completas, identifica celdas cuantitativas válidas:
        len(sc)==1, sc[0] in {1,2}, len(polygon_ids)==1.
        Solo convierte a Decimal el texto de esas celdas.
        """
        row_quant_map: List[Dict[int, Decimal]] = []
        qualifying_rows: List[int] = []

        for row_idx, row in enumerate(table_matrix):
            quant_values: Dict[int, Decimal] = {}

            if row_idx in complete_rows:
                for col_idx in range(min(len(row), H)):
                    cell = row[col_idx]
                    sc: List[int] = cell.get('semantic_clasification', [])
                    pids: List[str] = cell.get('polygon_ids', [])

                    if len(sc) == 1 and sc[0] in (1, 2) and len(pids) == 1:
                        text = str(cell.get('text', '') or '').strip()
                        try:
                            cleaned = clean_cuant(text)
                            quant_values[col_idx] = Decimal(cleaned)
                        except (InvalidOperation, ValueError):
                            continue

            row_quant_map.append(quant_values)
            if len(quant_values) >= 3:
                qualifying_rows.append(row_idx)

        return row_quant_map, qualifying_rows

    def _vote_hypothesis(self, row_quant_map: List[Dict[int, Decimal]], qualifying_rows: List[int]) -> Optional[Tuple[int, int, int]]:
        """Votación global: cada fila cualificada vota por permutaciones (C, PU, MTL)."""
        all_quant_cols: set[int] = set()
        for ri in qualifying_rows:
            all_quant_cols.update(row_quant_map[ri].keys())

        quant_col_list = sorted(all_quant_cols)
        if len(quant_col_list) < 3:
            return None

        perm_list: List[Tuple[int, int, int]] = list(permutations(quant_col_list, 3))
        scores: Dict[Tuple[int, int, int], float] = {p: 0.0 for p in perm_list}

        for row_idx in qualifying_rows:
            qv = row_quant_map[row_idx]
            valid = self._test_hypotheses(qv, perm_list)

            if len(valid) == 1:
                scores[valid[0]] += 1.0
            elif len(valid) == 2:
                scores[valid[0]] += 0.5
                scores[valid[1]] += 0.5

        if not any(s > 0 for s in scores.values()):
            return None

        return max(scores, key=lambda k: scores[k])

    def _test_hypotheses(self, row_values: Dict[int, Decimal], perm_list: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Prueba cada permutación contra los axiomas usando aritmética Decimal."""
        valid: List[Tuple[int, int, int]] = []
        ZERO = Decimal('0')

        for c_col, pu_col, mtl_col in perm_list:
            c = row_values.get(c_col)
            pu = row_values.get(pu_col)
            mtl = row_values.get(mtl_col)

            if c is None or pu is None or mtl is None:
                continue

            # Axiomas
            if c <= ZERO or pu <= ZERO or mtl <= ZERO:
                continue
            if mtl < c * pu:
                continue
            if pu < mtl:
                continue

            product = c * pu
            if product == ZERO:
                continue

            rel_diff = abs(product - mtl) / product
            if rel_diff <= self.arithmetic_tolerance:
                valid.append((c_col, pu_col, mtl_col))

        return valid

    # ── Inferencia de tipos por columna ──────────────────────────────────

    def _infer_column_types(self, table_matrix: List[List[Dict[str, Any]]], H: int) -> List[str]:
        """Clasifica columnas por mayoría de votos semánticos de sus polígonos."""
        types: List[str] = []
        for col_idx in range(H):
            sc_votes = self._collect_column_semantics(table_matrix, col_idx)
            quant_votes = sc_votes.get(1, 0) + sc_votes.get(2, 0) + sc_votes.get(4, 0) + sc_votes.get(5, 0)
            text_votes = sc_votes.get(0, 0) + sc_votes.get(-1, 0) + sc_votes.get(-2, 0)
            types.append("cuantitativo" if quant_votes > text_votes else "texto")
        return types

    def _collect_column_semantics(self, table_matrix: List[List[Dict[str, Any]]], col_idx: int) -> Dict[int, int]:
        votes: Dict[int, int] = {}
        for row in table_matrix:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            semantic_values = cell["semantic_clasification"]
            if isinstance(semantic_values, int):
                semantic_values = [semantic_values]
            if not isinstance(semantic_values, list):
                continue
            for val in semantic_values:
                try:
                    key = int(val)
                    votes[key] = votes.get(key, 0) + 1
                except Exception:
                    continue
        return votes

    def _table_matrix_to_dataframe(self, table_matrix: List[List[Dict[str, Any]]], columns: List[str]) -> pd.DataFrame:
        rows: List[List[str]] = []
        width = len(columns)
        for row in table_matrix:
            row_values: List[str] = []
            for col_idx in range(width):
                text_val = ""
                if col_idx < len(row):
                    text_val = str(row[col_idx].get("text", "") or "")
                row_values.append(text_val)
            rows.append(row_values)
        return pd.DataFrame(rows, columns=columns)

    def _format_num(self, val: Decimal) -> str:
        if val == val.to_integral_value():
            return str(int(val))
        return str(val.quantize(Decimal('0.01')))
