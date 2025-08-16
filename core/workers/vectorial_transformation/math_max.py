# PerfectOCR/core/postprocessing/math_max.py
import pandas
import logging
from itertools import permutations
import math
import numpy as np
from typing import Dict, Any, Optional
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter


logger = logging.getLogger(__name__)

class MatrixSolver(VectorizationAbstractWorker):
    """
    Resuelve inconsistencias matemáticas en una tabla estructurada usando un
    enfoque de puntuación global y validación final contra un total.
    """

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('math_max', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_structured", False)        
        self.total_mtl_tolerance = self.config.get('total_mtl_abs_tolerance', 0.05) # 5% de tolerancia relativa por defecto
        self.arithmetic_tolerance = self.config.get('row_relative_tolerance', 0.005) # 0.5% por defecto
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:

    def _clean_numeric_value(self, value):
        """Limpia símbolos comunes de valores numéricos antes de convertir a float."""
        if not isinstance(value, str):
            return value
        cleaned = value.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
        return cleaned

    def _get_numeric_matrix(self, matrix_data: list, semantic_types: list):
        """Extrae las columnas cuantitativas y las convierte en una matriz numérica."""
        quant_cols_indices = [i for i, s_type in enumerate(semantic_types) if s_type == "cuantitativo"]

        if len(quant_cols_indices) < 3:
            return None, None

        numeric_matrix = []
        for row in matrix_data:
            numeric_row = []
            for index in quant_cols_indices:
                cell_value = row[index]
                try:
                    # Limpia el valor antes de convertir
                    cleaned_value = self._clean_numeric_value(cell_value)
                    numeric_row.append(float(cleaned_value) if cleaned_value else None)
                except (ValueError, TypeError):
                    numeric_row.append(None)
            numeric_matrix.append(numeric_row)
        
        return numeric_matrix, quant_cols_indices

    def _log_matrix(self, title: str, matrix: list):
        """Registra una matriz numérica en el log de forma legible."""
        logger.info(f"--- {title} ---")
        if not matrix:
            logger.info("[]")
            return
        
        # Formateo para una visualización limpia
        formatted_rows = []
        for row in matrix:
            row_str = " | ".join([f"{item:<10.2f}" if isinstance(item, (int, float)) else f"{str(item):<10}" for item in row])
            formatted_rows.append(row_str)
        
        logger.info("\n".join(formatted_rows))
        logger.info("-" * (len(formatted_rows[0]) if formatted_rows else 0))

    def _get_valid_hypotheses_for_row(self, row, permutations_indices):
        """Encuentra todas las hipótesis válidas para una sola fila."""
        valid_hypotheses = []
        for p_indices in permutations_indices:
            c_idx, pu_idx, mtl_idx = p_indices
            # Asegurarse de que los índices están dentro de los límites de la fila
            if max(c_idx, pu_idx, mtl_idx) >= len(row):
                continue
            
            c, pu, mtl = row[c_idx], row[pu_idx], row[mtl_idx]

            if c is None or pu is None or mtl is None:
                continue

            if not all(isinstance(v, (int, float)) for v in [c, pu, mtl]):
                continue

            # Axiomas
            if c <= 0 or pu <= 0 or mtl <= 0: continue
            if mtl < pu: continue

            if math.isclose(c * pu, mtl, rel_tol=self.arithmetic_tolerance):
                valid_hypotheses.append(p_indices)
        
        return valid_hypotheses

    def solve(self, matrix: list, semantic_types: list, quarantined_data: list, document_totals: Optional[Dict] = None) -> Dict:
        """
        Punto de entrada principal para resolver la matriz.
        Ahora acepta los argumentos que le pasa el coordinador.
        """
        if document_totals:
            total_mtl = document_totals.get("total_mtl")
            total_c = document_totals.get("total_c")
            logger.info(f"[math_max] Totales previos recibidos: total_mtl={total_mtl}, total_c={total_c}")
        else:
            logger.info("[math_max] No se recibieron totales previos (document_totals=None)")

        numeric_matrix, quant_indices_map = self._get_numeric_matrix(matrix, semantic_types)

        if not numeric_matrix:
            logger.warning("No se encontraron suficientes columnas cuantitativas para el análisis.")
            return {"matrix": matrix, "semantic_math_types": semantic_types}
        
        self._log_matrix("Matriz Numérica Extraída (Antes de Correcciones)", numeric_matrix)

        # --- FASE 1: Puntuación y Selección de Hipótesis Candidata ---
        col_indices_in_numeric_matrix = list(range(len(quant_indices_map)))
        permutations_indices = list(permutations(col_indices_in_numeric_matrix, 3))
        hypothesis_scores = {p: 0 for p in permutations_indices}
        
        for row in numeric_matrix:
            valid_hypotheses = self._get_valid_hypotheses_for_row(row, permutations_indices)
            
            if len(valid_hypotheses) == 1:
                hypothesis_scores[valid_hypotheses[0]] += 1.0
            elif len(valid_hypotheses) == 2:
                hypothesis_scores[valid_hypotheses[0]] += 0.5
                hypothesis_scores[valid_hypotheses[1]] += 0.5
        
        if not any(s > 0 for s in hypothesis_scores.values()):
            logger.error("No se pudo encontrar ninguna hipótesis válida en toda la tabla. Abortando corrección matemática.")
            return {"matrix": matrix, "semantic_math_types": semantic_types}

        candidate_hypothesis_indices = max(hypothesis_scores, key=hypothesis_scores.get)
        c_idx, pu_idx, mtl_idx = candidate_hypothesis_indices
        logger.info(f"Hipótesis candidata seleccionada (índices en matriz numérica): C={c_idx}, PU={pu_idx}, MTL={mtl_idx} con {hypothesis_scores[candidate_hypothesis_indices]:.1f} puntos.")

        # --- FASE 2: Reconstrucción Total bajo la Hipótesis Candidata ---
        reconstructed_matrix = [row[:] for row in numeric_matrix]
        col_medians = {i: np.nanmedian([row[i] for row in reconstructed_matrix if row[i] is not None]) for i in col_indices_in_numeric_matrix}

        rows_with_two_missing = []
        for i, row in enumerate(reconstructed_matrix):
            missing_count = sum(1 for x in [row[c_idx], row[pu_idx], row[mtl_idx]] if x is None)
            
            if missing_count >= 2:
                rows_with_two_missing.append(i)
                continue

            if missing_count == 1:
                # Completar valor faltante
                try:
                    if row[mtl_idx] is None: row[mtl_idx] = row[c_idx] * row[pu_idx]
                    elif row[pu_idx] is None and row[c_idx] != 0: row[pu_idx] = row[mtl_idx] / row[c_idx]
                    elif row[c_idx] is None and row[pu_idx] != 0: row[c_idx] = row[mtl_idx] / row[pu_idx]
                except ZeroDivisionError:
                    logger.warning(f"División por cero al intentar completar la fila {i}. La fila permanecerá incompleta.")

            elif not math.isclose(row[c_idx] * row[pu_idx], row[mtl_idx], rel_tol=self.arithmetic_tolerance):
                # Corregir fila inconsistente
                dev_c = abs(row[c_idx] - col_medians.get(c_idx, row[c_idx]))
                dev_pu = abs(row[pu_idx] - col_medians.get(pu_idx, row[pu_idx]))
                dev_mtl = abs(row[mtl_idx] - col_medians.get(mtl_idx, row[mtl_idx]))
                
                max_dev = max(dev_c, dev_pu, dev_mtl)
                try:
                    if max_dev == dev_c: row[c_idx] = row[mtl_idx] / row[pu_idx]
                    elif max_dev == dev_pu: row[pu_idx] = row[mtl_idx] / row[c_idx]
                    else: row[mtl_idx] = row[c_idx] * row[pu_idx]
                except ZeroDivisionError:
                    logger.warning(f"División por cero al intentar corregir la fila {i}. La fila permanecerá inconsistente.")

        # --- FASE 3: Juicio Final y Casos Especiales ---
        total_mtl = document_totals.get("total_mtl") if document_totals else None
        
        if total_mtl is not None:
            if len(rows_with_two_missing) == 1:
                idx = rows_with_two_missing[0]
                row = reconstructed_matrix[idx]
                if row[mtl_idx] is None:
                    mtl_sum = sum(r[mtl_idx] for i, r in enumerate(reconstructed_matrix) if r[mtl_idx] is not None and i != idx)
                    row[mtl_idx] = total_mtl - mtl_sum
                    # Completar el último valor
                    if row[c_idx] is None and row[pu_idx] != 0: row[c_idx] = row[mtl_idx] / row[pu_idx]
                    elif row[pu_idx] is None and row[c_idx] != 0: row[pu_idx] = row[mtl_idx] / row[c_idx]
            
            final_mtl_sum = sum(r[mtl_idx] for r in reconstructed_matrix if r[mtl_idx] is not None)
            if not math.isclose(final_mtl_sum, total_mtl, rel_tol=self.total_mtl_tolerance):
                logger.error(f"VALIDACIÓN FALLIDA: Suma de MTL reconstruido ({final_mtl_sum:.2f}) no coincide con total del documento ({total_mtl:.2f}). Se descarta la reconstrucción.")
                return {"matrix": matrix, "semantic_math_types": semantic_types}
            else:
                logger.info(f"VALIDACIÓN EXITOSA: Suma de MTL reconstruido ({final_mtl_sum:.2f}) coincide con total del documento.")

        # Integrar la matriz numérica corregida de vuelta en la matriz de strings original
        final_matrix_str = [row[:] for row in matrix]
        for i, row_num in enumerate(reconstructed_matrix):
            for j, col_idx_num in enumerate(col_indices_in_numeric_matrix):
                original_col_index = quant_indices_map[col_idx_num]
                value = row_num[col_idx_num]
                if value is not None:
                    # Formatear a string con 2 decimales o como entero
                    final_matrix_str[i][original_col_index] = f"{value:.2f}" if value != int(value) else str(int(value))

        self._log_matrix("Matriz Numérica Corregida (Después de Correcciones)", reconstructed_matrix)
        
        # Generar los tipos semánticos finales para el output
        final_semantic_types = semantic_types[:]
        final_semantic_types[quant_indices_map[c_idx]] = "cuantitativo, c"
        final_semantic_types[quant_indices_map[pu_idx]] = "cuantitativo, pu"
        final_semantic_types[quant_indices_map[mtl_idx]] = "cuantitativo, mtl"

        return {"matrix": final_matrix_str, "semantic_math_types": final_semantic_types}