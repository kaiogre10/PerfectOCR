# PerfectOCR/core/postprocessing/math_max.py
import pandas as pd
import logging
import time
from itertools import permutations
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
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
        try:
            start_time = time.time()

            # Recupera el DataFrame estructurado directamente de memoria
            df = manager.get_structured_table()
            if df is None or df.empty:
                logger.warning("[MatrixSolver] No hay tabla estructurada para procesar")
                return False

            # Ejecutar solver 100% en pandas (sin totales, sin cuarentena)
            df_corr, semantic_types_final = self.solve(df)

            # Guardar el DataFrame corregido y los tipos semánticos inferidos
            manager.save_structured_table(df=df_corr, columns=list(df_corr.columns), semantic_types=semantic_types_final)

            elapsed = time.time() - start_time
            logger.debug(f"[MatrixSolver] Corrección matemática completada en {elapsed:.3f}s,  Se encontraron {len(df_corr)} filas.")
            return True
        except Exception as e:
            logger.error(f"Error en MatrixSolver.vectorize: {e}", exc_info=True)
            return False
            
    def solve(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Resuelve inconsistencias directamente sobre un DataFrame.
        - Infiera tipos semánticos básicos por columna.
        - Seleccione C, PU, MTL por axiomas y máxima puntuación.
        - Reconstruya valores faltantes/inconsistentes.
        - Devuelva DF corregido y tipos semánticos finales.
        """
        if df is None or df.empty:
            return df, []

        columns = list(df.columns)
        basic_types = self._infer_semantic_types_basic(df)
        # Índices de columnas cuantitativas
        quant_indices_map = [i for i, t in enumerate(basic_types) if t == "cuantitativo"]
        if len(quant_indices_map) < 3:
            logger.warning("[MatrixSolver] Menos de 3 columnas cuantitativas; no se aplica corrección.")
            return df, basic_types

        # Construir DataFrame numérico (solo columnas cuantitativas)
        quant_cols = [columns[i] for i in quant_indices_map]
        numeric_df = pd.DataFrame({col: self._to_numeric_series(df[col]) for col in quant_cols})

        # --- FASE 1: Selección de Hipótesis ---
        col_indices_in_numeric_matrix = list(range(len(quant_cols)))
        permutations_indices = list(permutations(col_indices_in_numeric_matrix, 3))
        hypothesis_scores = {p: 0.0 for p in permutations_indices}

        for _, row in numeric_df.iterrows():
            row_list = [None if (pd.isna(v)) else float(v) for v in row.tolist()]
            valids = self._get_valid_hypotheses_for_row(row_list, permutations_indices)
            if len(valids) == 1:
                hypothesis_scores[valids[0]] += 1.0
            elif len(valids) == 2:
                hypothesis_scores[valids[0]] += 0.5
                hypothesis_scores[valids[1]] += 0.5

        if not any(score > 0 for score in hypothesis_scores.values()):
            logger.error("[MatrixSolver] No se encontró hipótesis válida; no se corrige.")
            return df, basic_types
        
        c_idx, pu_idx, mtl_idx = max(hypothesis_scores, key=hypothesis_scores.get)
        c_name = quant_cols[c_idx]
        pu_name = quant_cols[pu_idx]
        mtl_name = quant_cols[mtl_idx]
        logger.debug(f"[MatrixSolver] Roles: C='{c_name}', PU='{pu_name}', MTL='{mtl_name}'")
        # --- FASE 2: Reconstrucción ---
        reconstructed = numeric_df.to_numpy(copy=True)
        # Calcular medianas por columna ignorando NaN
        col_medians = {i: np.nanmedian(reconstructed[:, i]) for i in col_indices_in_numeric_matrix}

        rows_with_two_missing = []
        for i in range(reconstructed.shape[0]):
            c, pu, mtl = reconstructed[i, c_idx], reconstructed[i, pu_idx], reconstructed[i, mtl_idx]
            present = [not np.isnan(v) for v in [c, pu, mtl]]
            missing_count = 3 - sum(present)

            if missing_count >= 2:
                rows_with_two_missing.append(i)
                continue

            # Completar un faltante si es posible
            try:
                if np.isnan(mtl) and (not np.isnan(c)) and (not np.isnan(pu)):
                    reconstructed[i, mtl_idx] = c * pu
                elif np.isnan(pu) and (not np.isnan(mtl)) and (not np.isnan(c)) and c != 0:
                    reconstructed[i, pu_idx] = mtl / c
                elif np.isnan(c) and (not np.isnan(mtl)) and (not np.isnan(pu)) and pu != 0:
                    reconstructed[i, c_idx] = mtl / pu
                else:
                    # Si no faltan, verificar consistencia y corregir la de mayor desviación
                    c, pu, mtl = reconstructed[i, c_idx], reconstructed[i, pu_idx], reconstructed[i, mtl_idx]
                    if not (np.isnan(c) or np.isnan(pu) or np.isnan(mtl)) and not math.isclose(c * pu, mtl, rel_tol=self.arithmetic_tolerance):
                        dev_c = abs(c - col_medians.get(c_idx, c))
                        dev_pu = abs(pu - col_medians.get(pu_idx, pu))
                        dev_mtl = abs(mtl - col_medians.get(mtl_idx, mtl))
                        max_dev = max(dev_c, dev_pu, dev_mtl)
                        if max_dev == dev_c and pu != 0:
                            reconstructed[i, c_idx] = mtl / pu
                        elif max_dev == dev_pu and c != 0:
                            reconstructed[i, pu_idx] = mtl / c
                        else:
                            reconstructed[i, mtl_idx] = c * pu
            except ZeroDivisionError:
                # Si ocurre, deja la fila como está
                pass

        # --- FASE 3: Integración al DF original ---
        corrected_df = df.copy()
        for j, col_name in enumerate(quant_cols):
            # Formatear como string conservando 2 decimales cuando aplique
            def fmt(v):
                if np.isnan(v):
                    return corrected_df[col_name]
                try:
                    return f"{v:.2f}" if not float(v).is_integer() else str(int(round(v)))
                except Exception:
                    return corrected_df[col_name]
            # Asignar con cuidado para no romper celdas no numéricas originales
            series_num = pd.Series(reconstructed[:, j], index=corrected_df.index)
            corrected_df[col_name] = series_num.apply(lambda x: (f"{x:.2f}" if not np.isnan(x) and not float(x).is_integer() else (str(int(round(x))) if not np.isnan(x) else None)))

            # Para celdas que originalmente no eran numéricas, dejamos el valor original si no hubo corrección
            mask_orig_non_num = ~df[col_name].apply(self._is_numeric_like)
            corrected_df.loc[mask_orig_non_num, col_name] = df.loc[mask_orig_non_num, col_name]

        # Tipos semánticos finales con roles
        final_semantic_types = basic_types[:]
        # Mapear índices numéricos a índices originales
        final_semantic_types[quant_indices_map[c_idx]] = "cuantitativo, c"
        final_semantic_types[quant_indices_map[pu_idx]] = "cuantitativo, pu"
        final_semantic_types[quant_indices_map[mtl_idx]] = "cuantitativo, mtl"

        return corrected_df, final_semantic_types

    def _clean_numeric_value(self, value):
        """Limpia símbolos comunes de valores numéricos antes de convertir a float."""
        if not isinstance(value, str):
            return value
        cleaned = value.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
        return cleaned

    def _is_numeric_like(self, v: Any) -> bool:
        try:
            if v is None:
                return False
            cleaned = self._clean_numeric_value(v) if isinstance(v, str) else v
            float(cleaned)
            return True
        except Exception:
            return False

    def _infer_semantic_types_basic(self, df: pd.DataFrame, numeric_ratio_threshold: float = 0.6) -> List[str]:
        types: List[str] = []
        for col in df.columns:
            series = df[col]
            total = len(series)
            if total == 0:
                types.append("texto")
                continue
            numeric_like = sum(1 for v in series if self._is_numeric_like(v))
            types.append("cuantitativo" if (numeric_like / total) >= numeric_ratio_threshold else "texto")
        return types

    def _to_numeric_series(self, series: pd.Series) -> pd.Series:
        def to_float(v):
            if v is None:
                return np.nan
            if isinstance(v, (int, float)):
                return float(v)
            try:
                cleaned = self._clean_numeric_value(v)
                return float(cleaned) if cleaned != "" else np.nan
            except Exception:
                return np.nan
        return series.apply(to_float)

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

    