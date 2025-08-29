# PerfectOCR/core/workers/vectorial_transformation/math_max.py
import pandas as pd # type: ignore
import logging
import time
from itertools import permutations
import math
import numpy as np
from typing import Dict, Any, List, Tuple
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
        self.output = self.enabled_outputs.get("math_max_corrected", False)
        self.total_mtl_tolerance = self.config.get('total_mtl_abs_tolerance', 0.05)
        self.arithmetic_tolerance = self.config.get('row_relative_tolerance', 0.05)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        try:
            start_time = time.time()

            # Recupera el DataFrame estructurado directamente de memoria
            df = manager.get_structured_table()
            if df is None or df.empty:
                logger.warning("[MatrixSolver] No hay tabla estructurada para procesar")
                return False

            corrected_df, final_semantic_types = self.solve(df)
            if self.output:
                self._save_debug_table(manager, context, corrected_df)

            manager.save_structured_table(df=corrected_df, columns=list(corrected_df.columns), semantic_types=final_semantic_types)

            total_time = time.time() - start_time
            logger.info(f"[MatrixSolver] Corrección matemática completada en {total_time:.6f}s,  Se encontraron {len(corrected_df)} filas.")
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
        if df is None or df.empty: # type: ignore
            return df, []

        columns: List[str] = list(df.columns)
        basic_types = self._infer_semantic_types_basic(df)
        quant_indices_map = [i for i, t in enumerate(basic_types) if t == "cuantitativo"]
        if len(quant_indices_map) < 3:
            logger.warning("[MatrixSolver] Menos de 3 columnas cuantitativas; no se aplica corrección.")
            return df, basic_types

        quant_cols = [columns[i] for i in quant_indices_map]
        numeric_df = pd.DataFrame({columns: self._to_numeric_series(df[columns]) for columns in quant_cols})

        # --- FASE 1: Selección de Hipótesis ---
        col_indices_in_numeric_matrix = list(range(len(quant_cols)))
        permutations_indices = list(permutations(col_indices_in_numeric_matrix, 3))
        hypothesis_scores = {p: 0.0 for p in permutations_indices}
            
        for _, row in numeric_df.iterrows():
            row_list: List[float] = [None if (pd.isna(v)) else float(v) for v in row.tolist()]
            
            valid_hypotheses = self._get_valid_hypotheses_for_row(row_list, permutations_indices)
            if len(valid_hypotheses) == 1:
                hypothesis_scores[valid_hypotheses[0]] += 1.0
            elif len(valid_hypotheses) == 2:
                hypothesis_scores[valid_hypotheses[0]] += 0.5
                hypothesis_scores[valid_hypotheses[1]] += 0.5

        if not any(score > 0 for score in hypothesis_scores.values()):
            logger.error("[MatrixSolver] No se encontró hipótesis válida; no se corrige.")
            return df, basic_types
        
        c_idx, pu_idx, mtl_idx = max(hypothesis_scores, key=lambda k: hypothesis_scores[k])
        c_name = quant_cols[c_idx]
        pu_name = quant_cols[pu_idx]
        mtl_name = quant_cols[mtl_idx]
        logger.info(f"[MatrixSolver] Roles: C='{c_name}', PU='{pu_name}', MTL='{mtl_name}'")
        # --- FASE 2: Reconstrucción ---
        reconstructed: np.ndarray[np.float64, Any] = numeric_df.to_numpy(copy=True)
                
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

        final_semantic_types = basic_types[:]
        final_semantic_types[quant_indices_map[c_idx]] = "cuantitativo, c"
        final_semantic_types[quant_indices_map[pu_idx]] = "cuantitativo, pu"
        final_semantic_types[quant_indices_map[mtl_idx]] = "cuantitativo, mtl"

        return corrected_df, final_semantic_types

    def _infer_semantic_types_basic(self, df: pd.DataFrame, numeric_ratio_threshold: float = 0.6) -> List[str]:
        basic_types: List[str] = []
        for columns in df.columns:
            series = df[columns]
            total = len(series)
            if total == 0:
                basic_types.append("texto")
                continue
            numeric_like = sum(1 for v in series if self._is_numeric_like(v))
            basic_types.append("cuantitativo" if (numeric_like / total) >= numeric_ratio_threshold else "texto")
        return basic_types
    
    def _to_numeric_series(self, series: pd.Series) -> pd.Series:
        def to_float(v: Any):
            if v is None:
                return np.nan
            if isinstance(v, (int, float)):
                
                return v
            try:
                cleaned = self._clean_numeric_value(v)
                return cleaned
            except Exception:
                return np.nan
        return series.apply(to_float)
        
    def _clean_numeric_value(self, v: Any) -> float:
        """Limpia símbolos comunes de valores numéricos antes de convertir a float."""
        cleaned = v.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
        cleaned = float(cleaned)
        return cleaned
    
    def _get_valid_hypotheses_for_row(self, row_list: List[float], permutations_indices: List[tuple[int, int, int]]) -> List[tuple[int, int, int]]:
        """Encuentra todas las hipótesis válidas para una sola fila."""
        valid_hypotheses: List[tuple[int, int, int]] = []
        for p_indices in permutations_indices:
            c_idx, pu_idx, mtl_idx = p_indices
            # Asegurarse de que los índices están dentro de los límites de la fila
            if max(c_idx, pu_idx, mtl_idx) >= len(row_list):
                continue
            
            c, pu, mtl = row_list[c_idx], row_list[pu_idx], row_list[mtl_idx]
            
            if c is None or pu is None or mtl is None: # type: ignore
                continue

            if not all(isinstance(v, (int,float)) for v in [c, pu, mtl]):
                continue

            # Axiomas
            if c <= 0 or pu <= 0 or mtl <= 0: continue
            if mtl < c * pu: continue
            if pu < mtl: continue

            if math.isclose(c * pu, mtl, rel_tol=self.arithmetic_tolerance):
                valid_hypotheses.append(p_indices)
        
        return valid_hypotheses
    
    def _is_numeric_like(self, v: Any) -> bool:
        try:
            if v is None:
                return False
            self._clean_numeric_value(v) if isinstance(v, str) else v
            return True
        except Exception:
            return False
            
    def to_float(self, v: Any) -> float:

        return float(v)

    def _save_debug_table(self, manager: DataFormatter, context: Dict[str, Any], corrected_df: pd.DataFrame):
        from services.output_service import save_table
        import os
        header_text: List[str] = []
        if manager and manager.workflow and manager.workflow.all_lines:
            for line_obj in manager.workflow.all_lines.values():
                if line_obj.header_line:
                    for poly_id in line_obj.polygon_ids:
                        if poly_id in manager.workflow.polygons:
                            poly_text = manager.workflow.polygons[poly_id].ocr_text
                            if poly_text:        
                                header_text.append(poly_text)
                    break
        if not header_text:
            header_text = list(corrected_df.columns)
            
        file_name = context.get("image_name")
        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir = os.path.join(path, "math_max")
            file_name = (f"{file_name}math_max_corrected.csv")
            save_table(corrected_df, output_dir, file_name, header_text)
        
        if output_paths:
            logger.info(f"Tabla corregida matemáticamente '{file_name}' guardada en {len(output_paths)} ubicaciones.")