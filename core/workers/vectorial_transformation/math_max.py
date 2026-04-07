# PerfectOCR/core/workers/vectorial_transformation/math_max.py
import pandas as pd # type: ignore
import logging
import time
from itertools import permutations
import math
import numpy as np
from typing import Dict, Any, List, Tuple, cast
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from services.output_service import save_debug_table

logger = logging.getLogger(__name__)

class MatrixSolver(VectorizationAbstractWorker):
    """
    Resuelve inconsistencias matemáticas en una tabla estructurada usando un
    enfoque de puntuación global y validación final contra un total.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('math_max', {})
        self.total_mtl_tolerance = worker_config.get('total_mtl_abs_tolerance')
        self.arithmetic_tolerance = worker_config.get('row_relative_tolerance')
        self.output = config.get("math_max_corrected", False)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        try:
            start_time = time.time()

            table_matrix = cast(List[List[Dict[str, Any]]], context.get("table_matrix", []))
            if not table_matrix:
                logger.error("No hay table_matrix en contexto para procesar")
                return False

            table_columns = cast(List[str], context.get("table_columns", []))
            if not table_columns:
                inferred_width = len(table_matrix[0]) if table_matrix else 0
                table_columns = [f"col_{i}" for i in range(inferred_width)]

            df = self._table_matrix_to_dataframe(table_matrix, table_columns)
            if df.empty:
                logger.error("La table_matrix no contiene filas/columnas válidas")
                return False

            # Log simple de cómo recibe la tabla (antes de corregir)
            logger.info("Tabla recibida para corrección matemática:\n" + df.to_string(index=False))

            corrected_df, final_semantic_types = self.solve(df, table_matrix)
            if df.equals(corrected_df):
                logger.info("No se corrigió tabla; se conserva versión original")
                corrected_df = df.copy()
            
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

            # Log simple de cómo queda la tabla ya corregida
            logger.info("Tabla tras corrección matemática:\n" + corrected_df.to_string(index=False))

            manager.save_structured_table(df=corrected_df, columns=list(corrected_df.columns), semantic_types=final_semantic_types)
            context["table_matrix_corrected"] = corrected_df.fillna("").astype(str).values.tolist()
            context["table_semantic_types"] = final_semantic_types

            total_time = time.time() - start_time
            logger.info(f"Corrección matemática completada en {total_time:.6f}s, Se encontraron {len(corrected_df)} filas.")
            return True
        except Exception as e:
            logger.error(f"Error en MatrixSolver.vectorize: {e}", exc_info=True)
            return False
            
    def solve(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]]) -> Tuple[pd.DataFrame, List[str]]:
        """Resuelve inconsistencias directamente sobre un DataFrame.
        - Infiera tipos semánticos básicos por columna.
        - Seleccione C, PU, MTL por axiomas y máxima puntuación.
        - Reconstruya valores faltantes/inconsistentes.
        - Devuelva DF corregido y tipos semánticos finales.
        """

        columns: List[str] = list(df.columns)
        
        # --- NUEVA LÓGICA: Inferencia de tipos basada en fragmentación y contenido ---
        # Identificamos columnas candidatas a cuantitativas por su contenido dominante
        basic_types = self._infer_semantic_types_basic(df, table_matrix)
        logger.info(f"BASI TIPES iniciales: {basic_types}")
        
        quant_indices_map = [i for i, t in enumerate(basic_types) if t == "cuantitativo"]
        if len(quant_indices_map) < 2:
            logger.info("Menos de 2 columnas cuantitativas; no se aplica corrección.")
            return df, basic_types

        quant_cols = [columns[i] for i in quant_indices_map]
        numeric_df = pd.DataFrame({col: self._to_numeric_series(df[col]) for col in quant_cols})

        # --- FASE 1: Selección de Hipótesis (Voto Global) ---
        col_indices_in_numeric_matrix = list(range(len(quant_cols)))
        permutations_indices = list(permutations(col_indices_in_numeric_matrix, 3))
        hypothesis_scores = {p: 0.0 for p in permutations_indices}
            
        for _, row in numeric_df.iterrows():
            row_list: List[float | None] = [None if (pd.isna(v)) else float(v) for v in row.tolist()]
            
            valid_hypotheses = self._get_valid_hypotheses_for_row(row_list, permutations_indices)
            if len(valid_hypotheses) == 1:
                hypothesis_scores[valid_hypotheses[0]] += 1.0
            elif len(valid_hypotheses) == 2:
                hypothesis_scores[valid_hypotheses[0]] += 0.5
                hypothesis_scores[valid_hypotheses[1]] += 0.5

        if not any(score > 0 for score in hypothesis_scores.values()):
            logger.error("No se encontró hipótesis válida; no se corrige.")
            return df, basic_types
        
        c_idx, pu_idx, mtl_idx = max(hypothesis_scores, key=lambda k: hypothesis_scores[k])
        
        # Índices globales de los roles fijos
        global_c_idx = quant_indices_map[c_idx]
        global_pu_idx = quant_indices_map[pu_idx]
        global_mtl_idx = quant_indices_map[mtl_idx]
        numeric_role_indices = {global_c_idx, global_pu_idx, global_mtl_idx}
        
        logger.info(f"Roles Globales: C='{columns[global_c_idx]}', PU='{columns[global_pu_idx]}', MTL='{columns[global_mtl_idx]}'")

        # --- FASE 2: Reconstrucción Numérica ---
        reconstructed: np.ndarray[Any, np.dtype[np.float32]] = numeric_df.to_numpy(dtype=np.float32, copy=True)
        col_medians = {i: np.nanmedian(reconstructed[:, i]) for i in col_indices_in_numeric_matrix}

        for i in range(reconstructed.shape[0]):
            c, pu, mtl = reconstructed[i, c_idx], reconstructed[i, pu_idx], reconstructed[i, mtl_idx]
            present = [not np.isnan(v) for v in [c, pu, mtl]]
            missing_count = 3 - sum(present)

            if missing_count >= 2: continue

            try:
                if np.isnan(mtl) and (not np.isnan(c)) and (not np.isnan(pu)):
                    reconstructed[i, mtl_idx] = c * pu
                elif np.isnan(pu) and (not np.isnan(mtl)) and (not np.isnan(c)) and c != 0:
                    reconstructed[i, pu_idx] = mtl / c
                elif np.isnan(c) and (not np.isnan(mtl)) and (not np.isnan(pu)) and pu != 0:
                    reconstructed[i, c_idx] = mtl / pu
                else:
                    c, pu, mtl = reconstructed[i, c_idx], reconstructed[i, pu_idx], reconstructed[i, mtl_idx]
                    if not (np.isnan(c) or np.isnan(pu) or np.isnan(mtl)) and not math.isclose(c * pu, mtl, rel_tol=self.arithmetic_tolerance):
                        dev_c = abs(c - col_medians.get(c_idx, c))
                        dev_pu = abs(pu - col_medians.get(pu_idx, pu))
                        dev_mtl = abs(mtl - col_medians.get(mtl_idx, mtl))
                        max_dev = max(dev_c, dev_pu, dev_mtl)
                        if max_dev == dev_c and pu != 0: reconstructed[i, c_idx] = mtl / pu
                        elif max_dev == dev_pu and c != 0: reconstructed[i, pu_idx] = mtl / c
                        else: reconstructed[i, mtl_idx] = c * pu
            except ZeroDivisionError: pass

        # --- FASE 3: Reasignación de Información (De afuera hacia adentro) ---
        corrected_df = df.copy()
        # Buscamos la columna de descripción (centro/texto)
        desc_col_idx = next((i for i, t in enumerate(basic_types) if t == "texto"), -1)
        if desc_col_idx == -1:
            desc_col_idx = next((i for i in range(len(columns)) if i not in numeric_role_indices), -1)

        for i in range(len(df)):
            final_vals = {idx: str(df.iloc[i, idx]) if pd.notna(df.iloc[i, idx]) else "" for idx in range(len(columns))}
            
            # Recolectamos información extra desplazada hacia el centro (desc_col_idx)
            left_info = []   # Info de columnas a la izquierda de la descripción
            right_info = []  # Info de columnas a la derecha de la descripción

            for role_global, local_idx in [(global_c_idx, c_idx), (global_pu_idx, pu_idx), (global_mtl_idx, mtl_idx)]:
                original_text = final_vals[role_global].strip()
                validated_val = reconstructed[i, local_idx]
                
                if not np.isnan(validated_val):
                    formatted_val = self._format_num(validated_val)
                    
                    # Extraemos lo que NO es el número validado
                    if original_text and original_text != formatted_val:
                        # Limpieza selectiva: solo removemos el número exacto si es idéntico o parte del token
                        # pero preservamos el resto (UMD, prefijos, etc)
                        clean_text = original_text.replace(formatted_val, "").strip()
                        # Backup: si es .00, a veces el OCR lo leyó pegado
                        if validated_val.is_integer():
                            clean_text = clean_text.replace(str(int(validated_val)), "").strip()
                        
                        if clean_text:
                            if role_global < desc_col_idx:
                                left_info.append(clean_text)
                            else:
                                right_info.append(clean_text)
                    
                    final_vals[role_global] = formatted_val
                elif original_text:
                    # Sin validación matemática, movemos todo el contenido al centro para no borrarlo
                    if role_global < desc_col_idx:
                        left_info.append(original_text)
                    else:
                        right_info.append(original_text)
                    final_vals[role_global] = ""

            # Ensamblamos la descripción (Afuera -> Adentro)
            if desc_col_idx != -1:
                current_desc = final_vals[desc_col_idx]
                # Orden: [Info de la izquierda] + [Descripción Original] + [Info de la derecha]
                parts = []
                if left_info: parts.extend(left_info)
                if current_desc: parts.append(current_desc)
                if right_info: parts.extend(right_info)
                
                final_vals[desc_col_idx] = " ".join(parts).strip()

            for idx in range(len(columns)):
                corrected_df.iloc[i, idx] = final_vals[idx]

        final_semantic_types = basic_types[:]
        final_semantic_types[global_c_idx] = "cuantitativo, c"
        final_semantic_types[global_pu_idx] = "cuantitativo, pu"
        final_semantic_types[global_mtl_idx] = "cuantitativo, mtl"

        return corrected_df, final_semantic_types

    def _format_num(self, x: float) -> str:
        """Formatea números para el DataFrame final."""
        if np.isnan(x): return ""
        return f"{x:.2f}" if not float(x).is_integer() else str(int(round(x)))

    def _infer_semantic_types_basic(self, df: pd.DataFrame, table_matrix: List[List[Dict[str, Any]]], numeric_ratio_threshold: float = 0.3) -> List[str]:
        """Infiere tipos semánticos priorizando semantic_clasification y luego contenido textual."""
        basic_types: List[str] = []
        for col_idx, columns in enumerate(df.columns):
            series = df[columns]
            # Filtrar valores vacíos o solo espacios
            non_empty = [v for v in series if v and str(v).strip() and str(v).strip() not in ['', 'nan', 'NaN', 'None']]
            semantic_votes = self._collect_column_semantics(table_matrix, col_idx)
            quantitative_votes = semantic_votes.get(2, 0)
            text_like_votes = semantic_votes.get(0, 0) + semantic_votes.get(-1, 0) + semantic_votes.get(-2, 0)
            total_votes = sum(semantic_votes.values())

            if total_votes > 0 and quantitative_votes > text_like_votes:
                basic_types.append("cuantitativo")
                continue
            
            if len(non_empty) == 0:
                basic_types.append("texto")
                continue
            
            # Contar valores numéricos sobre valores NO vacíos
            numeric_like = sum(1 for v in non_empty if self._is_numeric_like(v))
            
            # Detectar patrones monetarios ($ o valores con formato de precio)
            has_currency_pattern = any('$' in str(v) or '€' in str(v) for v in non_empty)
            
            # Si tiene patrón monetario Y al menos 30% son números, es cuantitativo
            # O si más del 30% de valores NO vacíos son numéricos
            if has_currency_pattern and numeric_like > 0:
                basic_types.append("cuantitativo")
            elif (numeric_like / len(non_empty)) >= numeric_ratio_threshold:
                basic_types.append("cuantitativo")
            else:
                basic_types.append("texto")
        return basic_types

    def _collect_column_semantics(self, table_matrix: List[List[Dict[str, Any]]], col_idx: int) -> Dict[int, int]:
        votes: Dict[int, int] = {}
        for row in table_matrix:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            semantic_values = cell.get("semantic_clasification", [])
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
    
    def _get_valid_hypotheses_for_row(self, row_list: List[float | None], permutations_indices: List[tuple[int, int, int]]) -> List[tuple[int, int, int]]:
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
            