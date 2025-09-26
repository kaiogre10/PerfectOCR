# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import math
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, normalize # type: ignore

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('cos_sim', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)        
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time: float = time.time()
            logger.info("Calculando matriz de similitud")

            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}

            header_line_id = self._find_header_line_id(polygons, all_lines)
            if header_line_id is not None:
                manager.update_header(header_line_id)
                logger.info(f"Linea de encabezado actualizada desde Coseno")
                
                # intentar obtener líneas detectadas por el scanner (DBSCAN)
                tabular_lines: List[str] = manager.get_tabular_lines()
                # normalizar: si manager devuelve lista de dicts extraer ids
                analysis: Dict[str, Dict[str, float]] = context.get("all_features", {})
                logger.info(f"Features recibidos por Scanner: {len(analysis)} líneas")

                # Si hay resultado del scanner, VALIDAR usando estrategia all-vs-all en el intervalo header+1 .. última del scanner
                if tabular_lines:
                    logger.info(f"Validando resultado del scanner con validación coseno all-vs-all ({len(tabular_lines)} líneas reportadas)")
                    validated = self._validate_scanner_interval_all_vs_all(analysis, all_lines, header_line_id, tabular_lines, manager)
                    if validated:
                        total_time = time.time() - start_time
                        logger.info(f"Validación coseno completada en {total_time:.6f}s. Líneas válidas: {len(validated)}")
                        success = manager.save_tabular_lines(validated)
                        if success:
                            logger.info("Lineas guardadas en el manager desde COSENO (validación all-vs-all)")
                            return True
                        else:
                            logger.error("Error al guardar líneas tabulares validadas en el workflow")
                            return False
                    else:
                        logger.info("Validación coseno rechazó las líneas detectadas por el scanner")
                        return False
            else:
                logger.info("Error detectando encabezado")
                return True
        except Exception as e:
            logger.info(f"Error en matriz de similitud coseno: {e}", exc_info=True)
            return False
        
    def _validate_scanner_interval_all_vs_all(self, analysis: Dict[str, Dict[str, float]], all_lines: Dict[str, AllLines], header_line_id: str, tabular_lines: List[str], manager: DataFormatter) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """
        try:
            similarity_threshold = float(self.worker_config.get("similarity_threshold", 0.85))
            min_cluster = int(self.worker_config.get("min_cluster", 2))

            line_ids: List[str] = list(all_lines.keys())
            if header_line_id not in line_ids:
                logger.error("Header no encontrado entre las líneas codificadas")
                return []
            header_idx = line_ids.index(header_line_id)
            
            # Convertir tabular_lines (IDs de línea) a índices numéricos
            tabular_indices = []
            for line_id in tabular_lines:
                if line_id in line_ids:
                    tabular_indices.append(line_ids.index(line_id))
        
            if not tabular_indices:
                logger.error("Ninguna línea tabular encontrada en line_ids")
                return []
                
            last_scanner_idx = max(tabular_indices)

            start_idx = header_idx + 1
            end_idx = last_scanner_idx

            if start_idx > end_idx:
                logger.info("Intervalo para validar vacío (header al final o scanner produjo líneas anteriores al header).")
                return []

            # LOG: Mostrar el intervalo de líneas a validar
            interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
            logger.info(f"Intervalo de validación: líneas {start_idx} a {end_idx} (total: {len(interval_line_ids)} líneas)")
            for i, line_id in enumerate(interval_line_ids):
                line_obj = all_lines.get(line_id)
                line_text = line_obj.text if line_obj else "SIN TEXTO"
                logger.info(f"[{start_idx + i}] {line_id}: '{line_text}'")

            # BLOQUEAR líneas que contienen key_field
            # MontoTotalDocumento, Subtotal, TotalProductos, MontoIVAGeneral, RFCProveedor, FolioDocumento, FechaDocumento, NombreCliente
     
            blocked_line_ids: Set[str] = set()
            if manager.workflow and manager.workflow.polygons:
                logger.debug(f"Total de polígonos disponibles: {len(manager.workflow.polygons)}")
                
                # Obtener IDs de polígonos con key_field (incluyendo HeaderWords)
                blocked_polygon_ids: Set[str] = set()
                for poly_id, polygon in manager.workflow.polygons.items():
                    if polygon.key_field:
                        blocked_polygon_ids.add(poly_id)
                        logger.debug(f"Polígono {poly_id}: key_field='{polygon.key_field}'")
                
                # Buscar líneas que contengan estos polígonos
                for line_id, line_obj in all_lines.items():
                    if line_obj.polygon_ids:
                        line_polygon_ids = set(line_obj.polygon_ids)
                        if line_polygon_ids.intersection(blocked_polygon_ids):
                            blocked_line_ids.add(line_id)
                            logger.info(f"Línea {line_id} bloqueada por contener polígonos con key_field")
            
            logger.info(f"Líneas bloqueadas por key_field: {sorted(blocked_line_ids)}")
            
            # Filtrar el intervalo excluyendo líneas bloqueadas
            blocked_in_interval = [idx for idx in range(start_idx, end_idx + 1) if line_ids[idx] in blocked_line_ids]
            if blocked_in_interval:
                first_blocked = min(blocked_in_interval)
                new_end = first_blocked - 1
                logger.info(f"Cortando intervalo en la primera línea bloqueada: bloqueada_idx={first_blocked}, nuevo end_idx={new_end}")
                end_idx = new_end

            # reconstruir intervalo y comprobar si quedó vacío tras el corte
            if start_idx > end_idx:
                logger.info("Intervalo quedó vacío tras cortar por líneas bloqueadas.")
                return []

            filtered_interval_indices: List[int] = [i for i in range(start_idx, end_idx + 1)]
            logger.info(f"Intervalo filtrado: {len(filtered_interval_indices)} líneas (de {end_idx - start_idx + 1} originales)")

            feature_keys: List[str] = [
                'count_rel',
                'mean_rel',
                'mean_margin',
                'skewness',
                "numeric_count_norm",
                "numeric_frec_rel",
                "numeric_ratio_frec",
                "num_above",
                "num_margin",
                "digit_char_frec",
                "area_norm",
                "norm_wid",
                "width_rel",
                "ratio_area",
                "aspcrat_inv_norm",
                "perimeter_norm",
                "diagonal_norm",
                "compact",
                "prev_xmin_align",
                "prev_xmax_align",
                "next_xmin_align",
                "next_xmax_align",
                "align_prev",
                "align_next",
                "center_aling",
            ]
            
                
            mat_rows: List[List[float]] = []
            candidate_indices: List[int] = []  # Cambio: debe ser List[int] para índices
            candidate_line_ids: List[str] = []

            for idx in filtered_interval_indices:  # Cambio: usar filtered_interval_indices en lugar de interval_line_ids
                line_id = line_ids[idx]  # Obtener el line_id del índice
                features = analysis.get(line_id, {})  # Cambio: buscar por line_id, no por 'aggregate_stats'
                
                if not features:
                    logger.warning(f"No se encontraron features para línea {line_id}")
                    continue
                    
                row: List[float] = [float(features.get(k, 0.0)) for k in feature_keys]
                mat_rows.append(row)
                candidate_indices.append(idx)
                candidate_line_ids.append(line_id)  # Cambio: usar line_id definido

            n = len(mat_rows)
            if n < min_cluster:
                logger.warning("No hay suficientes líneas válidas en el intervalo para validación coseno.")
                # Tomar como línea tabular el intervalo entre el encabezado y la línea bloqueada inmediata
                logger.info("Tomando intervalo entre encabezado y línea bloqueada inmediata como líneas tabulares.")
                interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
                return interval_line_ids
            try:
                X = np.array(mat_rows, dtype=np.float64)
                logger.info(f"{X}")

                # matriz todos contra todos (n x n)
                sims_mat: np.ndarray[Any, np.dtype[np.float64]] = cosine_similarity(X, dense_output=True).astype(dtype=np.float64)
            except Exception as e:
                logger.error(f"Error calculando matriz se similitud: {e}", exc_info=True)
            # Para mejorar la legibilidad, separamos filas y columnas y mostramos la matriz línea por línea
            mean = np.array(sims_mat)
            mean_log = np.mean(mean)
            logger.info(f"Promedio matriz: {mean_log}")
            logger.debug("Matriz de similitud (cosine_similarity):")
            logger.info("Filas/Columnas (en orden): %s", ", ".join(str(lid) for lid in candidate_line_ids))
            matriz_str = "\n".join(
                ["[" + "  ".join(f"{val:7.6f}" for val in row) + "]" for row in sims_mat]
            )
            logger.info("Matriz:\n%s", matriz_str)

            # para cada fila, calcular similitud media con las demás (excluir self)
            mean_sims: List[float] = []
            for i in range(n):
                if n == 1:
                    mean_sims.append(1.0)
                else:
                    mean_val = float((np.sum(sims_mat[i]) - 1.0) / (n - 1))
                    mean_sims.append(mean_val)

            matched_original_indices: List[int] = []
            consecutive_failures = 0
            for mean_sim, orig_idx, lid in zip(mean_sims, candidate_indices, candidate_line_ids):
                if mean_sim >= similarity_threshold:
                    matched_original_indices.append(int(orig_idx))
                    consecutive_failures +=1
                logger.info(f"Línea {lid} idx={orig_idx}: mean_sim={mean_sim:.4f}")

            # incluir header (aunque no participó en la similitud)
            final_indices = [header_idx] + matched_original_indices
            table_line_ids = [line_ids[i] for i in final_indices if i < len(line_ids)]
            return table_line_ids
        except Exception as e:
            logger.error(f"Error validando intervalo del scanner (all-vs-all): {e}", exc_info=True)
            return []

    def _find_header_line_id(self, polygons: Dict[str, Polygons], all_lines: Dict[str, AllLines]) -> Optional[str]:
        """Localiza la line_id del encabezado basada en HeaderWords."""
        try:
            hdr_poly_ids: List[str] = [pid for pid, p in polygons.items() if getattr(p, "key_field", None) == "HeaderWords"]
            if not hdr_poly_ids: 
                return None
            
            hdr_set = set(hdr_poly_ids)
            counts = {lid: len(set(lobj.polygon_ids).intersection(hdr_set)) for lid, lobj in all_lines.items() if lobj.polygon_ids}
            
            if not counts: 
                return None
        
            header_line_id: Optional[str] = max(counts, key=counts.get)
        
            if not header_line_id:
                return None
            else:
                logger.info(f"Header_line_id={header_line_id} (via HeaderWords)")
                return header_line_id
        
        except Exception as e:
            logger.error(f"No hubo encabezado textual por similitud de encabezado: {e}", exc_info=True)
