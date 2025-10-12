# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Set
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from scipy.sparse import csr_matrix # type: ignore

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
            if context.get("all_features", {}) is None:
                logger.debug("Las líneas tabulares y encabezado ya fueron detectados, no se ejecuta validación coseno.")
                return True

            start_time: float = time.time()
            logger.debug("Calculando matriz de similitud")

            tabular_lines: List[str] = manager.get_tabular_lines()
            analysis: Dict[str, Dict[str, float]] = context.get("all_features", {})
            logger.debug(f"Features recibidos por Scanner: {len(analysis)} líneas")

            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            line_ids: List[str] = list(all_lines.keys())
            header_line_id = [lid for lid, l in all_lines.items() if getattr(l, "header_line", not None)]
            header_line_id = header_line_id[0] if header_line_id else None
            if header_line_id is None:
                return False
            
            header_idx = line_ids.index(header_line_id)
            # Fallback si no hay líneas tabulares o están antes del encabezado
            fallback_needed = (
                not tabular_lines or
                (header_line_id and tabular_lines and line_ids.index(tabular_lines[0]) < line_ids.index(header_line_id))
            )

            if tabular_lines and not fallback_needed:
                logger.debug(f"Validando resultado del scanner con validación coseno all-vs-all ({len(tabular_lines)} líneas reportadas)")
                table_line_ids = self._validate_scanner_interval_all_vs_all(analysis, tabular_lines, manager, header_line_id, line_ids, header_idx)
                if table_line_ids:
                    total_time = time.time() - start_time
                    logger.info(f"Validación coseno completada en {total_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                    success = manager.save_tabular_lines(table_line_ids)
                    if success:
                        logger.debug("Lineas guardadas en el manager desde COSENO (validación all-vs-all)")
                        return True
                    else:
                        logger.error("Error al guardar líneas tabulares validadas en el workflow")
                        return False
                else:
                    logger.debug("Validación coseno rechazó las líneas detectadas por el scanner")
                    return False
            else:
                logger.warning("Ejecutando fallback: buscando líneas tabulares por similitud coseno con el encabezado")
                table_line_ids = self._fallback_cosine(analysis, header_line_id, line_ids, header_idx)
                if table_line_ids:
                    total_time = time.time() - start_time
                    logger.info(f"Fallback coseno completado en {total_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                    success = manager.save_tabular_lines(table_line_ids)
                    if success:
                        logger.debug("Lineas guardadas en el manager desde COSENO (fallback header)")
                        return True
                    else:
                        logger.error("Error al guardar líneas tabulares validadas en el workflow (fallback)")
                        return False
        except Exception as e:
            logger.debug(f"Error en matriz de similitud coseno: {e}", exc_info=True)
        return False

    def _validate_scanner_interval_all_vs_all(self, analysis: Dict[str, Dict[str, float]], tabular_lines: List[str], manager: DataFormatter, header_line_id: str, line_ids: List[str], header_idx: int) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """
        all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}

        similarity_threshold: float = self.worker_config.get("similarity_threshold")
        min_cluster = int(self.worker_config.get("min_cluster"))
        
        logger.warning(f"Usando header_line_id proporcionado por manager: {header_line_id}")

        # Convertir tabular_lines (IDs de línea) a índices numéricos
        tabular_indices: List[int] = []
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
            logger.error("Intervalo para validar vacío (header al final o scanner produjo líneas anteriores al header).")

        # LOG: Mostrar el intervalo de líneas a validar
        interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
        logger.debug(f"Intervalo de validación: líneas {start_idx} a {end_idx} (total: {len(interval_line_ids)} líneas)")
        for i, line_id in enumerate(interval_line_ids):
            line_obj = all_lines.get(line_id)
            line_text = line_obj.text if line_obj else "SIN TEXTO"
            logger.debug(f"[{start_idx + i}] {line_id}: '{line_text}'")

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
                        logger.debug(f"Línea {line_id} bloqueada por contener polígonos con key_field")
            
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
            logger.debug("Intervalo quedó vacío tras cortar por líneas bloqueadas.")
            return []

        filtered_interval_indices: List[int] = [i for i in range(start_idx, end_idx + 1)]
        logger.debug(f"Intervalo filtrado: {len(filtered_interval_indices)} líneas (de {end_idx - start_idx + 1} originales)")

        feature_keys: List[str] = [
                'count_rel',
                'mean_rel',
                'mean_margin',
                'skewness',
                "numeric_count_norm",
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
        candidate_indices: List[int] = []  
        candidate_line_ids: List[str] = []

        for idx in filtered_interval_indices:  
            line_id = line_ids[idx] 
            if not analysis:
                logger.error("Error extrayendo features")
            features = analysis.get(line_id, {}) 
            
            if not features:
                logger.warning(f"No se encontraron features para línea {line_id}")
                continue
                
            row: List[float] = [float(features.get(k, 0.0)) for k in feature_keys]
            mat_rows.append(row)
            candidate_indices.append(idx)
            candidate_line_ids.append(line_id)

        n = len(mat_rows)
        if n < min_cluster:
            logger.warning("No hay suficientes líneas válidas en el intervalo para validación coseno.")
            logger.debug("Tomando intervalo entre encabezado y línea bloqueada inmediata como líneas tabulares.")
            interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
            return interval_line_ids
            
        try:
            X = csr_matrix(mat_rows, dtype=np.float64)
            timecos0 = time.perf_counter()
            sims_mat = cosine_similarity(X, dense_output=False) # type: ignore
            logger.info(f"Coseno realizado en: {time.perf_counter()-timecos0:.10f}s")
        except Exception as e:
            logger.error(f"Error calculando matriz se similitud: {e}", exc_info=True)

        # Convertir la matriz dispersa a densa para mostrarla
        sims_mat_dense: np.ndarray[Any, Any] = sims_mat.toarray() # type: ignore
        mean_log = np.mean(sims_mat_dense) # type: ignore
        logger.info(f"Promedio matriz: {mean_log}")
        logger.debug("Matriz de similitud (cosine_similarity):")
        logger.info("Filas/Columnas (en orden): %s", ", ".join(str(lid) for lid in candidate_line_ids))
        matriz_str = "\n".join(
            ["[" + "  ".join(f"{val:7.6f}" for val in row) + "]" for row in sims_mat_dense] # type: ignore
        )
        logger.info("Matriz:\n%s", matriz_str)

        # para cada fila, calcular similitud media con las demás (excluir self)
        mean_sims: List[float] = []
        for i in range(n):
            if n == 1:
                mean_sims.append(1.0)
            else:
                mean_val = float((np.sum(sims_mat[i]) - 1.0) / (n - 1)) # type: ignore
                mean_sims.append(mean_val)

        matched_original_indices: List[int] = []
        consecutive_failures = 0
        for mean_sim, orig_idx, lid in zip(mean_sims, candidate_indices, candidate_line_ids):
            if mean_sim >= similarity_threshold:
                matched_original_indices.append(int(orig_idx))
                consecutive_failures +=1
            logger.info(f"Línea {lid} idx={orig_idx}: mean_sim={mean_sim:.4f}")

        table_line_ids = [line_ids[i] for i in matched_original_indices if i < len(line_ids)]
        return table_line_ids

    def _fallback_cosine(self, analysis: Dict[str, Dict[str, float]], header_line_id: str, line_ids: List[str], header_idx: int) -> List[str]:
        """
        Fallback: Busca un bloque continuo de líneas tabulares después del encabezado.
        Compara cada línea con la línea de referencia (la primera después del encabezado).
        Tolera un número de fallos consecutivos ('interval_margin') antes de cortar el bloque.
        """
        similarity_threshold: float = self.worker_config.get("similarity_threshold")
        min_cluster: int = int(self.worker_config.get("min_cluster"))
        interval_margin: int = int(self.worker_config.get("interval_margin", 1))

        feature_keys: List[str] = [
            'count_rel', 'mean_rel', 'mean_margin', 'skewness', "numeric_count_norm",
            "numeric_ratio_frec", "num_above", "num_margin", "digit_char_frec", "area_norm",
            "norm_wid", "width_rel", "ratio_area", "aspcrat_inv_norm", "perimeter_norm",
            "diagonal_norm", "compact", "prev_xmin_align", "prev_xmax_align", "next_xmin_align",
            "next_xmax_align", "align_prev", "align_next", "center_aling",
        ]

        # La línea de referencia es la que está justo después del encabezado
        ref_line_idx = header_idx + 1
        if ref_line_idx >= len(line_ids):
            logger.warning("No hay líneas después del encabezado para usar como referencia.")
            return [header_line_id]

        ref_line_id = line_ids[ref_line_idx]
        ref_features = analysis.get(ref_line_id, {})
        if not ref_features:
            logger.error(f"No se encontraron features para la línea de referencia {ref_line_id}")
            return [header_line_id]

        ref_vec = np.array([float(ref_features.get(k, 0.0)) for k in feature_keys]).reshape(1, -1)

        # Preparar datos para cálculo de similitud en bloque
        candidate_rows: List[List[float]] = []
        candidate_line_ids: List[str] = []
        for idx in range(header_idx + 2, len(line_ids)):
            line_id = line_ids[idx]
            features = analysis.get(line_id, {})
            if not features:
                logger.warning(f"Sin features para línea {line_id}, se omitirá en fallback.")
                continue
            
            row: List[float] = [float(features.get(k, 0.0)) for k in feature_keys]
            candidate_rows.append(row)
            candidate_line_ids.append(line_id)

        if not candidate_rows:
            logger.warning("No hay líneas candidatas después de la línea de referencia.")
            return [header_line_id, ref_line_id]

        # Calcular similitud y registrar la matriz
        X = np.array(candidate_rows, dtype=np.float64)
        sims = cosine_similarity(ref_vec, X)[0]

        logger.info(f"Promedio de similitud con línea de referencia '{ref_line_id}': {np.mean(sims):.6f}")
        logger.debug("Vector de similitud (ref vs candidates):")
        logger.info("Candidatas (en orden): %s", ", ".join(str(lid) for lid in candidate_line_ids))
        sims_str = "[" + "  ".join(f"{val:7.6f}" for val in sims) + "]"
        logger.info("Similitudes:\n%s", sims_str)

        validated_lines = [header_line_id, ref_line_id]
        last_success_idx = ref_line_idx
        consecutive_failures = 0

        # Iterar sobre los resultados de similitud
        for i, sim in enumerate(sims):
            current_idx = line_ids.index(candidate_line_ids[i])
            logger.debug(f"Comparando con ref {ref_line_id}: línea {candidate_line_ids[i]} (idx={current_idx}), sim={sim:.4f}")

            if sim >= similarity_threshold:
                consecutive_failures = 0
                last_success_idx = current_idx
            else:
                consecutive_failures += 1

            if consecutive_failures > interval_margin:
                logger.info(f"Se superó el margen de error ({interval_margin} fallos). Cortando en índice {last_success_idx}.")
                break
        
        # Construir el resultado final como un bloque continuo hasta el último éxito
        final_tabular_lines = [line_ids[i] for i in range(header_idx, last_success_idx + 1)]

        # Forzar 'min_cluster' si el resultado es muy pequeño
        if len(final_tabular_lines) < (min_cluster + 1): # +1 por el header
             logger.warning(f"El bloque continuo ({len(final_tabular_lines)}) es menor que min_cluster. No se forzará para mantener continuidad.")

        logger.info(f"Fallback encontró {len(final_tabular_lines)} líneas tabulares continuas.")
        return final_tabular_lines