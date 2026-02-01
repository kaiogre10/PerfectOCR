# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Set, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines
from core.utils.math_utils import cosine_similarity_global, calculate_similarity_ref

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('cos_sim', {})
        self.similarity_threshold: float = worker_config.get("similarity_threshold")
        self.min_cluster = int(worker_config.get("min_cluster"))
        self.interval_margin: int = int(worker_config.get("interval"))
        self.dummie_weights = worker_config["dummie_weights"]
        self.emergency_threshold = worker_config.get("emergency_threshold")
        self.output = config.get("table_lines", False)
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            analysis: np.ndarray[Any, Any] = context["all_features"]
            if analysis.size == 0:
                logger.warning("No hay features disponibles para procesar por que ya se detectaron lineas tabulares")
                return True            
            table_line_ids: List[str] = self._compare_vectors(manager, analysis)
            if table_line_ids:
                succes = manager.save_tabular_lines(table_line_ids)
                if succes:
                    
                    logger.info("Tablas guaradas en el manager desde coseno")
                    if self.output:
                        from services.output_service import save_debug_json
                        return_objects: bool = True
                        tab_info: Dict[str, Any] = manager.get_tabular_lines(return_objects) # type: ignore
                        file_name: str = manager.workflow.metadata.image_name # type: ignore
                        worker_name = context.get("worker_name", {})
                        output_paths = context.get("output_paths", [])
                        save_debug_json(output_paths, worker_name, tab_info, file_name)
    
                    return True
        except Exception as e:
            logger.error(f"Error en matriz coseno: {e}", exc_info=True)
        return True

    def _compare_vectors(self, manager: DataFormatter, analysis: np.ndarray[Any, Any]) -> List[str]:
        try:
            start_time: float = time.time()
            logger.debug("Calculando matriz de similitud")
            
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            line_ids: List[str] = [lid.lineal_id for lid in all_lines.values()]
            _, header_line_id = self.get_headers(all_lines)

            if not header_line_id:
                table_line_ids = self._emergency_fallback(analysis, line_ids)
                total_time = time.time() - start_time
                logger.debug(f"Validación coseno completada en {total_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                return table_line_ids
                
            else:
                return_objects: bool = False
                tabular_lines: List[str] = manager.get_tabular_lines(return_objects) # type: ignore
                logger.info(f"TABULAR LINES: {tabular_lines}")
                header_idx = line_ids.index(header_line_id)
                                
                if tabular_lines:
                    logger.debug(f"Validando resultado del scanner con validación coseno all-vs-all ({len(tabular_lines)} líneas reportadas)")
                    table_line_ids = self._validate_scanner_interval_all_vs_all(analysis, tabular_lines, manager, header_line_id, line_ids, header_idx, all_lines)
                    if table_line_ids:
                        total_time = time.time() - start_time
                        logger.info(f"Validación coseno completada en {total_time:.6f}s. Líneas: {table_line_ids}")
                        return table_line_ids
                else:
                    logger.warning("Ejecutando fallback: buscando líneas tabulares por similitud coseno con el encabezado")
                    table_line_ids = self._fallback_cosine(analysis, line_ids, header_idx)
                    if table_line_ids:
                        total_time = time.time() - start_time
                        logger.info(f"Fallback coseno completado en {total_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                        return table_line_ids
                    else:
                        table_line_ids = self._emergency_fallback(analysis, line_ids)
                        logger.warning("Método fallback falló, pasando al método de emergencia")
                        return table_line_ids
        except Exception as e:
            logger.error(f"Error en matriz de similitud coseno: {e}", exc_info=True)
        return []

    def _validate_scanner_interval_all_vs_all(self, analysis: np.ndarray[Any, Any], tabular_lines: List[str], manager: DataFormatter, header_line_id: str, line_ids: List[str], header_idx: int, all_lines: Dict[str, AllLines]) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """
        if line_ids.index(tabular_lines[0]) < line_ids.index(header_line_id):
            return []
            
        # Convertir tabular_lines (IDs de línea) a índices numéricos
        tabular_indices: List[int] = []
        for line_id in tabular_lines:
            if line_id in line_ids:
                tabular_indices.append(line_ids.index(line_id))
        
        if not tabular_indices:
            logger.error("Ninguna línea tabular encontrada en line_ids")
            return self._emergency_fallback(analysis, line_ids)
            
        last_scanner_idx = max(tabular_indices)

        start_idx = header_idx + 1
        end_idx = last_scanner_idx + 1

        if start_idx > end_idx:
            logger.error("Intervalo para validar vacío (header al final o scanner produjo líneas anteriores al header).")
            return self._emergency_fallback(analysis, line_ids)
        
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
            
        logger.debug(f"Líneas bloqueadas por key_field: {blocked_line_ids}")
        
        # Filtrar el intervalo excluyendo líneas bloqueadas
        blocked_in_interval = [idx for idx in range(start_idx, end_idx + 1) if line_ids[idx] in blocked_line_ids]
        if blocked_in_interval:
            first_blocked = min(blocked_in_interval)
            new_end = first_blocked - 1
            logger.debug(f"Cortando intervalo en la primera línea bloqueada: bloqueada_idx={first_blocked}, nuevo end_idx={new_end}")
            end_idx = new_end

        # reconstruir intervalo y comprobar si quedó vacío tras el corte
        if start_idx > end_idx:
            logger.debug("Intervalo quedó vacío tras cortar por líneas bloqueadas.")
            return self._emergency_fallback(analysis, line_ids)

        mask = (analysis[:, 0] > start_idx) & (analysis[:, 0] < end_idx)
        feature = np.compress(mask, analysis, 0).astype(np.float32)
        candidate_indices = feature[:, 0].copy().astype(np.int32)
        features = feature[: ,1:]
            
        n = features.shape[0]
        
        timecos0 = time.perf_counter()
        sims_mat_dense = cosine_similarity_global(features, dense_output=False)
        logger.debug(f"Coseno realizado en: {time.perf_counter()-timecos0:.10f}s")
            
        # Convertir la matriz dispersa a densa para mostrarla
        mean_log = np.mean(sims_mat_dense) 
        logger.debug(f"Promedio matriz: {mean_log}")
        logger.debug("Filas/Columnas (en orden): %s", ", ".join(str(lid) for lid in  line_ids))
        matriz_str = "\n".join(
            ["[" + "  ".join(f"{val:7.6f}" for val in row) + "]" for row in sims_mat_dense]
        )
        logger.debug("Matriz:\n%s", matriz_str)

        # para cada fila, calcular similitud media con las demás (excluir self)
        mean_sims: List[float] = []
        for i in range(n):
            if n == 1:
                mean_sims.append(1.0)
            else:
                mean_val = float((np.sum(sims_mat_dense[i]) - 1.0) / (n - 1)) # type: ignore
                mean_sims.append(mean_val)

        matched_original_indices: List[int] = []
        consecutive_failures = 0
        for mean_sim, orig_idx, lid in zip(mean_sims, candidate_indices,  line_ids):
            if mean_sim > self.similarity_threshold:
                matched_original_indices.append(orig_idx)
                consecutive_failures += 1
            logger.debug(f"Línea {lid} idx={orig_idx}: mean_sim={mean_sim:.6f}")

        # Si hay validaciones por coseno, devolver todo el intervalo hasta la última validada
        if matched_original_indices:
            last_valid_idx = max(matched_original_indices)
            final_end_idx = min(last_valid_idx, end_idx)  # respeta corte por key_field
            table_line_ids = [line_ids[i] for i in range(start_idx, final_end_idx + 1)]
            logger.debug(f"Intervalo asignado por coseno hasta último validado (idx={final_end_idx}): {len(table_line_ids)} líneas")
            return table_line_ids

        # Si ninguna línea superó el umbral, activar emergencia desde aquí
        logger.info("Ninguna línea validada por coseno en el intervalo; activando emergencia.")
        return self._emergency_fallback(analysis, line_ids)

    def _fallback_cosine(self, analysis: np.ndarray[Any, Any], line_ids: List[str], header_idx: int) -> List[str]:
        """
        Fallback: Busca un bloque continuo de líneas tabulares después del encabezado.
        Compara cada línea con la línea de referencia (la primera después del encabezado).
        Tolera un número de fallos consecutivos ('interval') antes de cortar el bloque.
        """
        logger.warning("INICIANDO MÉTODO FALLBACK")
        # La línea de referencia es la que está justo después del encabezado
        ref_line_idx = header_idx + 1
        if ref_line_idx > (len(line_ids) - self.min_cluster):
            logger.warning("No hay líneas después del encabezado para usar como referencia.")
            return []
        
        mask = (analysis[:, 0] > header_idx)

        line_cand = np.compress(mask, analysis, 0)
        cand_indx = line_cand[:, 0].copy().astype(np.int32)
        line_cands = line_cand[:, 1:]

        ref_vec = line_cands[0].reshape(1, -1)

        sims = calculate_similarity_ref(line_cands, ref_vec, dense_output=False)

        logger.debug(f"Promedio de similitud con línea de referencia '{ref_vec}': {np.mean(sims):.6f}")
        logger.debug("Candidatas (en orden): %s", ", ".join(str(lid) for lid in cand_indx))
        sims_str = "[" + "  ".join(f"{val:7.6f}" for val in sims) + "]"
        logger.debug("Similitudes:\n%s", sims_str)

        last_success_idx = ref_line_idx
        consecutive_failures = 0

        # Iterar sobre los resultados de similitud
        for i, sim in enumerate(sims):
            # Convertir el índice numérico a ID de línea string para buscarlo en line_ids
            candidate_line_id = line_ids[cand_indx[i]]
            current_idx = cand_indx[i]
            logger.debug(f"ref {ref_vec}: línea {candidate_line_id}, sim={sim:.6f}")

            if sim > self.similarity_threshold:
                consecutive_failures = 0
                last_success_idx = current_idx
            else:
                consecutive_failures += 1

            if consecutive_failures > self.interval_margin:
                logger.info(f"Se superó el margen de error ({self.interval_margin} fallos). Cortando en índice {last_success_idx}.")
                break
        
        # Construir el resultado final como un bloque continuo hasta el último éxito
        final_tabular_lines = [line_ids[i] for i in range(ref_line_idx, last_success_idx + 1)]

        logger.debug(f"Fallback encontró {len(final_tabular_lines)} líneas tabulares continuas.")
        return final_tabular_lines

    def _emergency_fallback(self, analysis: np.ndarray[Any, Any], line_ids: List[str]) -> List[str]:
        """
        Fallback de emergencia optimizado. Compara todas las líneas del documento contra vectores DUMMIE
        usando una similitud ponderada para encontrar el mejor cluster de líneas tabulares.
        """
        from core.utils.data_utils import VECTOR_MEAN_DUMMIE, VECTOR_MEDIAN_DUMMIE
        logger.warning(f"INICIANDO MÉTODO DE EMERGENCA")
        mean_w, median_w = self.dummie_weights

        median_ref_vec = VECTOR_MEDIAN_DUMMIE.reshape(1, -1)
        logger.info(f"{analysis.shape}")
        
        sims_median = calculate_similarity_ref(analysis[:, 1:], median_ref_vec, dense_output=False)
        logger.debug(f"Similitudes con Dummie MEDIAN: {sims_median}")

        mean_ref_vec = VECTOR_MEAN_DUMMIE.reshape(1, -1)
        sims_mean = calculate_similarity_ref(analysis[:, 1:], mean_ref_vec, dense_output=False)
        logger.debug(f"Similitudes con Dummie MEAN: {sims_mean}")

        # 3. Ponderación de resultados
        sims_final = (sims_median * median_w) + (sims_mean * mean_w)
        
        logger.debug(f"Promedio de similitud final ponderada: {np.mean(sims_final):.6f}")
        logger.debug("Todas las líneas ordenadas: %s", ", ".join(str(lid) for lid in line_ids))
        sims_str = "[" + "  ".join(f"{val:7.6f}" for val in sims_final) + "]"
        logger.debug("Similitudes de emergencia finales:\n%s", sims_str)

        # Log detallado por línea
        for _, (line_id, sim) in enumerate(zip(line_ids, sims_final)):
            logger.debug(f"{line_id}: Sim: {sim:7.4f}")
        try:
            matched_indices = [idx for idx, sim in enumerate(sims_final) if sim > self.similarity_threshold]

            # Si no hay coincidencias, intentar con el umbral de emergencia más bajo
            if not matched_indices:
                logger.warning(f"Ninguna línea superó el umbral de {self.similarity_threshold}. Intentando con umbral de emergencia de {self.emergency_threshold}.")
                matched_indices = [idx for idx, sim in enumerate(sims_final) if sim > self.emergency_threshold]

            if not matched_indices:
                logger.warning("Ninguna línea superó el umbral de emergencia. No se encontraron clusters.")
                return []

            # Obtener las line_ids que pasaron el umbral
            candidate_line_ids = [line_ids[i] for i in matched_indices]
            
            # Ordenar por line_id (ascendente)
            sorted_candidates = sorted(candidate_line_ids, key=lambda x: line_ids.index(x))
            
            # Encontrar el cluster más grande que respete min_cluster e interval
            table_line_ids = self._find_best_cluster(sorted_candidates, line_ids)
            
            logger.info(f"Cluster '{len(table_line_ids)}' encontrado por fallback de emergencia: {table_line_ids}")
            return table_line_ids
        except Exception as e:
            logger.error(f"Error en falback de emergencia: {e}", exc_info=True)
            return []
        
    def _find_best_cluster(self, sorted_candidates: List[str], line_ids: List[str]) -> List[str]:
        """Encuentra el mejor cluster respetando min_cluster e interval y devuelve todas las líneas del intervalo."""
        if len(sorted_candidates) < self.min_cluster:
            logger.warning(f"No hay suficientes candidatos '{len(sorted_candidates)}' para min_cluster: '{self.min_cluster}'")
            return []
        
        candidate_indices = [line_ids.index(lid) for lid in sorted_candidates]
        
        best_start = None
        best_end = None
        best_size = 0

        for i in range(len(candidate_indices)):
            start_idx = candidate_indices[i]
            end_idx = start_idx
            current_size = 1

            for j in range(i + 1, len(candidate_indices)):
                if candidate_indices[j] - candidate_indices[j-1] <= self.interval_margin:
                    end_idx = candidate_indices[j]
                    current_size += 1
                else:
                    break

            if current_size > self.min_cluster and current_size > best_size:
                best_start = start_idx
                best_end = end_idx
                best_size = current_size

        if best_start is not None and best_end is not None:
            # Devuelve todas las líneas entre best_start y best_end (inclusive)
            return line_ids[best_start:best_end+1]
        else:
            return []

    def get_headers(self, all_lines: Dict[str, AllLines]) -> Tuple[int, str]:
        try:
            for line_id, line_data in all_lines.items():
                h: int | None = line_data.header_line if line_data.header_line else None
                if h is not None:
                    header_line_id = line_id
                    # logger.info(f"H: {h}, id: {line_id}")
                    return h, header_line_id
        except Exception as e:
            logger.error(f"Error buscando encabezados: {e}")
        return 0, ""