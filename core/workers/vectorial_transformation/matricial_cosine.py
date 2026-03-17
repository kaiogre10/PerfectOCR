# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines
from core.utils.data_utils import VECTOR_MEAN_DUMMIE, VECTOR_MEDIAN_DUMMIE
from core.utils.math_utils import get_cosine_similarity, density_cluster

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('cos_sim', {})
        self.similarity_threshold: float = worker_config.get("similarity_threshold")
        self.min_cluster = int(worker_config.get("min_cluster"))
        self.dummie_weights = worker_config["dummie_weights"]
        self.emergency_threshold = worker_config.get("emergency_threshold")
        self.eps = float(worker_config.get("eps"))
        self.metric = worker_config.get("metric", "")
        self.output = config.get("table_lines", False)
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        timw9 = time.perf_counter()
        try:
            vectorice = context["vectorice"]
            if not vectorice:
               logger.warning("No hay features disponibles para procesar por que ya se detectaron lineas tabulares")
               return True
    
            analysis: np.ndarray[Any, Any] = context["all_features"]
            table_line_ids: List[str] = self._compare_vectors(manager, analysis)
            if table_line_ids:
                logger.debug(f"RESULTADOS COSENO: {time.perf_counter() - timw9:.6f}s {len(table_line_ids)} líneas tabulares"
                    "\n"f"{table_line_ids}")
                succes = manager.save_tabular_lines(table_line_ids)
                if succes:     
                    logger.debug("Tablas guaradas en el manager desde coseno")
                    if self.output:
                        from services.output_service import save_debug_json
                        return_objects: bool = True
                        tab_info: Dict[str, Any] = manager.get_tabular_lines(return_objects) # type: ignore
                        file_name: str = manager.workflow.metadata.image_name # type: ignore
                        worker_name = context.get("worker_name") or "matrix_cosine"
                        output_paths = context["output_paths"]
                        save_debug_json(output_paths, worker_name, tab_info, file_name)
    
                    return True
        except Exception as e:
            logger.error(f"Error en matriz coseno: {e}", exc_info=True)
        return True

    def _compare_vectors(self, manager: DataFormatter, analysis: np.ndarray[Any, Any]) -> List[str]:
        try:
            start_time = time.perf_counter()
            logger.debug("Calculando matriz de similitud")
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            line_ids: List[str] = [lid.lineal_id for lid in all_lines.values()]
            # tabular_lines: List[str] = manager.get_tabular_lines(False)
            # logger.info(f"Tbular lines: {tabular_lines}")

            tabular_lines: List[str] = self._apply_dbscan_clustering(analysis, manager)

            if tabular_lines:
                table_line_ids = self._validate_scanner_interval_all_vs_all(analysis, tabular_lines, line_ids)
                if table_line_ids:
                    logger.debug(f"Validación coseno completada en {time.perf_counter() - start_time:.6f}s. Líneas: {table_line_ids}")
                    return table_line_ids
            
            header_line_id, footer_line_id = self.get_headers(all_lines)
            if header_line_id > 0:
                table_line_ids = self._fallback_cosine(analysis, line_ids, header_line_id)
                logger.debug(f"Validación coseno con encabezado en {time.perf_counter() - start_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                return table_line_ids
            
            elif footer_line_id > 0:
                table_line_ids = self._fallback_cosine(analysis, line_ids, footer_line_id)
                logger.debug(f"Validación coseno con footer en {time.perf_counter() - start_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                return table_line_ids
                
            else:
                table_line_ids = self._emergency_fallback(analysis, line_ids)
                logger.warning("Método fallback falló, pasando al método de emergencia")
                return table_line_ids
                        
        except Exception as e:
            logger.error(f"Error en matriz de similitud coseno: {e}", exc_info=True)
        return []

    def _validate_scanner_interval_all_vs_all(self, analysis: np.ndarray[Any, Any], tabular_lines: List[str], line_ids: List[str]) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado.
        Poda por los extremos (como "cortar césped") basándose en la media de similitudes, 
        asegurando un intervalo contiguo de salida.
        """            
        # Convertir tabular_lines (IDs de línea) a índices numéricos
        tabular_indices: List[int] = []
        for line_id in tabular_lines:
            if line_id in line_ids:
                tabular_indices.append(line_ids.index(line_id))
                
        # Asegurarse de que están ordenados secuencialmente para buscar el intervalo
        tabular_indices.sort()
        tabular_idx = np.array(tabular_indices, np.int8)
        
        mask = np.isin(analysis[:, 0], tabular_idx, assume_unique=True)
        features_all = np.compress(mask, analysis, 0)
        # Ordenamos las filas según el índice original para estar alineados con tabular_indices
        features_all = features_all[features_all[:, 0].argsort()]
        
        features = np.ascontiguousarray(features_all[:, 1:], np.float32)
        
        timecos0 = time.perf_counter()
        sims_mat_dense = get_cosine_similarity(X=features, dense_output=False)
        logger.debug(f"Coseno realizado en: {time.perf_counter()-timecos0:.10f}'s")

        logger.debug(f"Promedio matriz: {np.mean(sims_mat_dense)}")
        logger.debug("Filas/Columnas (en orden):"
        "\n%s", ", ".join(line_ids[idx] for idx in tabular_indices))
        matriz_str = "\n".join(
            ["[" + "  ".join(f"{float(val):8.7f}" for val in row) + "]" for row in sims_mat_dense]
        )
        logger.debug(
        "\n%s", matriz_str)

        # para cada fila, calcular similitud media con las demás (excluir self)
        mean_sims: List[float] = []
        n = features.shape[0]
        for i in range(n):
            if n == 1:
                mean_sims.append(1.0)
            else:
                mean_val = float((np.sum(sims_mat_dense[i]) - 1.0) / (n - 1)) 
                mean_sims.append(mean_val)

        for mean_sim, orig_idx in zip(mean_sims, tabular_indices):
            lid = line_ids[orig_idx]
            logger.debug(f"Línea {lid} idx={orig_idx}: mean_sim={mean_sim:.6f}")

        # Recorte de intervalo contiguo (Poda por extremos)
        start_pos: int | None = None
        last_success_pos: int | None = None
        consecutive_failures = 0

        for pos, s in enumerate(mean_sims):
            if s >= self.similarity_threshold:
                if start_pos is None:
                    start_pos = pos
                last_success_pos = pos
                consecutive_failures = 0
            else:
                if start_pos is not None:
                    consecutive_failures += 1
                    if consecutive_failures > self.min_cluster:
                        break

        # Si hay validaciones por coseno y encontramos un intervalo válido, devolvemos el contiguo
        if start_pos is not None and last_success_pos is not None:
            # Reconstruir intervalo entre start y last_success usando tabular_indices
            start_idx = tabular_indices[start_pos]
            end_idx = tabular_indices[last_success_pos]
            
            # Devolver TODAS las líneas comprendidas entre start_idx y end_idx de line_ids
            # Garantizando la contigüidad absoluta en base al ID general.
            table_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
            logger.debug(f"Intervalo final podado: {len(table_line_ids)} líneas ({line_ids[start_idx]} a {line_ids[end_idx]}).")
            return table_line_ids

        # Si ninguna línea superó el umbral, activar emergencia desde aquí
        logger.info("Ninguna línea validada por coseno en el intervalo; activando emergencia.")
        return self._emergency_fallback(analysis, line_ids)

    def _fallback_cosine(self, analysis: np.ndarray[Any, Any], line_ids: List[str], cut_idx: int) -> List[str]:
        """
        Fallback: Busca un bloque continuo de líneas tabulares después del encabezado.
        Compara cada línea con la línea de referencia (la primera después del encabezado).
        Tolera un número de fallos consecutivos ('interval') antes de cortar el bloque.
        """
        logger.warning("INICIANDO MÉTODO FALLBACK")
        # La línea de referencia es la que está justo después del encabezado
        ref_line_idx = cut_idx + 1
        if ref_line_idx > (len(line_ids) - self.min_cluster):
            logger.warning("No hay líneas después del encabezado para usar como referencia.")
            return []
        
        mask = (analysis[:, 0] > cut_idx)

        line_cand = np.compress(mask, analysis, 0)
        cand_indx = line_cand[:, 0].astype(np.int32)
        line_cands = np.ascontiguousarray(line_cand[:, 1:], dtype=np.float32)

        ref_vec = line_cands[0].reshape(1, -1)

        sims = get_cosine_similarity(line_cands, ref_vec, dense_output=False)
        sims = np.ravel(sims)  # ← Aplanar a 1D para evitar el TypeError

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

            if consecutive_failures > self.min_cluster:
                logger.info(f"Se superó el margen de error ({self.min_cluster} fallos). Cortando en índice {last_success_idx}.")
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
        logger.warning(f"INICIANDO MÉTODO DE EMERGENCA")
        mean_w, median_w = self.dummie_weights
        
        median_ref_vec = VECTOR_MEDIAN_DUMMIE.reshape(1, -1)
        mean_ref_vec = VECTOR_MEAN_DUMMIE.reshape(1, -1)

        t0 = time.perf_counter()
        dummie_vect = np.row_stack([median_ref_vec, mean_ref_vec])
        analysis = np.ascontiguousarray(analysis[:, 1:], dtype=np.float32)
        sims_comb = get_cosine_similarity(analysis, dummie_vect, dense_output=False)
        # sims_median = calculate_similarity_ref(median_ref_vec, analysis[:, 1:] , dense_output=False)
        # logger.debug(f"Similitudes con Dummie MEDIAN: {sims_median}")

        # sims_mean = calculate_similarity_ref(mean_ref_vec, analysis[:, 1:] , dense_output=False)
        # logger.debug(f"Similitudes con Dummie MEAN: {sims_mean}")

        # # 3. Ponderación de resultados
        # sims_final = (sims_median * median_w) + (sims_mean * mean_w)
        
        sims_final = (sims_comb[:, 0] * median_w) + (sims_comb[:, 1] * mean_w)

        logger.debug(f"Tiempo: {time.perf_counter() - t0}")
        
        logger.debug(f"Promedio de similitud final: {sims_final}")
        logger.debug("Todas las líneas ordenadas: %s", ", ".join(str(lid) for lid in line_ids))
        sims_str = "[" + "  ".join(f"{val}" for val in sims_final) + "]"
        logger.debug("Similitudes de emergencia finales:\n%s", sims_str)

        # Log detallado por línea
        for _, (line_id, sim) in enumerate(zip(line_ids, sims_final)):
            logger.debug(f"{line_id}: Sim: {sim}")
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
            
            logger.debug(f"Cluster '{len(table_line_ids)}' encontrado por fallback de emergencia: {table_line_ids}")
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
                if candidate_indices[j] - candidate_indices[j-1] <= self.min_cluster:
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

    def get_headers(self, all_lines: Dict[str, AllLines]) -> Tuple[int, int]:
        try:
            header_line_id = 0
            footer_line_id = 0
            for line_data in all_lines.values():
                h: int | None = line_data.header_line if line_data.header_line else None
                if h is not None:
                    header_line_id += h
                    
                f: int | None = line_data.footer_line if line_data.footer_line else None
                if f is not None:
                    footer_line_id += f
                    
                if footer_line_id > 0 and footer_line_id > 0:
                    break 
                else:
                    continue
            
            logger.info(f"Header_idx: {header_line_id}, Footer_idx: {footer_line_id}")
            return header_line_id, footer_line_id
        except Exception as e:
            logger.error(f"Error buscando encabezados: {e}")
        return 0, 0
         
    def _apply_dbscan_clustering(self, features_array: np.ndarray[Any, Any], manager: DataFormatter) -> List[str]:
        """Aplica DBSCAN para agrupar líneas similares"""
        all_lines = manager.workflow.all_lines if manager.workflow else {}
        int_line_ids = features_array[:, 0].astype(np.int8)
        features_for_clustering = np.ascontiguousarray(features_array[:, 1:], dtype=np.float32)
        # logger.info(f"Features: {features_array}")

        # Crear un diccionario que mapea line_index (int) a line_id (str)
        index_to_id: Dict[int, str] = {}
        for line_id, line_obj in all_lines.items():
            # Extraer número de "line_X"
            idx = line_obj.line_index  # Esto ya es un int
            index_to_id[idx] = line_id
        
        # Obtener line_ids correspondientes
        line_ids = [index_to_id.get(int(idx), f"line_{int(idx)}") for idx in int_line_ids]
        # timedbscan = time.perf_counter()
        labels: np.ndarray[Any, Any] = density_cluster(features_for_clustering, self.eps, self.min_cluster, self.metric)
        # logger.info(f"Tiempo de DBSCAN: {time.perf_counter() - timedbscan:.6f}'s")
        
        unique_labels: List[int] = [l for l in set(labels) if l != -1]
        if not unique_labels:
            logger.warning("DBSCAN: No se encontraron clusters válidos.")
            return []
                
        cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
        main_cluster = max(cluster_sizes, key=cluster_sizes.get)

        table_line_ids: List[str] = [line_ids[i] for i, label in enumerate(labels) if label == main_cluster]
        # logger.info(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}, table_lines: {table_line_ids}")
        selected_indices = [all_lines[line_id].line_index for line_id in table_line_ids if line_id in all_lines]

        if not selected_indices:
            logger.warning("No se encontraron índices para table_line_ids.")
            return table_line_ids

        # Paso 2: Calcular el rango
        min_idx, max_idx = min(selected_indices), max(selected_indices)

        # Paso 3: Generar todos los índices en ese rango
        full_range_indices = range(min_idx, max_idx + 1)

        # Paso 4: Mapear de vuelta a line_id (str)
        full_range_line_ids = [index_to_id.get(idx, f"line_{idx:04d}") for idx in full_range_indices]

        # logger.info(f"DSSCAN: Rango de líneas tabulares: {full_range_line_ids}, total: {len(full_range_line_ids)}")

        return full_range_line_ids
