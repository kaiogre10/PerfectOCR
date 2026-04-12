# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import numpy as np
import time
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines
from core.utils.data_utils import VECTOR_MEAN_DUMMIE, VECTOR_MEDIAN_DUMMIE
from core.utils.math_utils import get_cosine_similarity, density_cluster
from services.output_service import save_debug_json

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('cos_sim', {})
        self.similarity_threshold: float = worker_config.get("similarity_threshold")
        self.min_cluster = int(worker_config.get("min_cluster", 1))
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
                logger.info(f"RESULTADOS COSENO: {time.perf_counter() - timw9:.6f}s {len(table_line_ids)} líneas tabulares"
                    "\n"f"{table_line_ids}")
                succes = manager.save_tabular_lines(table_line_ids)
                if succes:     
                    logger.debug("Tablas guaradas en el manager desde coseno")
                    if self.output:
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
            logger.debug("Calculando matriz de similitud")
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            line_ids: List[str] = [lid.lineal_id for lid in all_lines.values()]
            check_tabular_lines = [lid.tabular_line for lid in all_lines.values()]
            if not any(check_tabular_lines):
                logger.debug("Sin lineas tabulares, DBSCAN como soporte")
                tabular_lines: List[str] = self._apply_dbscan_clustering(analysis, manager)

                if tabular_lines:
                    tabular_lines = self._validate_scanner_interval_all_vs_all(analysis, tabular_lines, line_ids)
                    return tabular_lines
                else:
                    tabular_lines = self._emergency_fallback(analysis, line_ids)
                    return tabular_lines
            else:
                tabular_lines = manager.get_tabular_lines(False) #type: ignore
                return self._validate_scanner_interval_all_vs_all(analysis, tabular_lines, line_ids)
                                    
        except Exception as e:
            logger.error(f"Error en matriz de similitud coseno: {e}", exc_info=True)
        return []

    def _validate_scanner_interval_all_vs_all(self, analysis: np.ndarray[Any, Any], tabular_lines: List[str], line_ids: List[str]) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado.
        Poda por los extremos (como "cortar césped") basándose en la media de similitudes, 
        asegurando un intervalo contiguo de salida.
        """
        if 2 >= len(tabular_lines):
            return tabular_lines
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
            logger.info(f"Intervalo final podado: {len(table_line_ids)} líneas ({line_ids[start_idx]} a {line_ids[end_idx]}).")
            return table_line_ids

        # Si ninguna línea superó el umbral, activar emergencia desde aquí
        logger.info("Ninguna línea validada por coseno en el intervalo; activando emergencia.")
        return self._emergency_fallback(analysis, line_ids)

    def _emergency_fallback(self, analysis: np.ndarray[Any, Any], line_ids: List[str]) -> List[str]:
        """
        Fallback de emergencia optimizado. Compara todas las líneas del documento contra vectores DUMMIE
        usando una similitud ponderada para encontrar el mejor cluster de líneas tabulares.
        """
        # logger.warning(f"INICIANDO MÉTODO DE EMERGENCA")
        mean_w, median_w = self.dummie_weights
        
        median_ref_vec = VECTOR_MEDIAN_DUMMIE.reshape(1, -1)
        mean_ref_vec = VECTOR_MEAN_DUMMIE.reshape(1, -1)

        t0 = time.perf_counter()
        dummie_vect = np.row_stack([median_ref_vec, mean_ref_vec])
        analysi = np.ascontiguousarray(analysis[:, 1:], dtype=np.float32)
        sims_comb = get_cosine_similarity(analysi, dummie_vect, dense_output=False)
        
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
                logger.warning("Ninguna línea superó el umbral de emergencia. Usando fallback.")
                top_k = self.min_cluster
                top_k = min(top_k, len(sims_final))

                # top_indices ordenados por similitud desc (ancla = primero)
                top_indices = np.argsort(sims_final)[::-1][:top_k].astype(np.int32)

                return self._fallback_cosine(analysis, line_ids, top_indices.tolist())

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
            
    def _fallback_cosine(self, analysis: np.ndarray[Any, Any], line_ids: List[str], top_indices: List[int]) -> List[str]:
        """
        Fallback sin encabezados: fuerza un intervalo tabular continuo.
        - Ancla = top_indices[0] (mayor similitud).
        - Ref_vec = promedio de features de los top K (K = min_cluster).
        - Expansión bidireccional con margen de fallo self.min_cluster.
        - Si no entra nadie (intervalo no crece), devuelve el intervalo entre min(topK) y max(topK).
        """
        if len(line_ids) < 3:
            return []

        # Features sin columna índice
        analysis_feat = np.ascontiguousarray(analysis[:, 1:], dtype=np.float32)
        N = analysis_feat.shape[0]
        if N == 0:
            return []

        if not top_indices:
            # No hay ancla; no se puede hacer nada razonable
            return []

        # Sanitizar top_indices a rango válido
        top_indices = [int(i) for i in top_indices if 0 <= int(i) < N]
        if not top_indices:
            return []

        anchor_idx = int(top_indices[0])

        # Si min_cluster == 1: forzar al menos la línea ancla
        top_k = int(self.min_cluster) if int(self.min_cluster) > 0 else 1
        if top_k == 1:
            return [line_ids[anchor_idx]]

        # Tomar top_k (o menos si no alcanza)
        top_k = min(top_k, len(top_indices))
        top_k_idx = sorted(top_indices[:top_k])  # orden por posición (para intervalo fallback)

        # Vector de referencia: promedio de los top_k
        ref_vec = np.mean(analysis_feat[top_k_idx], axis=0, keepdims=True)  # (1, D)

        # Similitud de todas las líneas contra ref_vec
        sims_ref = get_cosine_similarity(analysis_feat, ref_vec, dense_output=False)
        sims_ref = np.ravel(sims_ref)

        # Expansión bidireccional desde el ancla
        start = anchor_idx
        fail = 0
        i = anchor_idx - 1
        while i >= 0:
            if sims_ref[i] >= self.similarity_threshold:
                start = i
                fail = 0
            else:
                fail += 1
                if fail > self.min_cluster:
                    break
            i -= 1

        end = anchor_idx
        fail = 0
        i = anchor_idx + 1
        while i < N:
            if sims_ref[i] >= self.similarity_threshold:
                end = i
                fail = 0
            else:
                fail += 1
                if fail > self.min_cluster:
                    break
            i += 1

        # Si no creció nada, devolver intervalo entre top_k_idx
        if start == end:
            start = int(top_k_idx[0])
            end = int(top_k_idx[-1])

        return [line_ids[i] for i in range(start, end + 1)]
        
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
         
    def _apply_dbscan_clustering(self, features_array: np.ndarray[Any, Any], manager: DataFormatter) -> List[str]:
        """Aplica DBSCAN para agrupar líneas similares"""
        all_lines = manager.workflow.all_lines if manager.workflow else {}
        int_line_ids = features_array[:, 0].astype(np.int8)
        features_for_clustering = np.ascontiguousarray(features_array[:, 1:], dtype=np.float32)

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
        # logger.debug(f"Tiempo de DBSCAN: {time.perf_counter() - timedbscan:.6f}'s")
        
        unique_labels: List[int] = [l for l in set(labels) if l != -1]
        if not unique_labels:
            logger.warning("DBSCAN: No se encontraron clusters válidos.")
            return []
                
        cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
        best_label = None
        best_score = -1e9
        best_density = -1.0
        for label in unique_labels:
            idxs = [i for i, lab in enumerate(labels) if lab == label]
            if not idxs:
                continue
            idxs.sort()
            span = (idxs[-1] - idxs[0] + 1)
            density = len(idxs) / span
            # “tabularidad” esperada: mayor = mejor (si tabular ~ 1 y no-tabular ~ -1)
            score = np.mean(features_for_clustering[idxs])
            if (score > best_score) or (score == best_score and density > best_density):
                best_label = label
                best_score = score
                best_density = density
        main_cluster = best_label

        table_line_ids: List[str] = [line_ids[i] for i, label in enumerate(labels) if label == main_cluster]
        logger.debug(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}, table_lines: {table_line_ids}")
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

        logger.info(f"DSSCAN: Rango de líneas tabulares: {full_range_line_ids}, total: {len(full_range_line_ids)}")

        return full_range_line_ids
