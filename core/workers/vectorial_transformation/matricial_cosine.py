# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons
from core.utils.data_utils import VECTOR_DUMMIE
from core.utils.math_utils import get_cosine_similarity, density_cluster, calculate_features, cosine_similarity_matrix, mean_cosine_per_row
from services.output_service import save_table_values

dummie_vect = VECTOR_DUMMIE.reshape(1, -1)

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('cos_sim', {})
        self.similarity_threshold: float = worker_config.get("similarity_threshold")
        self.min_cluster = worker_config.get("min_cluster", 1)
        self.emergency_threshold = worker_config.get("emergency_threshold")
        self.eps = worker_config.get("eps")
        self.metric = worker_config.get("metric", "")
        self.output = config.get("table_lines", False)
        self.output_features = config.get("features")
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        timw9 = time.perf_counter()
        try:
            vectorice = context["vectorice"]
            if not vectorice:
               logger.warning("No hay features disponibles para procesar por que ya se detectaron lineas tabulares")
               return True

            table_line_ids: List[str] = self._compare_vectors(manager)
            if table_line_ids:
                logger.debug(f"RESULTADOS COSENO: {time.perf_counter() - timw9:.6f}s {len(table_line_ids)} líneas tabulares"
                    "\n"f"{table_line_ids}")
                if manager.save_tabular_lines(table_line_ids):
                    logger.debug("Tablas guaradas en el manager desde coseno")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error en matriz coseno: {e}", exc_info=True)
        return True

    def _compare_vectors(self, manager: DataFormatter) -> List[str]:
        try:
            polygons_dict: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            all_lines_dict: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            if not polygons_dict or not all_lines_dict:
                return []
                
            img_dims: Tuple[int, int] = manager.workflow.metadata.img_dims if manager.workflow else (0, 0)
            
            line_ids = sorted(all_lines_dict.keys())
            sorted_lines = [all_lines_dict[k] for k in line_ids]

            analysis = calculate_features(sorted_lines, polygons_dict, img_dims)

            if self.output_features:
                all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
                line_id = np.array([id.lineal_id for id in all_lines.values()], np.str_)
                features_to_ind = analysis[:, 1:].astype(np.str_)
                features_id = np.column_stack([line_id, features_to_ind])
                file_name = manager.workflow.metadata.image_name
                save_table_values(file_name, features_id, "vectorizer")
                
            tabular_lines = [line.lineal_id for line in sorted_lines if line.lineal_id in line_ids and line.tabular_line]
            
            if tabular_lines:
                line_idx = np.array([line.line_index for line in sorted_lines if line.lineal_id in tabular_lines], np.uint8)
                if self.validate_similiraity_all_vs_all(analysis, line_idx):
                    return tabular_lines
                else:
                    tabular_array = self.cosine_dummies(analysis, line_idx)
                    if tabular_array[:, 0].size == len(tabular_lines):
                        return tabular_lines
                    new_tabular_idx = tabular_array[:, 0].tolist()
                    new_tabular_lines: List[str] = [tabular_lines[i] for i in new_tabular_idx]
                    return new_tabular_lines
            else:
                line_idx = np.array(line_ids, np.uint8)
                tabular_lines = self.cosine_dummies(analysis, line_ids)

                if tabular_lines:
                    # tabular_lines = self.similiraity_all_vs_all(analysis, tabular_lines, line_ids)
                    return tabular_lines
                else:
                    logger.info("Sin lineas tabulares, DBSCAN como soporte")
                    return self.scanner_clustering(analysis, manager)
                                    
        except Exception as e:
            logger.error(f"Error en matriz de similitud coseno: {e}", exc_info=True)
        return []

    def validate_similiraity_all_vs_all(self, analysis: np.ndarray[Any, Any], line_idx: np.ndarray[Any, np.dtype[np.uint8]]) -> bool:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado.
        Poda por los extremos (como "cortar césped") basándose en la media de similitudes, 
        asegurando un intervalo contiguo de salida.
        """
        if line_idx.size < 2:
            return True
        
        mask = np.isin(analysis[:, 0], line_idx, assume_unique=True)
        features_all = np.compress(mask, analysis, 0)
        # Ordenamos las filas según el índice original para estar alineados con tabular_indices
        features_all = features_all[features_all[:, 0].argsort()]
                
        features = np.ascontiguousarray(features_all[:, 1:], np.float32)
                
        # timecos1 = time.perf_counter()
        sims_mat_dense = cosine_similarity_matrix(features)
        # logger.info(f"Matriz NUMPY obtenida en: {time.perf_counter()-timecos1:.10f}'s")

        # para cada fila, calcular similitud media con las demás
        mean_sims = mean_cosine_per_row(sims_mat_dense)
        # mean_idx = np.column_stack([line_idx, mean_sims])
        # logger.info("PROMEDIO:\n"f"{np.column_stack([np.arange(n), mean_idx])}")
        return bool(np.all(mean_sims > self.similarity_threshold, keepdims=False))

    def cosine_dummies(self, analysis: np.ndarray[Any, Any], line_ids: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        Fallback de emergencia optimizado. Compara todas las líneas del documento contra vectores DUMMIE
        usando una similitud ponderada para encontrar el mejor cluster de líneas tabulares.
        """
        # logger.warning(f"INICIANDO MÉTODO DE EMERGENCA")
        
        t0 = time.perf_counter()
        # has_kf = analysis[:, -2] < 1
        analysis = analysis[line_ids]
        analysi = np.ascontiguousarray(analysis[:, 1:], dtype=np.float32)
   
        sims_final = get_cosine_similarity(analysi, dummie_vect, dense_output=False)

        logger.debug(f"Tiempo: {time.perf_counter() - t0}")
        # logger.info("SIMILITUDES:\n"f"{np.array2string(np.column_stack([line_ids, sims_final]), precision=4)}")
        
        n = line_ids.size
        sims_idx = np.where(sims_final > self.emergency_threshold)[0]
        consecutive_idx_size = sims_idx.size
        
        # logger.info("TAMAÑOS"
        # "n:\n"f"{n} | consecutive_idx: {consecutive_idx_size}")
        if consecutive_idx_size == n:
            # logger.info("LINEAS SUPERARON UMBRAL DE SIMILITUD")
            return line_ids
            
        d = np.diff(sims_idx)
        # logger.info("Consecutive:\n"f"{sims_idx}, {sims_idx.size}\n"f"D: {d}, {d.size}")
        cuts = np.where(d > self.min_cluster)[0]
        if cuts.size == 0:
            # logger.info("LINEAS SUPERARON UMBRAL REDUCIDO")
            return line_ids
        
        # logger.info(f"CUTS: {cuts}, SHAPE: {cuts.shape}")
        cutted_idx = np.arange((cuts[0]+1), dtype=np.uint8)
        mean_idx = sims_final[cutted_idx]
        lines_ids = line_ids[cutted_idx]
        # logger.info("CUTTED:\n"f"{np.column_stack([lines_ids, mean_idx])}, SHAPE: {mean_idx.shape[0]}")
        # tabular_ids = line_ids[0:mean_idx.shape[0]]
        # logger.info(f"{tabular_ids}")
        return np.column_stack([cutted_idx, lines_ids]).astype(np.uint8)
        
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
         
    def scanner_clustering(self, features_array: np.ndarray[Any, Any], manager: DataFormatter) -> List[str]:
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

        logger.info(f"DBSCAN: Rango de líneas tabulares: {full_range_line_ids}, total: {len(full_range_line_ids)}")

        return full_range_line_ids
