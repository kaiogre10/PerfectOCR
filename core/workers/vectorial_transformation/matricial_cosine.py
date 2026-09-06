# core/workers/vectorial_transformation/matricial_cosine.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from domain.abstract_worker import VectorizationAbstractWorker
from domain.data_formatter import DataFormatter
from utils.math_utils import get_cosine_similarity, calculate_features, cosine_similarity_matrix, mean_cosine_per_row
from services.output_service import save_table_values, serialize_arrays

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    __slots__ = ("similarity_threshold", "min_cluster", "tolerance_sim", "emergency_threshold", "eps", "metric", "min_internal_sim", "output_features", "output", "training_data")
    def __init__(self, project_root: str, config: Dict[str, Any]):
        super().__init__(config, project_root)
        worker_config = config.get('cos_sim', {})
        self.similarity_threshold: float = worker_config.get("similarity_threshold")
        self.min_cluster = worker_config.get("min_cluster")
        self.tolerance_sim = worker_config.get("tolerance_sim")
        self.emergency_threshold = worker_config.get("emergency_threshold")
        self.eps = worker_config.get("eps")
        self.metric = worker_config.get("metric", "")
        self.min_internal_sim = worker_config.get("min_internal_sim")
        
        self.output = config.get("table_lines", False)
        self.output_features = config.get("features")
        self.training_data = config.get("training_data")
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        timw9 = time.perf_counter()
        try:
            vectorice = context["vectorice"]
            if not vectorice:
               logger.warning("No hay features disponibles para procesar por que ya se detectaron lineas tabulares")
               return True

            table_line_ids, array_table = self._compare_vectors(manager)
            if table_line_ids:
                if manager.save_tabular_lines(table_line_ids):
                    logger.debug(f"{len(table_line_ids)} líneas tab encontradas en: {time.perf_counter() - timw9:.6f}'s")
                    if self.training_data:
                        serialize_arrays(array_table[:, 1:])
                    return True
                return False
        except Exception as e:
            logger.error(f"Error en matriz coseno: '{e}'", exc_info=True)
        return True

    def _compare_vectors(self, manager: DataFormatter) -> Tuple[List[str], Optional[np.ndarray[Any, np.dtype[np.float32]]]]:
        try:
            polygons_dict = manager.workflow.polygons if manager.workflow else {}
            all_lines_dict = manager.workflow.all_lines if manager.workflow else {}
            if not polygons_dict or not all_lines_dict:
                return [], None
            
            metadata = manager.workflow.metadata if manager.workflow else None
            img_dims: Tuple[int, int] = metadata.img_dims if metadata is not None else (0, 0)
            
            line_ids = sorted(all_lines_dict.keys())
            sorted_lines = [all_lines_dict[k] for k in line_ids]

            analysis = calculate_features(sorted_lines, polygons_dict, img_dims)
            all_idxs = np.asarray(np.arange(len(line_ids), dtype=np.uint8), dtype=np.uint8)
            if self.output_features:
                line_id = np.array([id.lineal_id for id in all_lines_dict.values()], np.str_)
                features_to_ind = analysis[:, 1:].astype(np.str_)
                features_id = np.column_stack([line_id, features_to_ind])
                file_name = metadata.image_name if metadata is not None else ""
                save_table_values(file_name, features_id)
                
            tabular_lines = [line.lineal_id for line in sorted_lines if line.lineal_id in line_ids and line.tabular_line]
            
            if tabular_lines:
                line_idx = np.asarray([line.line_index for line in sorted_lines if line.lineal_id in tabular_lines], dtype=np.uint8)
                # logger.info(f"TABLES IDX: {line_idx}")
                if self.validate_similiraity_all_vs_all(analysis, line_idx):
                    # logger.info("VALIDACIÓN GLOBAL PASADA")
                    return tabular_lines, (None if not self.training_data else analysis[line_idx])
                else:
                    tabular_array = self.cosine_dummies(analysis, line_idx)
                    # logger.info(f"DUMMIES IDX: {tabular_array}")
                    if tabular_array.size > 0 and tabular_array.size == len(tabular_lines):
                        # logger.info(f"DUMMIES TABULAREXACTO: {tabular_lines}")
                        return tabular_lines, (None if not self.training_data else analysis[tabular_array])
                    else:
                        new_tabular_idx: List[int] = all_idxs[tabular_array].tolist()
                        new_tabular_lines = [line_ids[i] for i in new_tabular_idx]
                        # logger.info(f"NEW DUMMIES tab: {new_tabular_lines}, {new_tabular_idx}")
                        return new_tabular_lines, (None if not self.training_data else analysis[tabular_array])
            else:
                tabular_array = self.cosine_dummies(analysis, all_idxs)
                # if tabular_array.size < 1:
                    # logger.info("Sin lineas tabulares, DBSCAN como soporte")
                    # return self.scanner_clustering(analysis, manager), (None if not self.training_data else analysis[tabular_array])

                new_tabular_idx = tabular_array.tolist()
                return [line_ids[i] for i in new_tabular_idx], (None if not self.training_data else analysis[tabular_array])
                                    
        except Exception as e:
            logger.error(f"Error en matriz de similitud coseno: {e}", exc_info=True)
        return [], None

    def validate_similiraity_all_vs_all(self, analysis: np.ndarray[Any, Any], line_idx: np.ndarray[Any, np.dtype[np.uint8]]):
        """Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado. Si todos son válidos, la validaciónes corrtecta"""
        if self.min_cluster >= line_idx.size:
            return True

        features_all = analysis[line_idx]
        # Ordenamos las filas según el índice original para estar alineados con tabular_indices
        features_all = features_all[features_all[:, 0].argsort()]
                
        features = np.ascontiguousarray(features_all[:, 1:], dtype=np.float32)
                
        # timecos1 = time.perf_counter()
        sims_mat_dense = cosine_similarity_matrix(features)
        # logger.info(f"Matriz NUMPY obtenida en: {time.perf_counter()-timecos1:.10f}'s")

        # para cada fila, calcular similitud media con las demás
        mean_sims = mean_cosine_per_row(sims_mat_dense)
        # logger.info("PROMEDIO ALL X ALL:\n"f"{np.column_stack([line_idx, mean_sims])}")
        return np.all(mean_sims > self.similarity_threshold, keepdims=False)

    def cosine_dummies(self, analysis: np.ndarray[Any, Any], line_ids: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Compara todas las líneas del documento contra vectores DUMMIE, usando una similitud ponderada para encontrar el mejor cluster de líneas tabulares."""
        # t0 = time.perf_counter()
        analysis = analysis[line_ids]
        analysis = np.ascontiguousarray(analysis[:, 1:], dtype=np.float32)
        sims_final = get_cosine_similarity(analysis)
        # logger.debug(f"Tiempo: {time.perf_counter() - t0}")
        # logger.info("SIMILITUD DUMMIE:\n"f"{np.array2string(np.column_stack([line_ids, sims_final]), precision=4)}")

        if sims_final[0] < self.min_internal_sim:
            # logger.info(f"PRIMERA LINEA RUIDOSA: {sims_final[0]}")
            line_ids = line_ids[1:]
            sims_final = sims_final[1:]

        sim_idx = np.where(sims_final > self.similarity_threshold)[0]                       # Índices donde se superó el umbral de similitud
        if sim_idx.size < 1:
            # logger.warning(f"SE USARA EL UMBRAL DE SEGURIDAD: '{self.emergency_threshold}'")
            sims_idx = np.where(sims_final > self.emergency_threshold)[0]
        else:
            sims_idx = sim_idx

        abs_idx = line_ids[sims_idx]
        # logger.info("\n"f"sims_idx: {sims_idx}\n"f"ABS IDX: {abs_idx}")

        deltas = np.ediff1d(sims_idx, to_begin=0)
        cuts_mask = np.where((deltas - 1) > self.tolerance_sim)[0]
        # logger.info("\n"f"DELTAS: {deltas}\n"f"MAKS: {cuts_mask}")
        if cuts_mask.size < 1:
            end_idx = line_ids[-1] if self.tolerance_sim > (line_ids[-1] - abs_idx[-1]) else abs_idx[-1]
            start_idx = abs_idx[0]
            cutted_idx = np.arange(start=start_idx, stop=(end_idx + 1), dtype=np.uint8)
            # logger.info(f"early cutted_idx: {cutted_idx}, start: {start_idx} end: {end_idx}")
            return cutted_idx

        cutted_idx = np.arange(cuts_mask[-1], dtype=np.uint8)
        # logger.info(f"cutted_idx: {cutted_idx}")
        return line_ids[cutted_idx]
        
    def _find_best_cluster(self, sorted_candidates: List[str], line_ids: List[str]) -> List[str]:
        """Encuentra el mejor cluster respetando min_cluster e interval y devuelve todas las líneas del intervalo."""
        total_cands = len(sorted_candidates)
        if total_cands < self.min_cluster:
            logger.warning(f"No hay suficientes candidatos '{len(sorted_candidates)}' para min_cluster: '{self.min_cluster}'")
            return []
        
        candidate_indices = [line_ids.index(lid) for lid in sorted_candidates]
        best_start = None
        best_end = None
        best_size = 0

        for i in range(total_cands):
            start_idx = candidate_indices[i]
            end_idx = start_idx
            current_size = 1

            for j in range(i + 1, total_cands):
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
         
    # def scanner_clustering(self, features_array: np.ndarray[Any, Any], manager: DataFormatter) -> List[str]:
    #     """Aplica DBSCAN para agrupar líneas similares"""
    #     logger.warning("DBSCAN PARA FALLBACK")
    #     all_lines = manager.workflow.all_lines if manager.workflow else {}
    #     if not all_lines:
    #         return [
                
    #         ]
    #     int_line_ids = features_array[:, 0].astype(np.int8)
    #     features_for_clustering = np.ascontiguousarray(features_array[:, 1:], dtype=np.float32)

    #     # Crear un diccionario que mapea line_index (int) a line_id (str)
    #     index_to_id: Dict[int, str] = {}
    #     for line_id, line_obj in all_lines.items():
    #         # Extraer número de "line_X"
    #         idx = line_obj.line_index  # Esto ya es un int
    #         index_to_id[idx] = line_id
        
    #     # Obtener line_ids correspondientes
    #     line_ids = [index_to_id.get(int(idx), f"line_{int(idx)}") for idx in int_line_ids]
    #     # timedbscan = time.perf_counter()
    #     labels: np.ndarray[Any, Any] = density_cluster(features_for_clustering, self.eps, self.min_cluster, self.metric)
    #     # logger.debug(f"Tiempo de DBSCAN: {time.perf_counter() - timedbscan:.6f}'s")
        
    #     unique_labels: List[int] = [l for l in set(labels) if l != -1]
    #     if not unique_labels:
    #         logger.warning("DBSCAN: No se encontraron clusters validos.")
    #         return []
                
    #     cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
    #     best_label = None
    #     best_score = -1e9
    #     best_density = -1.0
    #     for label in unique_labels:
    #         idxs = [i for i, lab in enumerate(labels) if lab == label]
    #         if not idxs:
    #             continue
    #         idxs.sort()
    #         span = (idxs[-1] - idxs[0] + 1)
    #         density = len(idxs) / span
    #         # “tabularidad” esperada: mayor = mejor (si tabular ~ 1 y no-tabular ~ -1)
    #         score = np.mean(features_for_clustering[idxs])
    #         if (score > best_score) or (score == best_score and density > best_density):
    #             best_label = label
    #             best_score = score
    #             best_density = density
    #     main_cluster = best_label

    #     table_line_ids: List[str] = [line_ids[i] for i, label in enumerate(labels) if label == main_cluster]
    #     logger.info(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}, table_lines: {table_line_ids}")
    #     selected_indices = [all_lines[line_id].line_index for line_id in table_line_ids if line_id in all_lines]

    #     if not selected_indices:
    #         logger.error("No se encontraron indices para table_line_ids.")
    #         return table_line_ids

    #     # Paso 2: Calcular el rango
    #     min_idx, max_idx = min(selected_indices), max(selected_indices)

    #     # Paso 3: Generar todos los índices en ese rango
    #     full_range_indices = range(min_idx, max_idx + 1)

    #     # Paso 4: Mapear de vuelta a line_id (str)
    #     full_range_line_ids = [index_to_id.get(idx, f"line_{idx:04d}") for idx in full_range_indices]

    #     logger.warning(f"DBSCAN: Rango de líneas tabulares: {full_range_line_ids}, total: {len(full_range_line_ids)}")

    #     return full_range_line_ids
        