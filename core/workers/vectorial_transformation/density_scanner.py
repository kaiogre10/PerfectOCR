# PerfectOCR/core/workflow/vectorial_transformation/density_scanner.py
from sklearn.cluster import DBSCAN #type: ignore
import numpy as np
import time
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class DensityScanner(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('dbscan', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.time()
        try:
            logger.info("DBSCScanner iniciado")
            valid_analyses: Dict[str, Dict[str, float]] = context.get("all_features", {})
            logger.info(f"Features recibidos por Scanner: {len(valid_analyses)} líneas")
            # logger.info(f"Features: {valid_analyses}")
            
            if not valid_analyses:
                logger.warning("No se recibieron features del Vectorizer")
                return False
                
            table_line_ids: List[str] = self._apply_dbscan_clustering(valid_analyses)
            logger.debug(f"{len(table_line_ids)} table_line_ids: {table_line_ids}")
            if table_line_ids:
                consecutive_indices: List[int] = self._get_consecutive_indices(table_line_ids, list(valid_analyses.keys()))
                logger.debug(f"{len(consecutive_indices)} consecutive_indices: {consecutive_indices}")
                expanded_line_ids: List[str] = self._expand_to_consecutive_interval_by_ids(consecutive_indices, list(valid_analyses.keys()))
                logger.debug(f"{len(expanded_line_ids)} expanded_line_ids: {expanded_line_ids}")
                success: bool = manager.save_tabular_lines(expanded_line_ids)

                total_time = time.time() - start_time
                logger.info(f"Detección de tablas en: {total_time:.6f}s. Encontradas {len(expanded_line_ids)}, {expanded_line_ids} ")

                if success:
                    logger.info("Líneas guardadas en el manager desde DBSCAN")
                    return True
                if self.output:
                    file_name: str = context.get("image_name", "")
                    self._save_output(context, expanded_line_ids, file_name, manager)
                    return True
                else:
                    logger.error("Error al guardar líneas tabulares en el workflow")
                    return False
        except Exception as e:
            logger.error(f"DBSCAN no detectó tablas en el documento: {e}", exc_info=True)
        return False

    def _apply_dbscan_clustering(self, valid_analyses: Dict[str, Dict[str, float]]) -> List[str]:
        """Aplica DBSCAN para agrupar líneas similares - versión que acepta diccionario."""
        min_cluster_size = int(self.worker_config.get("min_cluster_size", [])) # type: ignore
        eps = float(self.worker_config.get("eps", [])) # type: ignore
        if len(valid_analyses) < min_cluster_size:
            logger.warning("No hay suficientes líneas válidas para clustering.")
            return []
        # logger.info(f"Features RECIBIDOS PARA CLUSTER: {valid_analyses}")
        
        line_ids = list(valid_analyses.keys())
        features: List[List[float]] = []
        
        for line_data in valid_analyses.values():
            features.append(list(line_data.values()))
            
        features_array = np.array(features, dtype=np.float32)

        # scaler = StandardScaler()
        # features_scaled: np.ndarray[Any, np.dtype[np.float64]] = scaler.fit_transform(features_array)

        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
        labels: np.ndarray[Any, np.dtype[np.uint8]] = clustering.fit_predict(features_array).astype(dtype=np.uint8)
        
        logger.debug(f"DBSCAN: eps={eps}, min_samples={min_cluster_size}, labels={labels}")
        
        unique_labels: List[int] = [l for l in set(labels) if l != -1]
        if not unique_labels:
            logger.warning("DBSCAN: No se encontraron clusters válidos.")
            return []
        
        cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
        main_cluster = max(cluster_sizes, key=cluster_sizes.get)
        
        logger.debug(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}")
        table_line_ids: List[str] = [line_ids[i] for i, label in enumerate(labels) if label == main_cluster]
        return table_line_ids

    def _get_consecutive_indices(self, table_line_ids: List[str], all_line_ids: List[str]) -> List[int]:
        """Convierte line_ids a índices en la lista ordenada."""
        indices: List[int] = []
        for line_id in table_line_ids:
            if line_id in all_line_ids:
                indices.append(all_line_ids.index(line_id))
        return sorted(indices)

    def _expand_to_consecutive_interval_by_ids(self, consecutive_indices: List[int], all_line_ids: List[str]) -> List[str]:
        """Expande los line_ids detectados a un intervalo consecutivo."""
        if not consecutive_indices:
            return []
        
        # CAMBIO: Recibe los índices directamente, no los calcula de nuevo
        if not consecutive_indices:
            return []
        
        start_idx = min(consecutive_indices)
        end_idx = max(consecutive_indices)
        
        consecutive_line_ids: List[str] = []
        for i in range(start_idx, end_idx + 1):
            if i < len(all_line_ids):
                consecutive_line_ids.append(all_line_ids[i])
        
        return consecutive_line_ids

    def _save_output(self, context: Dict[str, Any], expanded_line_ids: List[str], file_name: str, manager: DataFormatter):
        from services.output_service import save_tabjson
        import os
        project_root = self.project_root
        output_file = context.get("output_paths", [])
        for path in output_file:
            output_dir: str = os.path.join(path, "dbscan")
            json_file_name = f"{os.path.splitext(file_name)[0]}.json"
            output_file = save_tabjson(expanded_line_ids, manager, output_dir, json_file_name, project_root)
        if output_file:
            logger.info(f"OCR Raw results para '{file_name}' guardado en {len(output_file)} ubicaciones.")