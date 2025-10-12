# PerfectOCR/core/vectorial_transformation/density_scanner.py
from sklearn.cluster import HDBSCAN, DBSCAN
import numpy as np
import time
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines

logger = logging.getLogger(__name__)

class DensityScanner(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('dbscan', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        # Si el vectorizador ya detectó las líneas tabulares (all_features es None), no se ejecuta el scanner
        if context.get("all_features", {}) is None:
            logger.debug("El vectorizador ya detectó líneas tabulares, el scanner no se ejecuta.")
            return True

        start_time = time.time()
        try:
            logger.debug("DBSCScanner iniciado")
            analyses: Dict[str, Dict[str, float]] = context.get("all_features", {})
            logger.debug(f"Features recibidos por Scanner: {len(analyses)} líneas")
            
            valid_analyses = self._get_interval(analyses, manager)
            
            table_line_ids: List[str] = self._apply_dbscan_clustering(valid_analyses)
            logger.info(f"DBSCAN: {len(table_line_ids)} table_line_ids: {table_line_ids}")
            # htable_line_ids: List[str] = self._apply_hdbscan_clustering(valid_analyses)
            # logger.info(f"HDBSCAN: {len(htable_line_ids)} table_line_ids: {htable_line_ids}")
            if table_line_ids:
                # consecutive_indices = self._get_consecutive_indices(table_line_ids, list(valid_analyses.keys()))
                # logger.debug(f"{len(consecutive_indices)} consecutive_indices: {consecutive_indices}")
                # expanded_line_ids: List[str] = self._expand_to_consecutive_interval_by_ids(consecutive_indices, list(valid_analyses.keys()))
                # logger.debug(f"{len(expanded_line_ids)} expanded_line_ids: {expanded_line_ids}")
                success: bool = manager.save_tabular_lines(table_line_ids)

                total_time = time.time() - start_time
                logger.debug(f"Detección de tablas en: {total_time:.6f}s. Encontradas {len(table_line_ids)}, {table_line_ids}")

                if success:
                    logger.debug("Líneas guardadas en el manager desde DBSCAN")
                    return True
                if self.output:
                    file_name: str = context.get("image_name", "")
                    self._save_output(context, table_line_ids, file_name, manager)
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
                
        line_ids = list(valid_analyses.keys())
        features: List[List[float]] = []
        
        for line_data in valid_analyses.values():
            features.append(list(line_data.values()))
            
        features_array = np.array(features, dtype=np.float32)

        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
        labels: np.ndarray[Any, np.dtype[np.int8]] = clustering.fit_predict(features_array)
        
        unique_labels: List[int] = [l for l in set(labels) if l != -1]
        if not unique_labels:
            logger.warning("DBSCAN: No se encontraron clusters válidos.")
            return []
                
        cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
        main_cluster = max(cluster_sizes, key=cluster_sizes.get)
        
        logger.info(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}")
        table_line_ids: List[str] = [line_ids[i] for i, label in enumerate(labels) if label == main_cluster]
    
        return table_line_ids
        
    def _apply_hdbscan_clustering(self, valid_analyses: Dict[str, Dict[str, float]]) -> List[str]:
        """Aplica DBSCAN para agrupar líneas similares - versión que acepta diccionario."""
        min_cluster_size = int(self.worker_config.get("min_cluster_size")) # type: ignore
        min_samples = int(self.worker_config.get("hmin_cluster_size")) # type: ignore
        if len(valid_analyses) < min_cluster_size:
            logger.warning("No hay suficientes líneas válidas para clustering.")
            return []
                
        line_ids = list(valid_analyses.keys())
        features: List[List[float]] = []
        
        for line_data in valid_analyses.values():
            features.append(list(line_data.values()))
            
        features_array = np.array(features, dtype=np.float32)
        
        hclustering = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=True
        )
        hlabels: np.ndarray[Any, np.dtype[np.signedinteger]] = hclustering.fit_predict(features_array)
        logger.info(f"HDBSCAN: min_samples={min_cluster_size}, labels={hlabels}")
        
        hunique_labels: List[int] = [l for l in set(hlabels) if l != -1]
        if not hunique_labels:
            logger.warning("HDBSCAN: No se encontraron clusters válidos.")
            return []
        
        hcluster_sizes: Dict[int, int] = {hlabel: list(hlabels).count(hlabel) for hlabel in hunique_labels}
        hmain_cluster = max(hcluster_sizes, key=hcluster_sizes.get)
        
        logger.info(f"HDBSCAN: hcluster_sizes={hcluster_sizes}, hmain_cluster={hmain_cluster}")
        table_line_ids: List[str] = [line_ids[i] for i, hlabel in enumerate(hlabels) if hlabel == hmain_cluster]
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

    def _get_interval(self, analyses: Dict[str, Dict[str, float]], manager: DataFormatter) -> Dict[str, Dict[str, float]]:
        """
        Filtra solo las líneas después del encabezado para evitar ruido.
        """

        all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
        header_line_id = [lid for lid, l in all_lines.items() if getattr(l, "header_line", not None)]
        header_line_id = header_line_id[0] if header_line_id else None
        if header_line_id is None:
            return analyses
        
        line_ids = list(analyses.keys())
        header_idx = line_ids.index(header_line_id)
        # Solo tomar líneas después del encabezado
        filtered_ids = line_ids[header_idx:]
        valid_analyses = {lid: analyses[lid] for lid in filtered_ids}
        logger.debug(f"Lineas filtradas: {len(valid_analyses)}: {filtered_ids} ")
        return valid_analyses

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
