# PerfectOCR/core/vectorial_transformation/density_scanner.py
import numpy as np
import time
import logging
from sklearn.cluster import DBSCAN # type: ignore
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
        self.min_cluster_size = int(self.worker_config.get("min_cluster_size")) 
        self.eps = float(self.worker_config.get("eps")) 
        self.enabled_outputs = config.get("image_load_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.time()
        try:
            logger.debug("DBSCScanner iniciado")
            analysis: Dict[str, Dict[str, float]] = context.get("all_features", {})
            
            if not analysis:
                logger.warning("No hay features disponibles para procesar por que ya se detectaron lineas tabulares")
                return True

            valid_analyses = self._cut_lines(analysis, manager)
            
            table_line_ids: List[str] = self._apply_dbscan_clustering(valid_analyses)
            logger.info(f"RESULTADOS DBSCAN: {len(table_line_ids)} table_line_ids: {table_line_ids}")
            if table_line_ids:
                success: bool = manager.save_tabular_lines(table_line_ids)
                total_time = time.time() - start_time
                
                if self.output:
                    from services.output_service import save_debug_json
                    tab_info: Dict[str, Any] = manager.get_tabular_lines(return_objects=True) #type: ignore
                    file_name: str = manager.workflow.metadata.image_name if manager.workflow else ""
                    worker_name = context.get("worker_name") or "density_scanner"
                    output_paths = context["output_paths"]
                    save_debug_json(output_paths, worker_name, tab_info, file_name)

                logger.debug(f"Detección de tablas en: {total_time:.6f}s. Encontradas {len(table_line_ids)}, {table_line_ids}")

                if success:
                    logger.debug("Líneas guardadas en el manager desde DBSCAN")
                    return True
                                        
                else:
                    logger.error("Error al guardar líneas tabulares en el workflow")
                    return False
                    
        except Exception as e:
            logger.error(f"DBSCAN no detectó tablas en el documento: {e}", exc_info=True)
        return False

    def _apply_dbscan_clustering(self, valid_analyses: Dict[str, Dict[str, float]]) -> List[str]:
        """Aplica DBSCAN para agrupar líneas similares - versión que acepta diccionario."""
        if len(valid_analyses) < self.min_cluster_size:
            logger.warning("No hay suficientes líneas válidas para clustering.")
            return []
                
        line_ids = list(valid_analyses.keys())
        features: List[List[float]] = []
        
        for line_data in valid_analyses.values():
            features.append(list(line_data.values()))
            
        features_array = np.array(features, dtype=np.float32)

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_cluster_size)
        labels: np.ndarray[Any, Any] = clustering.fit_predict(features_array)
        
        unique_labels: List[int] = [l for l in set(labels) if l != -1]
        if not unique_labels:
            logger.warning("DBSCAN: No se encontraron clusters válidos.")
            return []
                
        cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
        main_cluster = max(cluster_sizes, key=cluster_sizes.get)
        
        table_line_ids: List[str] = [line_ids[i] for i, label in enumerate(labels) if label == main_cluster]
        logger.debug(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}, table_lines: {table_line_ids}")
    
        return table_line_ids

    def _cut_lines(self, analysis: Dict[str, Dict[str, float]], manager: DataFormatter) -> Dict[str, Dict[str, float]]:
        """
        Filtra solo las líneas después del encabezado y antes del footer para evitar ruido.
        """
        all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
        
        header_line_ids = [lid for lid, l in all_lines.items() if getattr(l, "header_line", not None)]
        header_line_id = header_line_ids[0] if header_line_ids else None
        footer_line_ids = [lid for lid, l in all_lines.items() if getattr(l, "footer_line", None) is not None]
        footer_line_id = footer_line_ids[0] if footer_line_ids else None

        if not header_line_id and not footer_line_id:
            logger.info("No hay header ni footer: se devuelve el análisis completo")
            return analysis

        line_ids = list(analysis.keys())
        if header_line_id:
            if header_line_id in line_ids:
                header_idx = line_ids.index(header_line_id) + 1  # después del header
                return {lid: analysis[lid] for lid in line_ids[header_idx:]}
            logger.warning(f"header_line_id {header_line_id} no encontrada en analysis keys")
            return analysis

        # Si llegamos aquí, existe footer_line_id y header_line_id es None
        if footer_line_id:
            if footer_line_id in line_ids:
                footer_idx = line_ids.index(footer_line_id)  # antes del footer
                return {lid: analysis[lid] for lid in line_ids[:footer_idx]}
            logger.warning(f"footer_line_id {footer_line_id} no encontrada en analysis keys")
            
            return analysis
        return analysis