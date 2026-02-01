# PerfectOCR/core/vectorial_transformation/density_scanner.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines
from core.utils.math_utils import density_cluster

logger = logging.getLogger(__name__)

class DensityScanner(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('dbscan', {})
        self.min_cluster_size = int(worker_config.get("min_cluster_size")) 
        self.eps = float(worker_config.get("eps"))
        self.metric = worker_config.get("metric", "")
        self.output = config.get("table_lines", False)

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.perf_counter()
        try:
            logger.debug("DBSCScanner iniciado")
            analysis: np.ndarray[Any, Any] = context["all_features"]
            
            if analysis.size == 0:
               logger.warning("No hay features disponibles para procesar por que ya se detectaron lineas tabulares")
               return True

            valid_analyses = self._cut_lines(analysis, manager)
            
            table_line_ids: List[str] = self._apply_dbscan_clustering(analysis, manager)
            logger.debug(f"RESULTADOS DBSCAN: {len(table_line_ids)} table_line_ids: {table_line_ids}")
            if table_line_ids:
                success: bool = manager.save_tabular_lines(table_line_ids)
            
                if self.output:
                    from services.output_service import save_debug_json
                    tab_info: Dict[str, Any] = manager.get_tabular_lines(return_objects=True) #type: ignore
                    file_name: str = manager.workflow.metadata.image_name if manager.workflow else ""
                    worker_name = context.get("worker_name") or "density_scanner"
                    output_paths = context["output_paths"]
                    save_debug_json(output_paths, worker_name, tab_info, file_name)

                logger.info(f"Detección de tablas en: {time.perf_counter() - start_time:.6f}s. Encontradas {len(table_line_ids)}, {table_line_ids}")

                if success:
                    logger.debug("Líneas guardadas en el manager desde DBSCAN")
                    return True
                                        
                else:
                    logger.error("Error al guardar líneas tabulares en el workflow")
                    return False
                    
        except Exception as e:
            logger.error(f"DBSCAN no detectó tablas en el documento: {e}", exc_info=True)
        return False
       
    def _apply_dbscan_clustering(self, features_array: np.ndarray[Any, Any], manager: DataFormatter) -> List[str]:
        """Aplica DBSCAN para agrupar líneas similares"""
        all_lines = manager.workflow.all_lines if manager.workflow else {}
        int_line_ids = features_array[:, 0].astype(int)
        features_for_clustering = features_array[:, 1:]

        # Crear un diccionario que mapea line_index (int) a line_id (str)
        index_to_id: Dict[int, str] = {}
        for line_id, line_obj in all_lines.items():
            # Extraer número de "line_X"
            idx = line_obj.line_index  # Esto ya es un int
            index_to_id[idx] = line_id
        
        # Obtener line_ids correspondientes
        line_ids = [index_to_id.get(int(idx), f"line_{int(idx)}") for idx in int_line_ids]
        labels: np.ndarray[Any, Any] = density_cluster(features_for_clustering, self.eps, self.min_cluster_size, self.metric)
        
        unique_labels: List[int] = [l for l in set(labels) if l != -1]
        if not unique_labels:
            logger.warning("DBSCAN: No se encontraron clusters válidos.")
            return []
                
        cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
        main_cluster = max(cluster_sizes, key=cluster_sizes.get)
        
        table_line_ids: List[str] = [line_ids[i] for i, label in enumerate(labels) if label == main_cluster]
        logger.debug(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}, table_lines: {table_line_ids}")
    
        return table_line_ids

    def _cut_lines(self, analysis: np.ndarray[Any, Any], manager: DataFormatter):
        """
        Filtra solo las líneas después del encabezado y antes del footer para evitar ruido.
        """
        all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
        
        header = self.get_headers(all_lines)
        footer = self.get_footers(all_lines)
        
        if not header and not footer:
            logger.info("No hay header ni footer: se devuelve el análisis completo")
            return analysis
        
        elif header and footer:
            mask = (analysis[:, 0] > header[0]) & (analysis[:, 0] < footer[0])
            return np.compress(mask, analysis, 0)
        
        elif header and not footer:
            return np.compress(analysis[:, 0] > header[0], analysis, 0)
        
        elif footer:
            return np.compress(analysis[:, 0] < footer[0], analysis, 0)
            
        else:
            return analysis
    
    def get_headers(self, all_lines: Dict[str, AllLines]) -> Tuple[int, str]:
        try:
            for line_id, line_data in all_lines.items():
                h = line_data.header_line if line_data.header_line else None
                if h is not None:
                    header_line_id = line_id
                    # logger.info(f"H: {h}, id: {line_id}")
                    return h, header_line_id
        except Exception as e:
            logger.error(f"Error buscando encabezados: {e}", exc_info=True)
        return 0, ""
    
    def get_footers(self, all_lines: Dict[str, AllLines]) -> Tuple[int, str]:
        try:
            for line_id, line_data in all_lines.items():
                f: int | None = line_data.footer_line if line_data.footer_line else None
                if f is not None:
                    footer_line_id = line_id
                    # logger.info(f"f: {f} id: {line_id}")
                    return f, footer_line_id
        except Exception as e:
            logger.error(f"Error buscando footer: {e}", exc_info=True)
        return 0, ""