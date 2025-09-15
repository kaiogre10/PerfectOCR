import os
import time
from typing import Dict, Any, List, Optional
import logging
from numba.core.types import ExceptionInstance
from core.domain.data_models import Polygons
from data.scripts.word_finder import WordFinder
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class DataFinder(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('data_finder', {})

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time = time.time()
            logger.debug("Data Finder iniciado")
            
            if not manager or not getattr(manager, "workflow", None):
                logger.warning("Manager o workflow ausente")
                return False
            
            workflow = manager.workflow
            polygons: Dict[str, Polygons] = getattr(workflow, "polygons", {}) or {}
                
            if not polygons:
                logger.info("No hay polygons para procesar")
                return False
            
            # Llamar al método original que funciona
            polygon_updates = self._find_data(polygons, manager)

            # Actualiza las líneas marcadas como encabezado en las dataclasses
            if polygon_updates:
                success: bool = manager.update_polygon_data(polygon_updates)
                return success
            # Guardar resultados en el contexto
            total_time = time.time() - start_time
            logger.info(f"Key Fields detectados (por palabra): {len(polygon_updates)} en {total_time:.4f}s")
            
        except Exception as e:
            logger.error(f"Error detectando encabezados por palabra: {e}", exc_info=True)
            return False

    def _find_data(self, polygons: Dict[str, Polygons], manager: DataFormatter) -> Optional[Dict[str, str]]:
        min_similarity = self.worker_config.get("min_similarity", 0.90)
        try:

            logger.debug("_find_headers: inicio de búsqueda de encabezados")

            # ruta al modelo configurable
            model_path = None
            try:
                model_path = self.worker_config.get("wordfinder_model_path") or self.config.get("wordfinder_model_path")
            except Exception:
                model_path = None
            if not model_path:
                model_path = os.path.join(self.project_root or ".", "data", "wordfinder_model.pkl")
            logger.debug(f"_find_headers: ruta modelo WordFinder -> {model_path}")

            try:
                wf = WordFinder(model_path)
                logger.info("_find_headers: WordFinder inicializado correctamente")
            except Exception as e:
                logger.warning(f"WordFinder no pudo inicializarse con {model_path}: {e}", exc_info=True)
                return []

            if not polygons:
                logger.info("No hay polígonos para procesar")
                return None
            else:

                logger.info(f"Data_finder: cantidad polygons={len(polygons)}")

            processed_count = 0
            polygon_updates: Dict[str, str] = {}

            for pid, poly in polygons.items():
                processed_count += 1
                
                # Obtener texto del polígono
                text = getattr(poly, "ocr_text", "") or ""
                if not text:
                    continue
                
                # Buscar con WordFinder
                results = wf.find_keywords(text)
                if not results:
                    continue
                
                # Filtrar por similitud mínima
                valid_results = [r for r in results if r.get('similarity', 0.0) >= min_similarity]
                
                if valid_results:
                    best_result = max(valid_results, key=lambda x: x.get('similarity', 0.0))
                    key_field = best_result.get('key_field')
                    if key_field:
                        polygon_updates[pid] = key_field
                        logger.info(f"Similitud por palabra{best_result}")

            return polygon_updates
                    
        except Exception as e:
            logger.info(f"Fallo en búsqueda de datos globales{e}", exc_info=True)