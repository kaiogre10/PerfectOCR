import time
from typing import Dict, Any, Optional, List
import logging
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class DataFinder(OCRAbstractWorker):
    def __init__(self, config: Dict[str, str], cfg: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config = config
        self.cfg = cfg
        self.worker_config = cfg.get('data_finder', {})
        self._model = None
    
    @property
    def model(self) -> Optional[Any]:
        try:
            if self._model is None:
                from core.domain.models_manager import ModelsManager
                model_manager = ModelsManager.get_instance()
                self._model = model_manager.word_finder
                logger.debug("DataFinder: Modelo de búsqueda obtenido del ModelsManager")
            
            return self._model
        except Exception as e:
            logger.error(f"DataFinder: Modelo de búsqueda no disponible en ModelManager{e}", exc_info=True)
            return None

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.time()
        try:
            logger.debug("Data Finder iniciado")
            
            if not manager or not getattr(manager, "workflow", None):
                logger.warning("Manager o workflow ausente")
                return False
            
            workflow = manager.workflow
            polygons: Dict[str, Polygons] = getattr(workflow, "polygons", {}) or {}
                
            if not polygons:
                logger.debug("No hay polygons para procesar")
                return False
            
            # Llamar al método original que funciona
            polygon_updates = self._find_data(polygons, manager)

            # Actualiza las líneas marcadas como encabezado en las dataclasses
            success: bool = manager.update_polygon_data(polygon_updates)
            
            # Guardar resultados en el contexto
            total_time = time.time() - start_time
            logger.debug(f"Key Fields detectados en {total_time:.4f}s")
            return success  # Siempre retorna True para continuar con el pipeline
        except Exception as e:
            logger.error(f"Error detectando encabezados por palabra: {e}", exc_info=True)
            return True  # Retorna True para continuar con fallbacks

    def _find_data(self, polygons: Dict[str, Polygons], manager: DataFormatter) -> Dict[str, str]:
        threshold = self.worker_config.get("min_similarity", 0.90)
        max_q_lenght = self.worker_config.get("max_q_lenght", 15)
        
        if self.model is None:
            logger.error("DataFinder no iniciado, no se puede búsacar texto")
            return None
        try:
            logger.debug("DataFinder: inicio de búsqueda de encabezados")
            if not polygons:
                logger.debug("No hay polígonos para procesar")
                return None
            else:

                logger.debug(f"Data_finder: cantidad polygons={len(polygons)}")

            processed_count = 0
            polygon_updates: Dict[str, str] = {}
            skipped_numeric = 0
            skipped_len = 0

            for pid, poly in polygons.items():
                processed_count += 1

                # Omitir numéricos
                semantic = getattr(poly, "semantic_type", "") or ""
                if semantic == "numeric":
                    skipped_numeric += 1
                    continue
                
                # Obtener texto del polígono
                text = getattr(poly, "ocr_text", "") or ""
                if not text:
                    continue

                try: 
                    lenght = len(text.replace(" ", ""))
                except Exception:
                    lenght = len(text)

                try:
                    max_len_cfg = int(max_q_lenght) if max_q_lenght is not None else None
                except Exception:
                    max_len_cfg = None

                if max_len_cfg is not None and lenght > max_len_cfg:
                    skipped_len += 1
                    logger.debug(f"DataFinder: Polígono {pid} omitido por largo ({lenght} > {max_len_cfg})")
                    continue
                
                # Buscar con WordFinder
                valid_results: List[str] = self.model.find_keywords(text, threshold)
                if not valid_results:
                    continue
                
                if valid_results:
                    best_result = max(valid_results, key=lambda x: x.get('similarity', 0.0))
                    key_field = best_result.get('key_field')
                    if key_field:
                        polygon_updates[pid] = key_field
                        logger.debug(f"Similitud por palabra: {pid}: {best_result}")

            if polygon_updates:
                logger.debug(f"Encontradas {len(polygon_updates)} coincidencias de palabras clave")
                logger.debug(f"DataFinder: {skipped_numeric} polígonos 'numeric' omitidos")
                return polygon_updates
            else:
                logger.debug("DataFinder: No se encontraron coincidencias de palabras clave - usando fallback")
                logger.debug(f"DataFinder: {skipped_numeric} polígonos 'numeric' omitidos")
                    
        except Exception as e:
            logger.warning(f"Fallo en búsqueda de datos globales{e}", exc_info=True)
        