#core/workers/ocr/text_refiner.py
import logging
import time 
# import dataclasses
from typing import Dict, Any, Tuple, List
from core.domain.data_formatter import DataFormatter
# from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.semantic_clasificator import SemanticClasificator
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.text_corrector import TextCorrector
from core.workers.ocr.fragmenter import Fragmenter
# from fuzzywuzzy import utils #type: ignore
# from core.utils.semantic_clasifier import clasify_words
# from core.utils.textual_cleaner import cleanning_text
# from core.utils.binarization import binarice
# from core.utils.pattern_finder import is_acronym
# from core.utils.word_divider import fragment

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR con clasificación selectiva optimizada.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, clasificator: SemanticClasificator, cleaner: TextCleaner, fragmenter: Fragmenter, corrector: TextCorrector):
        super().__init__(config, project_root)
        self.worker_config = self.config.get("text_refiner", {})
        self.semantic_config = config.get("semantic_clasificator", {})
        self.cleanner_config = config.get("text_cleaner", {})
        self.binarizator_config = self.config.get('binarizator', {})
        self.fragmenter_config = self.config.get('fragmenter', {})
        self.num_passes = self.worker_config.get("num_passes")
        self.clasificator = clasificator
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento con clasificación selectiva.
        """
        start_time = time.perf_counter()
        # try:
        #     encoder: Dict[str, float] = manager.get_density_encoder()
        #     frecuency_char: Dict[str, float] = manager.get_frecuency_char()
        #     min_area = self.get_min_area(manager)
            
        #     polygons_in: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
        #     sorted_poly_ids = sorted(polygons_in.keys())
        #     list_of_final_polygons: List[Polygons] = []
            
        #     eliminated_polys = 0
            
        #     for poly_id in sorted_poly_ids:
        #         polygon = polygons_in[poly_id]
                
        #         cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
        #         if cropped_img is None:
        #             logger.warning(f"Cropped_img de {poly_id} es None")
        #             eliminated_polys += 1
        #             continue
                            
        #         binarizator_config = self.binarizator_config
        #         blob_metrics = binarice(cropped_img, binarizator_config, min_area)
        #         if blob_metrics:
        #             continue
            
        #         text = polygon.ocr_text or ""
        #         if not utils.validate_string(text): #type: ignore
        #             eliminated_polys += 1
        #             continue
                
        #         sc: Tuple[str, bool] = clasify_words(text, encoder, self.semantic_config)
                
        #         logger.debug(f"{poly_id}: {text}, {sc}")
        #         ocr_confidence = polygon.ocr_confidence or 0.0
                            
        #         # Si el texto corresponde a una sigla (p.e. 'P.U.C.D', 'I.V.A.') se conserva intacto
        #         if is_acronym(text):
        #             logger.info(f"No fragmentando sigla detectada: '{text}'")
        #             logger.info(f"{poly_id}: {text}")
        #             continue

        #         cleanned_word = cleanning_text(self.cleanner_config, text, sc, frecuency_char, ocr_confidence)
        #         if cleanned_word is None:
        #             eliminated_polys += 1
        #             continue
                
        #         sc = clasify_words(cleanned_word, encoder, self.semantic_config)
                
        #         fragment_dicts = fragment(self.fragmenter_config, polygon, blob_metrics, sc)
        
        #         # Reconstruir cada fragmento como Polygons
        #         for frag_dict in fragment_dicts:
        #             base_polygon = frag_dict['polygon']
        #             geom_updates = frag_dict.get('geometry_updates')
        #             text_updates = frag_dict.get('text_updates')
        #             was_refined = frag_dict.get('was_refined', False)
                    
        #             # Aplicar updates si existen
        #             if geom_updates:
        #                 new_geom = dataclasses.replace(
        #                     base_polygon.geometry,
        #                     **geom_updates
        #                 )
        #                 updated_polygon = dataclasses.replace(
        #                     base_polygon,
        #                     geometry=new_geom,
        #                     ocr_text=text_updates.get('ocr_text') if text_updates else base_polygon.ocr_text,
        #                     was_refined=was_refined
        #                 )
        #             else:
        #                 updated_polygon = base_polygon
                    
        #             list_of_final_polygons.append(updated_polygon)
                    
        #     final_polygons_dict: Dict[str, Polygons] = {}
        #     for idx, poly_obj in enumerate(list_of_final_polygons):
        #         new_id = f"poly_{idx:04d}"
        #         final_poly = dataclasses.replace(poly_obj, polygon_id=new_id)
        #         final_polygons_dict[new_id] = final_poly
                            
        #     logger.debug(f"Refinamiento textual terminado en {time.perf_counter() - start_time:.6f}")
        #     return True
        # except Exception as e:
        #     logger.error(f"Error en el refinamiento textual: {e}", exc_info=True)
        #     return False
                
        logger.debug(f"Refinador inicializado para {self.num_passes} pasadas.")

        try:
            for i in range(self.num_passes):
                pass_num = i + 1
                logger.debug(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")
                
                # Determinar si usar filtro selectivo (solo en pasadas 2+)
                use_filter = (i > 0)

                logger.debug(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica (filtro={use_filter})")
                self.clasificator.transcribe(context, manager, filter_modified=use_filter)

                logger.debug(f"Bucle #{pass_num}: Limpieza de Texto")
                self.cleaner.transcribe(context, manager)

                logger.debug(f"Pasada 2, bucle #{pass_num}: Clasificación Semántica (solo corregidos)")
                self.clasificator.transcribe(context, manager, filter_modified=True)

                logger.debug(f"Bucle #{pass_num}: Fragmentación de Texto")
                self.fragmenter.transcribe(context, manager)

                logger.debug(f"Pasada 3, bucle #{pass_num}: Clasificación Semántica (solo limpiados)")
                self.clasificator.transcribe(context, manager, filter_modified=True)
                
                logger.debug(f"Bucle #{pass_num}: Corrección textual")
                self.corrector.transcribe(context, manager)

            # Clasificación final completa para asegurar consistencia
            logger.debug(f"Pasada final: Clasificación Semántica (completa)")
            self.clasificator.transcribe(context, manager, filter_modified=False)
            
            logger.debug(f"Clasificación Semántica Final Completada en: {time.perf_counter()-start_time:.6f}s")
            return True
        
        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False

    # def get_min_area(self, manager: DataFormatter) -> int:
    #     polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else None #type: ignore
    #     polys_areas: List[int] = []
        
    #     for poly_data in polygons.values():
    #         poly_area = poly_data.cropped_img.cropped_img.size
    #         polys_areas.append(poly_area)

    #     return min(polys_areas)

