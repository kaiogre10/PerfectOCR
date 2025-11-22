# core/workers/ocr/text_refiner.py
from typing import Dict, Any
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.semantic_clasificator import SemanticClasificator
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.text_corrector import TextCorrector
from core.workers.ocr.fragmenter import Fragmenter
from core.utils.image_analizer import analize_bin_img
from core.utils.image_utils import binarice_img
from core.utils.text_validator import validate_text
from core.domain.data_models import Polygons
import logging

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR con clasificación selectiva optimizada.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, clasificator: SemanticClasificator, cleaner: TextCleaner, fragmenter: Fragmenter, corrector: TextCorrector):
        super().__init__(config, project_root)
        self.worker_config = config.get("text_refiner", {})
        self.percentile = config["percentile"]
        self.worker_config["percentile"] = self.percentile 
        self.num_passes = self.worker_config.get("num_passes")
        self.delete_cropp = config.get("fragmented_polys")
        self.output = config.get("binarized_polygons")
        self.clasificator = clasificator
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento con clasificación selectiva.
        """
        logger.debug(f"Refinador inicializado para {self.num_passes} pasadas.")

        try:
            polygons_in: Dict[str, Polygons] = manager.workflow.polygons
            sorted_poly_ids = sorted(polygons_in.keys())
            blob_metrics: Dict[str, Any] = {}

            for poly_id in sorted_poly_ids:
                polygon = polygons_in[poly_id]
                
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                
                if cropped_img is None:
                    logger.warning(f"Cropped_img de {poly_id} es None")
                    continue

                # Analiza la imagen y asigna el diccionario de métricas resultante a la clave correspondiente al poly_id actual.
                # logger.info(f"{poly_id}:")
                bin_img = binarice_img(cropped_img, {})

                if self.output:
                    from services.output_service import save_croped_image
                    worker_name = "binarized"
                    image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    output_paths = context["output_paths"]
                    save_croped_image(image_name, poly_id, bin_img, output_paths, worker_name, method=worker_name)
                
                if not validate_text(polygon.ocr_text or ""):
                    continue
                
                self.worker_config["text"] = polygon.ocr_text
                metrics = analize_bin_img(bin_img, self.worker_config, False)
                if metrics.get('needs_fragmentation'):

                    logger.info(f"{poly_id}: para fragmentar: {polygon.ocr_text} en: {metrics.get("num_blobs", {})}")
                    
                blob_metrics[poly_id] = metrics
    
                # logger.info(f"{poly_id}: Blobs={metrics.get('num_blobs', {})} | Palabras: {num_words} | Texto: '{polygon.ocr_text}'")

            if self.delete_cropp:
                logger.info("Fragmenter liberara las imagenes")
            else:
                manager.delete_cropped_images()
                logger.info("Cropped_img liberadas")

            # logger.info(f"Métricas :{blob_metrics}.")

            if not blob_metrics:
                context["blob_metrics"] = None

            else: 
                context["blob_metrics"] = blob_metrics
             
            for i in range(self.num_passes):
                pass_num = i + 1
                logger.debug(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")

                logger.debug(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica")
                self.clasificator.transcribe(context, manager)
                
                logger.debug(f"Bucle #{pass_num}: Limpieza de Texto")
                self.cleaner.transcribe(context, manager)

                logger.debug(f"Pasada 2, bucle #{pass_num}: Clasificación Semántica (solo corregidos)")
                self.clasificator.transcribe(context, manager)
                
                logger.debug(f"Bucle #{pass_num}: Fragmentación de Texto")
                self.fragmenter.transcribe(context, manager)

                logger.debug(f"Pasada 3, bucle #{pass_num}: Clasificación Semántica (solo limpiados)")
                self.clasificator.transcribe(context, manager)
                
                logger.debug(f"Bucle #{pass_num}: Corrección textual")
                self.corrector.transcribe(context, manager)

            logger.debug(f"Pasada final: Clasificación Semántica completa")
            self.clasificator.transcribe(context, manager, final_pass='final_class')
        
            return True
        
        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False
