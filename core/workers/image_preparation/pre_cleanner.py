# PerfectOCR/core/workers/image_preparation/pre_cleanner.py
import logging
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import clean_img

logger = logging.getLogger(__name__)

class ImageCleaner(ImagePrepAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('cleaner', {})
        # self.dpi_range = config["dpi_range"]
        self.enabled_outputs = config.get("image_load_outputs", {})
        self.output = self.enabled_outputs.get("pre_clean", False)
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
            logger.debug("Full_img obtenida con éxito")

            full_img = clean_img(full_img, self.worker_config)
            
            corrected = True

            if self.output:
                from services.output_service import save_croped_image
                image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                worker_name = context.get("worker_name") or "pre_cleanner"
                output_paths = context["output_paths"]
                poly_id = f"full_img_{image_name}_{worker_name}"
                save_croped_image(image_name, poly_id, full_img, output_paths, worker_name, method=None)
                logger.debug(f"Imagen preprocesada  guardada como output intermedio 'pre_cleanner'")
                
            manager.update_full_img(corrected, full_img)
                
            return True
            
        except Exception as e:
            logger.error(f"Cleaner: {e}", exc_info=True)
            return False
