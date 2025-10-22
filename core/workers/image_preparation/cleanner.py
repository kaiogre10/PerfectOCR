# PerfectOCR/core/workers/image_preparation/cleanner.py
import cv2
import logging
from typing import Dict, Any
import numpy as np
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImageCleaner(ImagePrepAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('cleaner', {})
        self.enabled_outputs = config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("pre_clean", False)
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        std_low = self.worker_config.get("std_low")
        sp_thr: float = self.worker_config.get("sp_thr")
        clahe_clip_base = self.worker_config.get("clahe_clip")
        clahe_grid = self.worker_config.get("clahe_grid", [])
        kernel_size: int = self.worker_config.get("kernel_size")

        try:
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
            logger.debug("Full_img obtenida con éxito")
            
            img_dims: Dict[str, int] = {}
            if manager.workflow and hasattr(manager.workflow, "metadata") and hasattr(manager.workflow.metadata, "img_dims"):
                img_dims = dict(getattr(manager.workflow.metadata, "img_dims", {}))

            size = img_dims.get("size")
            if size is None:
                size = full_img.size
            else:
                size = float(size)
                
            ext_low = float((full_img <= 5).sum())
            ext_high = float((full_img >= 250).sum())
            sp_ratio = (ext_low + ext_high) / size
    
            # 1) Desruido sal‑y‑pimienta (rápido, solo si aplica)
            if sp_ratio > sp_thr:
                den = cv2.medianBlur(src=full_img, ksize=kernel_size)
                full_img[...] = den

            # Recalcular contraste
            std1 = float(np.std(full_img))

            # 2) Contraste local con CLAHE (solo si contraste bajo)
            if std1 < std_low:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip_base, tileGridSize=clahe_grid)
                en1 = clahe.apply(full_img)
                full_img[...] = en1

                # Si siguió bajo, subir ligeramente el clipLimit
                std2 = float(np.std(full_img))
                if std2 < std_low:
                    clahe2 = cv2.createCLAHE(clipLimit=clahe_clip_base + 0.5, tileGridSize=clahe_grid)
                    en2 = clahe2.apply(full_img)
                    full_img[...] = en2

            # 3) Nitidez local (unsharp adaptativo)
            lap = cv2.Laplacian(full_img, cv2.CV_64F)
            lap_var = float(lap.var())
            stdf = float(np.std(full_img))

            if lap_var < 20.0 or stdf <= 25.0:
                alpha, beta = 1.2, -0.2  # suave
            elif lap_var < 60.0:
                alpha, beta = 1.4, -0.4  # medio
            else:
                alpha, beta = 1.1, -0.1  # mínimo

            blur = cv2.GaussianBlur(full_img, (3, 3), 0)
            sharp = cv2.addWeighted(full_img, alpha, blur, beta, 0)
            np.clip(sharp, 0, 255, out=sharp)
            if sharp.dtype != np.uint8:
                sharp = sharp.astype(np.uint8, copy=False)

            full_img[...] = sharp
            
            manager.update_full_img(full_img)
                
            return True
            
        except Exception as e:
            logger.error(f"Cleaner: {e}", exc_info=True)
            return False
