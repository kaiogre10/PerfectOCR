# PerfectOCR/core/workers/image_preparation/cleanner.py
import cv2
import logging
from typing import Dict, Any, Optional
import numpy as np
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImageCleaner(ImagePrepAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('cleaning', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("pre_clean", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        corrections = self.config      
        std_low: float = float(corrections.get("std_low", 15.0))
        sp_thr: float = float(corrections.get("sp_thr", 0.015))
        clahe_clip_base: float = float(corrections.get("clahe_clip", 2.0))
        clahe_grid = tuple(corrections.get("clahe_grid", (8, 8)))
        
        try:
            full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]] = context.get("full_img")
            size = context.get("metadata", {}).get("size")
            if full_img is None:
                logger.error("Cleaner: full_img no encontrado en contexto")
                return False
            if  size is None:
                size = full_img.size
            else:
                size = float(size)
                
            ext_low = float((full_img <= 5).sum())
            ext_high = float((full_img >= 250).sum())
            sp_ratio = (ext_low + ext_high) / size
    
            # 1) Desruido sal‑y‑pimienta (rápido, solo si aplica)
            if sp_ratio > sp_thr:
                den = cv2.medianBlur(full_img, 3)
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

            return True
        except Exception as e:
            logger.error(f"Cleaner: {e}", exc_info=True)
            return False