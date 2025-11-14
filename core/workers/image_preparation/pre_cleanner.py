# PerfectOCR/core/workers/image_preparation/pre_cleanner.py
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
        self.std_low = self.worker_config.get("std_low")
        self.sp_thr: float = self.worker_config.get("sp_thr")
        self.clahe_clip_base = self.worker_config.get("clahe_clip")
        self.clahe_grid = self.worker_config["clahe_grid"]
        self.dimension_thresholds_px = self.worker_config["dimension_thresholds_px"]
        self.kernel_size: int = self.worker_config.get("kernel_size")
        self.enabled_outputs = config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("pre_clean", False)
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
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

            size = img_dims.get("size") or full_img.size
                
            ext_low = float((full_img <= 5).sum())
            ext_high = float((full_img >= 250).sum())
            sp_ratio = (ext_low + ext_high) / size
    
            # 1) Desruido sal‑y‑pimienta (rápido, solo si aplica)
            if sp_ratio > self.sp_thr:
                den = cv2.medianBlur(src=full_img, ksize=self.kernel_size)
                full_img[...] = den

            # Recalcular contraste
            std1 = float(np.std(full_img))

            # 2) Contraste local con CLAHE (solo si contraste bajo)
            if std1 < self.std_low:
                h, w = full_img.shape[:2]
                max_dim = max(h, w)
                
                if max_dim < self.dimension_thresholds_px[0]:
                    grid_size = tuple(self.clahe_grid[0])
                elif max_dim < self.dimension_thresholds_px[1]:
                    grid_size = tuple(self.clahe_grid[1])
                else:
                    grid_size = tuple(self.clahe_grid[2])
                
                logger.debug(f"CLAHE para full_img con grid: {grid_size} basado en max_dim: {max_dim}")

                clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_base, tileGridSize=grid_size)
                en1 = clahe.apply(full_img)
                full_img[...] = en1

                # Si siguió bajo, subir ligeramente el clipLimit
                std2 = float(np.std(full_img))
                if std2 < self.std_low:
                    clahe2 = cv2.createCLAHE(clipLimit=self.clahe_clip_base + 0.5, tileGridSize=grid_size)
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
