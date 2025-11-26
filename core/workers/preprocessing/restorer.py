
import numpy as np
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_analizer import extract_cc_metrics
from core.utils.image_utils import cropp_img
from services.output_service import save_shapes

logger = logging.getLogger(__name__)

class ImageRestorer(PreprocessingAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("restorer", {})
        self.area_threshold: int = self.worker_config.get("area_threshold", {})
        self.output1 = config.get("contours")
        self.output2 = config.get("components")

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:            
            if not manager.validate_cropped_img():
                logger.info(f"Sin cropped_img en el formatter")
                return False
                
            logger.debug("Polygonos revisados")
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False
                        
            # 1. Analysis Phase
            for poly_id, polygon in polygons.items():
                logger.info(f"{poly_id}")
                # Acceso correcto a la imagen desde la dataclass
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None:
                    logger.warning(f"Imagen no encontrada para el polígono '{poly_id}'")
                    continue
               
                noise_cc, cc, noise_cont, blobs_cont = self._analice_morphology(cropped_img.copy(), context, manager, poly_id)


                return False
            
            return False

        except Exception as e:
            logger.error(f"Error restaurando: {e}", exc_info=True)
            return False

    def _analice_morphology(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any], manager: DataFormatter, poly_id: str):
        metrics, bin_img = extract_cc_metrics(cropped_img, worker_config={}, binarice=True)
        if bin_img is None:
            return []
        
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics["cont_array_dict"]
        noise_cont: Dict[int, Any] = {}
        blobs_cont: Dict[int, Any] = {}
        for pos, countours in cont_array_dict.items():
            cont_area = countours["cont_area"]
            cont_coords = countours["cont_coords"]
            cont_bbox = countours["cont_bbox"]
            if cont_area == self.area_threshold:
                noise_cont[pos] = {
                    "cont_area": cont_area, 
                    "cont_coords":cont_coords, 
                    "cont_bbox": cont_bbox
                }
            else:
                blobs_cont[pos] = {
                    "cont_area": cont_area, 
                    "cont_coords":cont_coords, 
                    "cont_bbox": cont_bbox
                }

        # logger.info(f"noise_cont: {noise_cont}, blobs_cont: {blobs_cont}")
        mapped_stats: np.ndarray[Any, np.dtype[np.uint16]] = metrics["mapped_stats"]
        bboxes_cc = np.column_stack([mapped_stats[:, 2], mapped_stats[:, 3], mapped_stats[:, 4], mapped_stats[:, 5]])
        
        mask = mapped_stats[:, 1] != self.area_threshold
        noise_cc = np.compress(mask, bboxes_cc, axis=0).astype(np.int32)

        cc = np.compress(mask, bboxes_cc, axis=0).astype(np.int32)

        # logger.info(f"cc: {cc}, noise_cc: {noise_cc}")

        if self.output2:
            from services.output_service import save_shapes
            worker_name = context.get("worker_name") or "restorer"
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            output_paths = context["output_paths"]

            
            if self.output2:
                contours1: List[np.ndarray[Any, Any]] = []
                for item in noise_cont.values():
                    contour1 = np.array(item["cont_coords"], dtype=np.int32)
                    contours1.append(contour1)

                contours2: List[np.ndarray[Any, Any]] = []
                for item in blobs_cont.values():
                    contour2 = np.array(item["cont_coords"], dtype=np.int32)
                    contours2.append(contour2)

                # logger.info(f"Comparativa CONT: {contours2}")
                                    
                save_shapes(image_name, poly_id, cropped_img, output_paths, worker_name, contours1, contours2, method="contours")

            if self.output2:
                noise_cc_reshaped = noise_cc.reshape(-1, 2)
                cc_reshaped=cc.reshape(-1, 2)
                contours1: List[np.ndarray[Any, Any]]  = list(noise_cc)
                contours2: List[np.ndarray[Any, Any]]  = list(cc_reshaped)
                logger.info(f"SHAPES CC: {cc_reshaped.shape, noise_cc.shape}")
                # logger.info(f"Comparativa: CC: {cc}, Reshaped: {cc_reshaped}")

                cropped_image1 = cropp_img(bin_img, noise_cc)
                # cropped_image2 = cropp_img(bin_img, cc_reshaped)
                save_shapes(image_name, poly_id, cropped_image1, output_paths, worker_name, contours1, contours2, method="cc")
                # save_shapes(image_name, poly_id, cropped_image2, output_paths, worker_name, contours1, contours2, method="reshaped")



        return noise_cc, cc, noise_cont, blobs_cont