# PerfectOCR/core/workers/vectorial_transformation/linal_reconstructor.py
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from services.output_service import save_raw_json, save_croped_image
from core.utils.math_utils import define_intervals
from core.utils.text_validator import validate_text
from core.utils.image_utils import cropp_img

logger = logging.getLogger(__name__)

class LinealReconstructor(VectorizationAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('lineal', {})
        self.overlap_threshold = self.worker_config.get('overlap_threshold')
        self.output = config.get("reconstructed_lines", False)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time = time.perf_counter()
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False

            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False
            
            lines_info= self._reconstruct_lines(polygons, context, full_img)
            if not lines_info:
                logger.error("LinealReconstructor: Error al guardar lineas de texto en el workflowdict")
                return False
            
            logger.info(f"'{len(lines_info)}' líneas amadas en {time.perf_counter() - start_time:.10f}")

            success = manager.create_text_lines(lines_info)
            if success:
                logger.debug(f"Lineas guardads correctamente en el manager")
                if self.output:
                    
                    file_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    worker_name = context.get("worker_name") or "lineal"
                    output_paths = context["output_paths"]
                    save_raw_json( output_paths, worker_name, lines_info, file_name)

                return True
                            
        except Exception as e:
            logger.error(f"error {e}", exc_info=True)
        return False
        
    def _reconstruct_lines(self, polygons: Dict[str, Polygons], context: Dict[str, Any], full_img: np.ndarray[Any, np.dtype[np.uint8]]) -> Optional[Dict[str, Any]]:
        """
        Reconstruye líneas agrupando polígonos y devuelve un dict con la debug completa de cada línea,
        incluyendo los textos OCR concatenados.
        """
        prepared_sorted = sorted(
            polygons.values(),
            key=lambda p: p.geometry.centroid[1]
        )
                
        lines_info: Dict[str, Any] = {}
        current_line_polys: List[Polygons] = []
        current_line_bbox: Optional[List[float]] = None
        line_counter = 0

        bboxes: List[np.ndarray[Any, Any]] = []        
        lines_bbox: List[Any] = []
        for poly in prepared_sorted:
            bbox = poly.geometry.bounding_box
            if bbox.size == 0:
                continue

            bboxes.append(bbox)

            if not current_line_polys or current_line_bbox is None:
                current_line_polys = [poly]
                current_line_bbox = list(bbox)
            else:
                y1_min, y1_max = current_line_bbox[1], current_line_bbox[3]
                y2_min, y2_max = bbox[1], bbox[3]
                overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                min_h = min(y1_max - y1_min, y2_max - y2_min)
                overlap = overlap_abs / min_h if min_h > 1e-5 else 0.0

                if overlap > self.overlap_threshold:
                    current_line_polys.append(poly)
                    all_bboxes = [p.geometry.bounding_box for p in current_line_polys]
                    if all_bboxes:
                        all_xs = [b[0] for b in all_bboxes] + [b[2] for b in all_bboxes]
                        all_y_mins = [b[1] for b in all_bboxes]
                        all_y_maxs = [b[3] for b in all_bboxes]
                        avg_y_min = sum(all_y_mins) / len(all_y_mins)
                        avg_y_max = sum(all_y_maxs) / len(all_y_maxs)
                        current_line_bbox = [min(all_xs), avg_y_min, max(all_xs), avg_y_max]
                        
                        # Antes de cerrar la línea, ordena los polígonos actuales de la línea por el eje X (centroide[0])
                        current_line_polys.sort(key=lambda p: p.geometry.centroid[0])
                else:
                    # Finaliza la línea actual y guarda la debug
                    polygon_ids = [p.polygon_id for p in current_line_polys]
                    texts = [p.ocr_text or "" for p in current_line_polys]
                    joined_text = " ".join(texts).strip()
                    
                    # Validar el texto antes de crear la entrada
                    if not validate_text(joined_text):
                        # Si no es válido, iniciar una nueva línea sin incrementar el contador
                        current_line_polys = [poly]
                        current_line_bbox = list(bbox)
                        continue
                    
                    lines_bbox.append(current_line_bbox)  # Agregar aquí: bbox de la línea completada
                    
                    # El centroide de la línea se calcula como el centroide del bounding box de la línea
                    line_centroid = [           
                            (current_line_bbox[0] + current_line_bbox[2]) / 2,
                            (current_line_bbox[1] + current_line_bbox[3]) / 2
                    ] if current_line_bbox else [0, 0]
                    
                    line_id = f"line_{line_counter:04d}"
                    lines_info[line_id] = {
                        "line_bbox": current_line_bbox,
                        "line_centroid": line_centroid,
                        "polygon_ids": polygon_ids,
                        "text": joined_text
                    }
                            
                    line_counter += 1
                    current_line_polys = [poly]
                    current_line_bbox = list(bbox)
                
                    # logger.info(f"{line_id}: '{joined_text}' | {polygon_ids}")
                    logger.info(f"{line_id}: '{joined_text}'")

        # Finaliza la última línea
        if current_line_polys:
            polygon_ids = [p.polygon_id for p in current_line_polys]
            texts = [p.ocr_text or "" for p in current_line_polys]
            joined_text = " ".join(texts).strip()
            
            # Validar también el texto de la última línea
            if validate_text(joined_text): #type: ignore
                current_line_polys.sort(key=lambda p: p.geometry.centroid[0])
                lines_bbox.append(current_line_bbox)
                
                line_centroid = [
                    (current_line_bbox[0] + current_line_bbox[2]) / 2,
                    (current_line_bbox[1] + current_line_bbox[3]) / 2
                ] if current_line_bbox else [0, 0]
                
                line_id = f"line_{line_counter:04d}"
                lines_info[line_id] = {
                    "line_bbox": current_line_bbox,
                    "line_centroid": line_centroid,
                    "polygon_ids": polygon_ids,
                    "text": joined_text
                }

                # logger.info(f"{line_id}: '{joined_text}' | {polygon_ids}")
                logger.info(f"{line_id}: '{joined_text}'")

        # lines_array = np.array(lines_bbox)
        # logger.info(f"ARRAY: {lines_array}, SHPAE: {lines_array.shape}")
        # logger.info(f"BBOXES: {lines_bbox}")
        # bboxes_array = np.array(bboxes)
        # logger.info(f"SHAE:{bboxes_array.shape}")
        # intervals = define_intervals(bboxes_array, overlap_threshold=0.50)
        # logger.info(f"Lineas vectorizadas: {intervals.reshape}")

        # image_name = "2"
        # worker_name = context.get("worker_name") or "lineal"
        # output_paths = context["output_paths"]
        # # for i in range(len(intervals)):
        #     image_id = f"vector_{image_name}_{worker_name}_{i+1}"
        #     full_array = cropp_img(full_img.copy(), intervals)
        #     save_croped_image(image_name, image_id, full_array, output_paths, worker_name)

        # for i in range(len(lines_array)):
        #     lines_full = cropp_img(full_img, lines_array)
            
        #     imge_id = f"clasic_{image_name}_{worker_name}_{i+1}"
        #     save_croped_image(image_name, imge_id, lines_full, output_paths, worker_name)
        return lines_info
