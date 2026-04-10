# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple, Set
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import extract_contours_metrics, make_contiguous
from core.utils.math_utils import extract_contours_histogram
from services.output_service import save_shapes, save_croped_image

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('ink_enhancement', {})
        self.white = list(worker_config["white"]) or [255, 255, 255]
        self.black = list(worker_config["black"]) or [0, 0, 0]
        self.aspect_ratio_range: Tuple[float, float] = worker_config["aspect_ratio_range"]
        self.angle_threshold: float = worker_config.get("angle_threshold", {})
        self.thr = worker_config.get("thr")
        self.black_thr = worker_config.get("black_thr")
        self.solid_thr = worker_config.get("solid_thr")
        self.shape_thr = worker_config.get("shape_thr")
        self.output = config.get("bin_full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.perf_counter()
            logger.debug("Mejoramiento de tinta empezado con éxito")
            
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            context["image_name"]= image_name

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False

            # enhanced, enhanced_cont = self.enhance_ink(full_img)

            # gap_img, white_gaps, black_gaps = self.fill_gaps(full_img)
            # all_gaps = white_gaps.copy()
            # all_gaps.extend(black_gaps.copy())
            # bin_gap = binarice_img(gap_img.copy(), {})
            correct, out_conts, out_conts2 = self.delete_outliers(full_img)
            # all_outliers = out_conts.copy()
            # all_outliers.extend(out_conts2.copy())
            # bin_correct = binarice_img(correct.copy(), {})
            # corrected = cv2.morphologyEx(correct, cv2.MORPH_CLOSE, self.kernelr, iterations=2, borderType=cv2.BORDER_REFLECT)

            #self.refine_text_quality(grey_img.copy(), context, image_name)
            if not manager.update_full_img(True, correct):

                logger.warning("No se actualizo imagen en escala de grises del enhancer", exc_info=True)
                return True
            else:
                logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                if self.output:
                    worker_name = context.get("worker_name") or "inker"
                    output_paths = context["output_paths"]
                    
                    imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                    image_id = f"outliers_{image_name}_{worker_name}"
                    # id = f"bin_gap_{image_name}_{worker_name}"
                    # gaps_id = f"gaps_{image_name}_{worker_name}"
                    # img_id = f"bin_correct_{image_name}_{worker_name}"
                    # all_cont_id = f"all_contours_{image_name}_{worker_name}"
            
                    save_croped_image(image_name, imag_id, correct, output_paths, worker_name)
                    # save_croped_image(image_name, id, bin_gap, output_paths, worker_name)
                    # save_croped_image(image_name, img_id, bin_correct, output_paths, worker_name)
                    
                    # save_shapes(image_name, gaps_id, full_img, output_paths, scan_cont, scan_cont2)
                    save_shapes(image_name, image_id, full_img, output_paths, out_conts, out_conts2)

                    # save_shapes(image_name, all_cont_id, full_img, output_paths, all_gaps, all_outliers)
                            
                return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
        return True

    def delete_outliers(self, full_img: np.ndarray[Any, Any]):
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = extract_contours_metrics(grey_img)
        # logger.info(f"Total de contornos outliers: {metrics.shape[0]}")
        area_hist = extract_contours_histogram(metrics[:, 1])
        dist_values = extract_contours_histogram(metrics[:, -1])

        area_outliers = area_hist[0] if area_hist[0] > 0 else 1
        dist_var_outliers = dist_values[0] if dist_values[0] > 0 else 1

        # 1. Outliers de Área
        # if area_outliers < 1:
        #     out_index = {-1}
        # else:
        top_areas = np.sort(metrics[:, 1])[::-1][:area_outliers]
        outlier_mask = (metrics[:, 1] >= (np.min(top_areas) - 0.1))
        child_metric = metrics[outlier_mask]
        # Indexación booleana directa a la columna 0
        child_metrics = child_metric[child_metric[:, 11] == 1, 0] 
        out_index: Set[int] = set(child_metrics.astype(np.int8).tolist())

        # 2. Varianza de Distancia
        med_short_side = np.median(metrics[:, 17])
        mad = np.median(np.abs(metrics[:, 17] - med_short_side))
        
        top_dist_var = np.sort(metrics[:, -1])[::-1][:dist_var_outliers]
        dist_outlier_mask = (metrics[:, -1] > (np.min(top_dist_var) - 0.1))

        dist_metrics = metrics[dist_outlier_mask]
        var_mask = ((dist_metrics[:, 10] > self.aspect_ratio_range[1]) & 
                    ((dist_metrics[:, 17] < (med_short_side - mad)) | 
                     (dist_metrics[:, 17] > (med_short_side + mad))))
        dist_var = dist_metrics[var_mask, 0]

        # 3. Shape y Líneas
        shape_mask = self.shape_thr > metrics[:, 16]
        irreg = metrics[shape_mask, 0]

        aspc_ratio_mask_high = (metrics[:, 10] > self.aspect_ratio_range[1])
        lines = metrics[aspc_ratio_mask_high, 0]

        # 4. Angulo y Deskew
        angle_mask1 = (metrics[:, 4] < self.angle_threshold)
        angle_mask2 = (metrics[:, 4] > (180 - self.angle_threshold))
        
        mask_deskew = angle_mask2 | angle_mask1

        vert_metrics = metrics[~mask_deskew]
        mask_vertical = (vert_metrics[:, 10] > self.aspect_ratio_range[1])
        vertical = vert_metrics[mask_vertical, 0]

        metrics = metrics[mask_deskew]

        # 5. Black Ratio, Solidez y Ratio Shapes
        black_ratio = metrics[:, 15] / metrics[:, 14]
        black_ratio_mask = (black_ratio > self.black_thr)

        solidez = (metrics[:, 1] / metrics[:, 7])
        solid = (solidez > self.solid_thr)

        aspc_ratio_mask_low = (metrics[: ,10] > self.aspect_ratio_range[0])
        mask_solidity = solid & aspc_ratio_mask_low

        solidity = metrics[mask_solidity, 0]

        ratio_1 = (metrics[:, 12] < (1.0 + self.thr)) & (metrics[:, 12] > (1.0 - self.thr)) & solid & black_ratio_mask
        ratio_2 = (metrics[:, 13] < (1.0 + self.thr)) & (metrics[:, 13] > (1.0 - self.thr)) & solid & black_ratio_mask
        
        rect1 = metrics[ratio_1]
        rect1 = rect1[rect1[:, 11] == 1, 0]

        rect2 = metrics[ratio_2]
        rect2 = rect2[rect2[:, 11] == 1, 0]

        # Transformación a Sets
        rect1ind: Set[int] = set(rect1.astype(np.int16).tolist())
        rect2ind: Set[int] = set(rect2.astype(np.int16).tolist())
        solidity_indices: Set[int] = set(solidity.astype(np.int16).tolist())
        vertical_indices: Set[int] = set(vertical.astype(np.int16).tolist())
        lines_indices: Set[int] = set(lines.astype(np.int16).tolist())
        irreg_indices: Set[int] = set(irreg.astype(np.int16).tolist())
        dist_indices: Set[int] = set(dist_var.astype(np.int16).tolist())

        outlier_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        outlier_cont2: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct = 0

        # Unimos las condiciones en un único set O(1) de chequeo rápido
        group_outliers2 = dist_indices | irreg_indices | rect1ind | rect2ind | vertical_indices | lines_indices | solidity_indices

        for idx, cont_coords in cont_coords_list:
            if idx in out_index:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont2.append(cont_coords)
                lines_correct += 1
            elif idx in group_outliers2:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont.append(cont_coords)
                lines_correct += 1

        # logger.info(f"Outliers: {lines_correct}")
        return make_contiguous(grey_img), outlier_cont, outlier_cont2
    
    def fill_gaps(self,  full_img: np.ndarray[Any, Any]):
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = extract_contours_metrics(grey_img)
        # logger.info(f"Total de contornos gaps: {metrics.shape[0]}")
        hist_values = extract_contours_histogram(metrics[: ,1])
        perc_val = hist_values[1]

        max_area = np.percentile(metrics[:, 1], perc_val)
        # logger.info(f"HIS{hist_values}, PERCENTIL: {max_area}")
        metrics = np.compress(max_area > metrics[:, 1], metrics, 0)
        
        mask_lon = (metrics[:, 9] == 0)
        cont_array = np.compress(mask_lon, metrics, 0)
        lone_ind: Set[int] = set(cont_array[:, 0].astype(np.int32)) if len(cont_array[:, 0]) > 0 else set()

        metrics = np.compress(1==metrics[:, 8], metrics, 0)
        all_ind: Set[int] = set(metrics[:, 0].astype(np.int32)) if len(metrics[:, 0]) > 0 else set()
        white = 0
        black = 0

        white_gaps: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        black_gaps: List[np.ndarray[Any, np.dtype[np.int32]]] = []

        for idx, cont_coords in cont_coords_list:
            if idx in lone_ind:
                cv2.drawContours(grey_img, [cont_coords], -1, self.black, thickness=cv2.FILLED)
                white_gaps.append(cont_coords)
                white += 1
            
            elif idx in all_ind:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                black_gaps.append(cont_coords)
                black += 1
            else:
                continue
                    
        logger.info(f"Contornos pintado de Ngero: {black}, solitarios: {white}")
        return grey_img, white_gaps, black_gaps
