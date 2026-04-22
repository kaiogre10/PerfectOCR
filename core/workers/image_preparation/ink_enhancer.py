# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
# import time
from typing import Dict, Any, List, Set
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import make_contiguous, get_contours_values
from core.utils.math_utils import soft_histogram
from services.output_service import save_shapes, save_croped_image

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('ink_enhancement', {})
        self.white = worker_config["white"]
        self.black = worker_config["black"]
        self.aspect_ratio_range = worker_config["aspect_ratio_range"]
        self.angle_threshold = worker_config.get("angle_threshold")
        self.thr = worker_config.get("thr")
        self.black_thr = worker_config.get("black_thr")
        self.solid_thr = worker_config.get("solid_thr")
        self.shape_thr = worker_config.get("shape_thr")
        self.output = config.get("bin_full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        # start_time = time.perf_counter()
        try:
            logger.debug("Mejoramiento de tinta empezado con éxito")
            
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            context["image_name"]= image_name

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False

            correct, out_conts = self.enhance_ink(full_img)

            # gap_img, white_gaps, black_gaps = self.fill_gaps(full_img)
            # all_gaps = white_gaps.copy()
            # all_gaps.extend(black_gaps.copy())
            # bin_gap = binarice_img(gap_img.copy(), {})
            # correct, out_conts, out_conts2 = self.delete_outliersvec(full_img)
            # all_outliers = out_conts.copy()
            # all_outliers.extend(out_conts2.copy())
            # bin_correct = binarice_img(correct.copy(), {})
            # corrected = cv2.morphologyEx(correct, cv2.MORPH_CLOSE, self.kernelr, iterations=2, borderType=cv2.BORDER_REFLECT)

            #self.refine_text_quality(grey_img.copy(), context, image_name)
                
            if manager.update_full_img(True, correct):
                # logger.info(f"Corrección de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                if self.output:
                    worker_name = context.get("worker_name") or "inker"
                    
                    imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                    # image_id = f"outliers_{image_name}_{worker_name}"
                    # id = f"bin_gap_{image_name}_{worker_name}"
                    # gaps_id = f"gaps_{image_name}_{worker_name}"
                    # img_id = f"vec_correct_{image_name}_{worker_name}"
                    # all_cont_id = f"all_contours_{image_name}_{worker_name}"
            
                    save_croped_image(image_name, imag_id, correct, worker_name)
                    # save_croped_image(image_name, id, bin_gap, worker_name)
                    # save_croped_image(image_name, img_id, correctvect, worker_name)
                    
                    # save_shapes(image_name, gaps_id, full_img,  scan_cont, scan_cont2)
                    # save_shapes(image_name, image_id, full_img, out_conts, [])

                    # save_shapes(image_name, all_cont_id, full_img, output_paths, all_gaps, all_outliers)
                            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
        return True
            
    def enhance_ink(self, full_img: np.ndarray[Any, Any]):
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = get_contours_values(grey_img)
        
        cont_area = metrics[:, 1]
        rect_height = metrics[:, 3]
        rect_area = metrics[:, 5]
        cont_perim = metrics[:, 6]
        convex_area= metrics[:, 7]
        convex_perim = metrics[:, 8]
        
        bbox_widith = metrics[:, 9]
        bbox_height = metrics[:, 10]
        
        black_pixels = metrics[:, 11]
        total_pixels = metrics[:, 12]
        has_childs = metrics[:, -1]
        _, relat = soft_histogram(cont_area)
        
        min_areas_mask = np.percentile(cont_area, relat) 
        # top_areas = np.argsort(metrics[:, 1])[::-1]
        
        min_areas = np.where(min_areas_mask >= cont_area)[0]
        
        area_bbox = bbox_height * bbox_widith
        white_pixels = total_pixels - black_pixels

        # out_idx = (top_areas >= total_outliers) & (has_childs ==False) & (white_pixels > black_pixels)
        # shape_mask = (cont_area > rect_area)
        total_black = (black_pixels == total_pixels) & (convex_area > cont_area)
        # idx_black = metrics[total_black, 0]
        # half_boox = area_bbox / 2
        # color_mask = (white_pixels > half_boox) | (black_pixels < white_pixels) | (black_pixels < 2)
        idx_black = metrics[total_black, 0]
        idx_areas = metrics[min_areas]

        outlier_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct = 0
        for idx, cont_coords in cont_coords_list:
            if idx in idx_black:
                # cont = cont_coords[idx]
                cv2.drawContours(grey_img, [cont_coords], -1, self.black, thickness=cv2.FILLED)
                outlier_cont.append(cont_coords)
                lines_correct += 1
            elif idx in idx_areas:
                # cont = cont_coords[idx]
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont.append(cont_coords)
                lines_correct += 1
                outlier_cont.append(cont_coords)
      #  logger.info(f"RUIDO: {lines_correct}")
        return make_contiguous(grey_img), outlier_cont
        points = metrics[:, [-5, -6, -7, -8]]
        # logger.info("BLANCOS DENTRO:\n"f"{points}")
        maskblasj = metrics[:, -7] < 255
        nomer = metrics[:, -6] < 255
        white = metrics[:, -8] == 0
        white2 = metrics[:, -5] > 0 
        non_nlacj = nomer & maskblasj & white & white2
        id = np.where(non_nlacj)[0]
        metrics = metrics[id]
        lines_correct = 0
        outlier_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        for idx, cont_coords in cont_coords_list:
            if idx in id:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont.append(cont_coords)
                lines_correct += 1
                continue
        return grey_img, outlier_cont, []

        
    def delete_outliersvec(self, full_img: np.ndarray[Any, Any]):
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = get_contours_values(grey_img)
        
        # logger.info(
        #     f"Features cv2 Vectorizadas | SHAPE:{metrics.shape}\n"
        #     f"{np.array2string(metrics[:, -1:-8:], precision=2, suppress_small=True)}"
        # )
        area_hist = soft_histogram(metrics[:, 1])

        area_outliers = area_hist[0] if area_hist[0] > 0 else 1

        # 1. Outliers de Área
        top_areas = np.sort(metrics[:, 1])[::-1][:area_outliers]
        outlier_mask = (metrics[:, 1] >= (np.min(top_areas) - 0.1))
        child_metric = metrics[outlier_mask]
        # Indexación booleana directa a la columna 0
        child_metrics = child_metric[child_metric[:, 11] == 1, 0] 
        out_index: Set[int] = set(child_metrics.astype(np.int8).tolist())

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
        
        rect1 = metrics[ratio_1]
        rect1 = rect1[rect1[:, 11] == 1, 0]

        # Transformación a Sets
        rect1ind: Set[int] = set(rect1.astype(np.int16).tolist())
        solidity_indices: Set[int] = set(solidity.astype(np.int16).tolist())
        vertical_indices: Set[int] = set(vertical.astype(np.int16).tolist())
        lines_indices: Set[int] = set(lines.astype(np.int16).tolist())
        irreg_indices: Set[int] = set(irreg.astype(np.int16).tolist())

        outlier_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        outlier_cont2: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct = 0

        # Unimos las condiciones en un único set O(1) de chequeo rápido
        group_outliers = irreg_indices | rect1ind | vertical_indices | lines_indices | solidity_indices 
        try:
            for idx, cont_coords in cont_coords_list:
                if idx in out_index:
                    cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                    outlier_cont.append(cont_coords)
                    lines_correct += 1
                    continue
                elif idx in group_outliers:
                    # cont = cont_coords[idx]
                    cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                    outlier_cont.append(cont_coords)
                    lines_correct += 1
                outlier_cont2.append(cont_coords)
        except cv2.error as e:
            logger.info(f"ERROR DIBUJANDO CONTORNOS: '{e}'", exc_info=True)
        logger.info(f"Outliers: {lines_correct}")
        return make_contiguous(grey_img), outlier_cont, outlier_cont2
        
    def fill_gaps(self,  full_img: np.ndarray[Any, Any]):
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = get_contours_values(grey_img)
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
