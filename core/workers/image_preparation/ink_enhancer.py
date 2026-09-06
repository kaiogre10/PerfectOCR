# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
from domain.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import make_contiguous, get_contours_values
from utils.math_utils import soft_histogram
from services.output_service import save_shapes
from core.assets.assets import WHITE, BLACK

_white = WHITE
_black = BLACK

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('ink_enhancement', {})
        self.metric = worker_config.get("manhattan", "manhattan")
        self.aspect_ratio_range = worker_config["aspect_ratio_range"]
        self.angle_threshold = worker_config.get("angle_threshold")
        self.thr = worker_config.get("thr")
        self.black_thr = worker_config.get("black_thr")
        self.solid_thr = worker_config.get("solid_thr")
        self.shape_thr = worker_config.get("shape_thr")
        self.output = config.get("bin_full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter):
        """Detecta y restaura texto con tinta gastada."""
        # start_time = time.perf_counter()
        try:
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            context["image_name"]= image_name
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False

            correct, out_conts, out_conts2 = self.delete_outliersvec(full_img)
            # correct, out_conts, out_conts2 = self.enhance_ink(full_img)
            # correct, out_conts, out_conts2 = self.fill_gaps(full_img)

            if manager.update_full_img(True, full_img):
                # logger.info(f"Corrección de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                if self.output:
                    
                    imag_id = f"corrected_blobs"
                    image_id = f"outliers"
                    # id = f"bin_gap_{image_name}_{worker_name}"
                    # gaps_id = f"gaps_{image_name}_{worker_name}"
                    # img_id = f"vec_correct_{image_name}_{worker_name}"
                    # all_cont_id = f"all_contours_{image_name}_{worker_name}"
            
                    # save_croped_image(image_name, imag_id, correct, worker_name)
                    # save_croped_image(image_name, id, bin_gap, worker_name)
                    # save_croped_image(image_name, img_id, correctvect, worker_name)
                    
                    # save_shapes(image_name, gaps_id, full_img,  scan_cont, scan_cont2)
                    save_shapes(image_name, image_id, full_img, out_conts, out_conts2)

                    # save_shapes(image_name, all_cont_id, full_img, output_paths, all_gaps, all_outliers)
                            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
        return True
            
    def enhance_ink(self, full_img: np.ndarray[Any, Any]):
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = get_contours_values(grey_img)
        idx = metrics[:, 0]
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

        _, perc_val = soft_histogram(metrics[: ,1])
        min_areas_mask = np.percentile(cont_area, perc_val)
        top_areas = np.argsort(metrics[:, 1])[::-1]
        
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
        # logger.info(f"RUIDO: {lines_correct}")

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
                cv2.drawContours(grey_img, [cont_coords], -1, _white, thickness=cv2.FILLED)
                outlier_cont.append(cont_coords)
                lines_correct += 1
                continue
        return grey_img, outlier_cont, []

    def delete_outliersvec(self, full_img: np.ndarray[Any, Any]):
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = get_contours_values(grey_img)
        cont_area = metrics[:, 1]
        outer = metrics[:, -3]
        hollow_outer = metrics[:, -2]
        solid_outer = metrics[:, -1]

        area_hist = soft_histogram(cont_area)

        area_outliers = area_hist[0] if area_hist[0] > 0 else 1

        # 1. Outliers de Área
        top_areas_idx = np.argpartition(cont_area, -area_outliers)[-area_outliers:]
        top_areas = cont_area[top_areas_idx]
        top_areas_idx = (cont_area >= np.min(top_areas))
        non_child_idx = (hollow_outer == 1)
        outlier_idx = np.where(top_areas_idx & non_child_idx)[0]
        #logger.info(f"{outlier_idx} shape: {outlier_idx.size}")
        if outlier_idx.size < 1:
        #    logger.info(f"SIN OUTLIERS")
            return grey_img, [], []
        # logger.info(f"TOP AREAS: {top_areas}\n"f" IDX: {top_areas_idx}\n"f"outlier_mask: {outlier_idx}")

        # Indexación booleana directa a la columna 0

        # # 3. Shape y Líneas
        # shape_mask = self.shape_thr > metrics[:, 16]
        # irreg = metrics[shape_mask, 0]
        #
        # aspc_ratio_mask_high = (metrics[:, 10] > self.aspect_ratio_range[1])
        # lines = metrics[aspc_ratio_mask_high, 0]
        #
        # # 4. Angulo y Deskew
        # angle_mask1 = (metrics[:, 4] < self.angle_threshold)
        # angle_mask2 = (metrics[:, 4] > (180 - self.angle_threshold))
        #
        # mask_deskew = angle_mask2 | angle_mask1
        #
        # vert_metrics = metrics[~mask_deskew]
        # mask_vertical = (vert_metrics[:, 10] > self.aspect_ratio_range[1])
        # vertical = vert_metrics[mask_vertical, 0]
        #
        # metrics = metrics[mask_deskew]
        #
        # # 5. Black Ratio, Solidez y Ratio Shapes
        # black_ratio = metrics[:, 15] / metrics[:, 14]
        # black_ratio_mask = (black_ratio > self.black_thr)
        #
        # solidez = (metrics[:, 1] / metrics[:, 7])
        # solid = (solidez > self.solid_thr)
        #
        # aspc_ratio_mask_low = (metrics[: ,10] > self.aspect_ratio_range[0])
        # mask_solidity = solid & aspc_ratio_mask_low
        #
        # solidity = metrics[mask_solidity, 0]
        #
        # ratio_1 = (metrics[:, 12] < (1.0 + self.thr)) & (metrics[:, 12] > (1.0 - self.thr)) & solid & black_ratio_mask
        #
        # rect1 = metrics[ratio_1]
        # rect1 = rect1[rect1[:, 11] == 1, 0]
        #
        # # Transformación a Sets
        # rect1ind: Set[int] = set(rect1.astype(np.int16).tolist())
        # solidity_indices: Set[int] = set(solidity.astype(np.int16).tolist())
        # vertical_indices: Set[int] = set(vertical.astype(np.int16).tolist())
        # lines_indices: Set[int] = set(lines.astype(np.int16).tolist())
        # irreg_indices: Set[int] = set(irreg.astype(np.int16).tolist())

        outlier_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        # outlier_cont2: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct = 0

        # Unimos las condiciones en un único set O(1) de chequeo rápido
        # group_outliers = irreg_indices | rect1ind | vertical_indices | lines_indices | solidity_indices
        try:
            for idx, cont_coords in cont_coords_list:
                if idx in outlier_idx:
                    cv2.drawContours(grey_img, [cont_coords], -1, _white, thickness=cv2.FILLED)
                    outlier_cont.append(cont_coords)
                    lines_correct += 1
                    continue
                # elif idx in group_outliers:
                #     cont = cont_coords[idx]
                #     cv2.drawContours(grey_img, cont, -1, _white, thickness=cv2.FILLED)
                #     outlier_cont.append(cont_coords)
                #     lines_correct += 1
                # outlier_cont2.append(cont_coords)
        except cv2.error as e:
            logger.error(f"ERROR DIBUJANDO CONTORNOS: '{e}'", exc_info=True)
        #logger.info(f"Outliers: {lines_correct}, size: {outlier_idx.size}")
        return make_contiguous(grey_img), outlier_cont, []
        
    def fill_gaps(self, full_img: np.ndarray[Any, Any]):
        time_0 = time.perf_counter()
        grey_img = make_contiguous(full_img)
        cont_coords_list, metrics = get_contours_values(grey_img)
        has_childs = metrics[:, -1]
        x = metrics[:, 10]
        y = metrics[:, 11]
        w = metrics[:, 12]
        h = metrics[:, 13]

        width = (w - x)
        height = (h - y)
        
        black_pixels = metrics[:, -2]
        total_inside = metrics[:, -1]

        non_child_mask = (has_childs < 1)
        white_pixels = total_inside - black_pixels

        ink = (black_pixels == total_inside)
        background = (white_pixels == total_inside)

        density = np.zeros_like(total_inside, dtype=np.float32)
        nonzero_mask = total_inside > 0
        density[nonzero_mask] = black_pixels[nonzero_mask] / total_inside[nonzero_mask]

        to_dbsca = metrics[:, 1:-3]
        labels = h_density_cluster(to_dbsca, h_min_samples=2, h_metric=self.metric)
        # idx = metrics[:, 0]
        # mapped = np.column_stack([idx, labels])
        noise_mask = (labels < 0) & non_child_mask & (ink==False)
        noise_idx = np.where(noise_mask)[0]

        unique_labels, counts = np.unique(labels, return_counts=True)
        # logger.info("\n"f"UNIQUE:\n"f"{unique_labels}\n"f"COUNTS:\n"f"{counts}")

        cluster_id = np.argmax(counts)
        mask_cluster = (labels == unique_labels[cluster_id]) & non_child_mask & (ink==False)

        noise_idx1 = np.where(mask_cluster)[0]

        # logger.info(f"Clusters detectados: {unique_labels.size} | TOP CLUSTER: {np.amax(counts)}")  # | Distribución:\n"f"{noise_idx1}")

        outlier_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        outlier_cont2: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct = 0

        for idx, cont_coords in cont_coords_list:
            if idx in noise_idx1:
                cv2.drawContours(grey_img, [cont_coords], -1, _black, thickness=cv2.FILLED)  # Rojo
                outlier_cont.append(cont_coords)
                lines_correct += 1

            elif idx in noise_idx:
                cv2.drawContours(grey_img, [cont_coords], -1, _white, thickness=cv2.FILLED)   # Azul
                outlier_cont2.append(cont_coords)
                lines_correct += 1
        
        #logger.info(f"Se filtraron {lines_correct} objetos en: {time.perf_counter() - time_0}'s")
        return make_contiguous(grey_img), outlier_cont, outlier_cont2
