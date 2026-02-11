# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple, Set
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_contours_metrics, extract_contours_histogram
from services.output_service import save_croped_image, save_shapes
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('ink_enhancement', {})
        self.noise_kernel= np.array(worker_config["noise_kernel"]).astype(np.uint8)
        self.white = list(worker_config["white"]) or [255, 255, 255]
        self.black = list(worker_config["black"]) or [0, 0, 0]
        self.restorer_kernel = np.array(worker_config["restorer_kernel"]).astype(np.uint8)
        self.isolation_range = worker_config["isolation_range"]
        self.start_restoring = worker_config.get("start_restoring")
        self.iterations: int = worker_config.get("iterations") or 1
        self.aspect_ratio_range: Tuple[float, float] = worker_config["aspect_ratio_range"]
        self.angle_threshold: float = worker_config.get("angle_threshold", {})
        self.output = config.get("bin_full_img")
        self.kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))
        self.kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))

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

            correct, out_conts, out_conts2 = self.delete_outliers(full_img)
            # lines_cont, angle_cont, corrected = self.compare_areas(outed_img)
            # correct, black_gaps, white_gaps = self.fill_gaps(corrected)
            # corrected = cv2.morphologyEx(correct, cv2.MORPH_CLOSE, self.kernelr, iterations=2, borderType=cv2.BORDER_REFLECT)

            #self.refine_text_quality(grey_img.copy(), context, image_name)
            # all_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
            # all_gaps: List[np.ndarray[Any, np.dtype[np.int32]]] = []
            if not manager.update_full_img(True, correct):

                logger.warning("No se actualizo imagen en escala de grises del enhancer", exc_info=True)
                return False    
            else:
                logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                if self.output:
                    worker_name = context.get("worker_name") or "inker"
                    output_paths = context["output_paths"]
                    
                    imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                    image_id = f"outliers_{image_name}_{worker_name}"
                    line_cont_id= f"lines_{image_name}_{worker_name}"
                    gaps_id = f"gaps_{image_name}_{worker_name}"
                    #img_id= f"open_{image_name}_{worker_name}"
            
                    save_croped_image(image_name, imag_id, correct, output_paths, worker_name)
                    #save_croped_image(image_name, img_id, eroded, output_paths, worker_name)
                    # lines_cont.extend(angle_cont)
                    # black_gaps.extend(white_gaps)
                    # all_cont.extend(lines_cont)
                    # all_gaps.extend(black_gaps)
                    
                    # save_shapes(image_name, gaps_id, full_img, output_paths, black_gaps, white_gaps)
                    save_shapes(image_name, image_id, full_img, output_paths, out_conts, out_conts2)
                    # save_shapes(image_name, line_cont_id, full_img, output_paths, lines_cont, contours2=angle_cont)
                            
                return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
        return True
    
    def delete_outliers(self, grey_img: np.ndarray[Any, Any]):
        logger.info("Eliiminando outliers")
        thr = 0.3
        black_thr = 0.65
        solid_thr = 0.80
        cont_coords_list, metrics = extract_contours_metrics(grey_img)
        hist_values =  extract_contours_histogram(metrics)
        area_outliers = hist_values[0]

        top_areas = np.sort(metrics[:, 1])[::-1][:area_outliers]
        outlier_mask = (metrics[:, 1] > np.min(top_areas))
        child_metric = np.compress(outlier_mask, metrics, 0)
        child_metrics = np.compress(child_metric[:, 11] == 1, child_metric[:, 0], 0)

        aspc_ratio_mask_high = (metrics[: ,10] > self.aspect_ratio_range[1])
        lines = np.compress(aspc_ratio_mask_high, metrics[:, 0])

        angle_to_horizontal = np.minimum(metrics[:, 4], 180.0 - metrics[:, 4])

        angle_mask1 = (angle_to_horizontal < self.angle_threshold) 
        angle_mask2 = (angle_to_horizontal > (180 - self.angle_threshold))
        
        mask_deskew = angle_mask2 | angle_mask1

        vert_metrics = np.compress(~mask_deskew, metrics, 0)
        
        mask_vertical = (vert_metrics[:, 10] > self.aspect_ratio_range[1])
        vertical = np.compress(mask_vertical, vert_metrics[:, 0])

        metrics = np.compress(mask_deskew, metrics, 0)

        black_ratio = metrics[:, 15] / metrics[:, 14]
        
        black_ratio_mask = (black_ratio > black_thr)

        solidez = (metrics[:, 1] / metrics[:, 7])
        solid = (solidez > solid_thr)

        aspc_ratio_mask_low = (metrics[: ,10] > self.aspect_ratio_range[0])

        mask_solidity = solid & aspc_ratio_mask_low

        solidity = np.compress(mask_solidity, metrics[:, 0], 0)

        ratio_1 = (metrics[:, 12] < (1.0 + thr)) & (metrics[:, 12] > (1.0 - thr)) & solid & black_ratio_mask
        ratio_2 =  (metrics[:, 13] < (1.0 + thr)) & (metrics[:, 13] > (1.0- thr)) & solid & black_ratio_mask
        
        rect1 = np.compress(ratio_1, metrics, 0)
        rect1 = np.compress(rect1[:, 11]==1, rect1[:,0], 0)

        rect2 = np.compress(ratio_2, metrics, 0)
        rect2 = np.compress(rect2[:, 11]==1, rect2[:,0], 0)

        out_index: Set[int] = set(child_metrics.astype(np.int32)) if len(child_metrics) > 0 else set()
        rect1ind: Set[int] = set(rect1.astype(np.int32)) if len(rect1) > 0 else set()
        rect2ind: Set[int] = set(rect2.astype(np.int32)) if len(rect2) > 0 else set()
        solidity_indices: Set[int] = set(solidity.astype(np.int32)) if len(solidity) > 0 else set()
        vertical_indices: Set[int] = set(vertical.astype(np.int32)) if len(vertical) > 0 else set()
        lines_indices: Set[int] = set(lines.astype(np.int32)) if len(lines) > 0 else set()

        outlier_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        outlier_cont2: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct = 0

        for idx, cont_coords in cont_coords_list:

            if idx in out_index:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont.append(cont_coords)
                lines_correct += 1
            
            elif idx in solidity_indices:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont2.append(cont_coords)
                lines_correct += 1

            if idx in rect1ind:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont2.append(cont_coords)
                lines_correct += 1

            if idx in rect2ind:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont2.append(cont_coords)
                lines_correct += 1

            if idx in vertical_indices:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont.append(cont_coords)
                lines_correct += 1

            if idx in lines_indices:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                outlier_cont2.append(cont_coords)
                lines_correct += 1

            else:
                continue

        logger.info(f"Outliers: {lines_correct}")
        return grey_img, outlier_cont2, outlier_cont

    def compare_areas(self, grey_img: np.ndarray[Any, Any]):
        logger.info("Eliiminando rayas")
        cont_coords_list, metrics = extract_contours_metrics(grey_img)
        hist_values =  extract_contours_histogram(metrics)
        perc_val = hist_values[1]
        
        # Corregir: usar & en lugar de and, y paréntesis correctos
        mask_lines = metrics[:, 10] > self.aspect_ratio_range[1]
        lines = np.compress(mask_lines, metrics[:, 0])
       
        # Corregir: comprimir sobre metrics, no sobre lines
        angle_to_horizontal = np.minimum(metrics[:, 4], 180.0 - metrics[:, 4])
        mask_deskew = (metrics[:, 10] > self.aspect_ratio_range[1]) & (angle_to_horizontal > self.angle_threshold)
        deskew = np.compress(mask_deskew, metrics[:, 0])

        angle_to_vertical = np.abs(metrics[:, 4] - 90.0)
        mask_vertical = (metrics[:, 10] > self.aspect_ratio_range[1]) & (angle_to_vertical > self.angle_threshold)
        vertical = np.compress(mask_vertical, metrics[:, 0])

        mask_solidity = metrics[:, 1] / metrics[:, 7]
        solidity = np.compress(mask_solidity < 0.85, metrics[:, 0])

        top_areas_list = np.sort(metrics[:, 1])[::-1]

        top_areas_no_out = 5
        mask = metrics[:,1] >= np.min(top_areas_list[:top_areas_no_out]) - 1
        top_area = np.compress(mask, metrics[:,0], 0)
        
        min_area = np.percentile(metrics[:, 1], perc_val)
        metrics = np.compress(min_area > metrics[:, 1], metrics, 0)

        # Extraer índices (primera columna) y convertir a set
        lines_indices: Set[int] = set(lines.astype(np.int32)) if len(lines) > 0 else set()
        deskew_indices: Set[int] = set(deskew.astype(np.int32)) if len(deskew) > 0 else set()
        vertical_indices: Set[int] = set(vertical.astype(np.int32)) if len(vertical) > 0 else set()
        solidity_indices: Set[int] = set(solidity.astype(np.int32)) if len(solidity) > 0 else set()
        
        top_ind: Set[int] = set(top_area.astype(np.int32)) if len(top_area) > 0 else set()
        
        angle_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct: int = 0
        for idx, cont_coords in cont_coords_list:
            # if idx in top_ind and idx not in solidity_indices:
            #     cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
            #     lines_cont.append(cont_coords)
            #     lines_correct += 1
                
            if idx in lines_indices:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                angle_cont.append(cont_coords)
                lines_correct += 1

            # elif idx in solidity_indices:
            #     cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
            #     angle_cont.append(cont_coords)
            #     lines_correct += 1
            
            elif idx in deskew_indices:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                angle_cont.append(cont_coords)
                lines_correct += 1

            elif idx in vertical_indices:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                angle_cont.append(cont_coords)
                lines_correct += 1
            else:
                continue

        logger.info(f"Rayas eliminados: {lines_correct}")
        # grey_img =cv2.morphologyEx(grey_img, cv2.MORPH_CLOSE, self.kernel1, iterations=1, borderType=cv2.BORDER_REPLICATE)
        return lines_cont, angle_cont, grey_img
        
    def fill_gaps(self, grey_img: np.ndarray[Any, Any]):
        # grey_img = cv2.morphologyEx(grey_img, cv2.MORPH_OPEN, self.kernelr, iterations=1, borderType=cv2.BORDER_REPLICATE)
        logger.info("Eliiminando gaps")

        cont_coords_list, metrics = extract_contours_metrics(grey_img)
        hist_values =  extract_contours_histogram(metrics)
        perc_val = hist_values[1]

        max_area = np.percentile(metrics[:, 1], perc_val)
        logger.info(f"HIS{hist_values}, PERCENTIL: {max_area}")
        metrics = np.compress(max_area > metrics[:, 1], metrics, 0)
        
        mask_lon = (metrics[:, 9] == 0)
        cont_array = np.compress(mask_lon, metrics.copy(), 0)
        lone_ind: Set[int] = set(cont_array[:, 0].astype(np.int32)) if len(cont_array[:, 0]) > 0 else set()

        metrics = np.compress(1==metrics[:, 8], metrics, 0)
        all_ind: Set[int] = set(metrics[:, 0].astype(np.int32)) if len(metrics[:, 0]) > 0 else set()
        white = 0
        black = 0

        white_gaps: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        black_gaps: List[np.ndarray[Any, np.dtype[np.int32]]] = []

        for idx, cont_coords in cont_coords_list:
            if idx in lone_ind:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                white_gaps.append(cont_coords)
                white += 1
            
            elif idx in all_ind:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                black_gaps.append(cont_coords)
                black += 1
            else:
                continue
                    
        logger.info(f"Contornos pintado de Ngero: {black}, solitarios: {white}")
        return grey_img, black_gaps, white_gaps
    
    def refine_text_quality(self,grey_img: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any], image_name: str):
        """
        Aplica limpieza adaptativa según tamaño de imagen.
        """
        try:
            cont_coords_list, metrics = extract_contours_metrics(grey_img, False)
            min_area = np.percentile(metrics[:, 1], 15)
            metrics = np.compress(min_area > metrics[:, 1], metrics, 0)

            lonely = metrics[:, 9]
            mask_lon = (lonely == 1)
            metrics = np.compress(mask_lon, metrics, 0)
            
            X: np.ndarray[Any, np.dtype[np.float32]] = metrics[:, 1:7].astype(np.float32)

            n_samples = X.shape[0]
            n_neighbors = min(5, n_samples)
            
            if n_samples < 2:
                logger.warning(f"Insuficientes contornos para análisis de vecindad: {n_samples}")
                return

            # Crear el modelo de vecinos más cercanos
            nearest = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nearest.fit(X)
            
            # Obtener distancias y índices de los vecinos más cercanos
            distance = nearest.kneighbors(X)
            avg_distances = distance[0].astype(np.float32)
            indices = distance[1].astype(np.int32)

            # avg_distances = np.mean(distances[:, 1:])

            if avg_distances.size > 0:
                logger.info(f"Distancia promedio a vecinos: min={avg_distances.min():.2f}, "
                        f"max={avg_distances.max():.2f}, mean={avg_distances.mean():.2f}")
            else:
                logger.warning("No hay suficientes vecinos para calcular distancias")
            
            worker_name = context.get("worker_name") or "inker"
            output_paths = context["output_paths"]
            
            # Convertir a BGR una sola vez
            grey_bgr = cv2.cvtColor(grey_img, cv2.COLOR_GRAY2BGR)
            
            # Obtener centroides
            centroids = metrics[:, 5:7].astype(np.int32)
            
            # Generar colores únicos para cada blob
            np.random.seed(42)
            colors = np.random.randint(50, 255, size=(len(metrics), 3), dtype=np.uint8)
            
            # ========== IMAGEN 1: K VECINOS MÁS CERCANOS ==========
            knn_img = grey_bgr.copy()
            
            for i in range(len(metrics)):
                center = tuple(centroids[i])
                color = tuple(int(c) for c in colors[i])
                
                cv2.circle(knn_img, center, 6, color, 3)
                
                for neighbor_idx in indices[i][1:]:
                    neighbor_center = tuple(centroids[neighbor_idx])
                    cv2.line(knn_img, center, neighbor_center, color, 2)
                    cv2.circle(knn_img, neighbor_center, 3, color, 2)
            
            knn_image_id = f"knn_{image_name}_{worker_name}"
            save_croped_image(image_name, knn_image_id, knn_img, output_paths, worker_name)
            logger.info(f"Imagen KNN guardada: {knn_image_id}")
            
            # ========== IMAGEN 2: BÚSQUEDA POR RADIO ==========
            search_radius = 50
            # if search_radius > 1:
            radio_distances, radio_search = nearest.radius_neighbors(X, radius=search_radius, return_distance=True, sort_results=True)
            
            radius_img = grey_bgr.copy()
            
            for i in range(len(metrics)):
                center = centroids[i]
                color = tuple(int(c) for c in colors[i])
                
                cv2.circle(radius_img, center, search_radius, color, 1)
                # cv2.circle(radius_img, center, 5, (0, 255, 0), -1)
                
                for neighbor_idx in radio_search[i]:
                    neighbor_idx = int(neighbor_idx)  # Convertir a entero nativo
                    if neighbor_idx != i:
                        neighbor_center = centroids[neighbor_idx]
                        # cv2.circle(radius_img, neighbor_center, 3, color, 2)
                        cv2.line(radius_img, center, neighbor_center, color, 2)
            
            radius_image_id = f"radius_{image_name}_{worker_name}"
            save_croped_image(image_name, radius_image_id, radius_img, output_paths, worker_name)
            logger.info(f"Imagen Radio guardada: {radius_image_id}")
        except Exception as e:
            logger.info(f"Error: {e}", exc_info=True)
    

