# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple, Set
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_contours_metrics
from services.output_service import save_croped_image, save_shapes
from core.utils.image_utils import normalice_image
from core.utils.math_utils import dilate_contour
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('ink_enhancement', {})
        self.noise_kernel= np.array(worker_config["noise_kernel"]).astype(np.uint8)
        self.white = worker_config["white"]
        self.black = worker_config["black"]
        self.restorer_kernel = np.array(worker_config["restorer_kernel"]).astype(np.uint8)
        self.isolation_range = worker_config["isolation_range"]
        self.start_restoring = worker_config.get("start_restoring")
        self.iterations: int = worker_config.get("iterations") or 1
        self.threshold_black: int = worker_config.get("threshold_black")
        self.threshold_white: int = worker_config.get("threshold_white")
        self.aspect_ratio_range: Tuple[float, float] = worker_config["aspect_ratio_range"]
        self.angle_threshold = worker_config.get("angle_threshold")
        self.min_area = worker_config.get("min_area")
        self.output = config.get("bin_full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.perf_counter()
            logger.debug("Mejoramiento de tinta empezado con éxito")
            
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            context["image_name"]= image_name

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                
            grey_img = self._decolorate(full_img)
            # cv2.morphologyEx(grey_img, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            cont_coords_list, metrics = extract_contours_metrics(grey_img, True)

            lines_cont, angle_cont, grey_image = self.compare_areas(cont_coords_list, metrics, grey_img)
            #eroded = cv2.dilate(grey_image, kernel, iterations=1)
            # grey_image, white_gaps, black_gaps = self.fill_gaps(eroded, metrics, cont_coords_list)
            
            correct, contours_list, blacked_contours = self.alternate_restore(grey_image)

            #self.refine_text_quality(correct.copy(), context, image_name)
            
            if not manager.update_full_img(True, correct):

                logger.warning("No se actualizo imagen en escala de grises del enhancer", exc_info=True)
                return False    
            else:
                if self.output:
                    worker_name = context.get("worker_name") or "inker"
                    output_paths = context["output_paths"]
                    
                    imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                    image_id = f"contours_{image_name}_{worker_name}"
                    line_cont_id= f"lines_{image_name}_{worker_name}"
                    # gaps_id = f"gaps_{image_name}_{worker_name}"
                    #img_id= f"open_{image_name}_{worker_name}"
            
                    save_croped_image(image_name, imag_id, correct, output_paths, worker_name)
                    #save_croped_image(image_name, img_id, eroded, output_paths, worker_name)
                    
                    # save_shapes(image_name, gaps_id, grey_img, output_paths, white_gaps, black_gaps)
                    save_shapes(image_name, image_id, grey_img, output_paths, contours_list, contours2=blacked_contours)
                    save_shapes(image_name, line_cont_id, grey_img, output_paths, lines_cont, contours2=angle_cont)
                            
                logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False
        
    def alternate_restore(self, grey_img: np.ndarray[Any, Any]):
        """
        Por cada iteración ejecuta 2 fases (clean/restore) en el orden dictado por start_restoring.
        metrics: se calcula UNA sola vez por iteración.
        Los contornos procesados en una fase NO se reprocesarán en la otra dentro de la misma iteración.
        """
        img_dims = grey_img.shape
        contours_list: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        blacked_contours: List[np.ndarray[Any, np.dtype[np.int32]]] = []

        restorer_kernel = self.restorer_kernel.copy()
        restorer_kernel[restorer_kernel % 2 == 0] += 1

        noise_kernel = self.noise_kernel.copy()
        noise_kernel[noise_kernel % 2 == 0] += 1

        for i in range(self.iterations):
            valid_coords, metrics = extract_contours_metrics(grey_img, False)

            noise_bin = np.percentile(metrics[:, 1], 5)
            # Filtrar por noise_bin
            noise_coords = metrics[metrics[:, 1] < noise_bin]
            noise_ids = noise_coords[:, 0].astype(np.int32)
            
            # Crear candidates usando los índices correctos
            noise_candidates: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = []
            for metric_idx in noise_ids:
                # Buscar en valid_coords el que tenga ese índice
                for vc_idx, (vc_id, vc_coords) in enumerate(valid_coords):
                    if vc_id == metric_idx:
                        noise_candidates.append((vc_id, vc_coords))
                        break

            condition = (i == 0)
            phases = ("restore", "clean") if self.start_restoring else ("clean", "restore")

            # Track contornos ya procesados EN ESTA ITERACIÓN (solo si fueron MODIFICADOS)
            processed_ids: Set[int] = set()

            for phase in phases:
                # Filtra candidatos que ya fueron procesados (modificados) en esta iteración
                remaining_candidates = [(idx, coords) for idx, coords in noise_candidates if idx not in processed_ids]

                if not remaining_candidates:
                    continue

                if phase == "clean":
                    kernel = np.where(condition, noise_kernel, (noise_kernel + (i * 2)))
                    grey_img, c_list, modified_ids = self._restore_faded_ink(
                        grey_img,
                        self.isolation_range[1],
                        kernel,
                        img_dims,
                        remaining_candidates,
                    )
                    contours_list.extend(c_list)
                    processed_ids.update(modified_ids)
                else:
                    grey_img_inv = 255 - grey_img
                    kernel = np.where(condition, restorer_kernel, (restorer_kernel + (i * 2)))
                    grey_img_inv, c_list, modified_ids = self._restore_faded_ink(grey_img_inv, self.isolation_range[0], kernel, img_dims, remaining_candidates)
                    blacked_contours.extend(c_list)
                    processed_ids.update(modified_ids)
                    grey_img = 255 - grey_img_inv

        return grey_img, contours_list, blacked_contours

    def _restore_faded_ink(self, gray_img: np.ndarray[Any, Any], isolation_range: int, kernel_shape: np.ndarray[Any, Any], img_dims: Any, noise_candidates: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]],) -> Tuple[np.ndarray[Any, Any], List[np.ndarray[Any, np.dtype[np.int32]]], Set[int]]:
        """
        Restaura la intensidad del texto y elimina el ruido aislado.

        Importante:
        - Un contorno se considera "procesado" solo si realmente se MODIFICA (se pinta).
        """
        img_h, img_w = img_dims
        contours_list: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        modified_ids: Set[int] = set()

        for idx, cont_coords in noise_candidates:
            # 1. Expandir
            cont_expanded = dilate_contour(cont_coords, kernel_shape)

            # 2. ROI basada en el expandido
            x_min, x_max = cont_expanded[:, 0].min(), cont_expanded[:, 0].max()
            y_min, y_max = cont_expanded[:, 1].min(), cont_expanded[:, 1].max()

            win_x1, win_y1 = max(0, x_min), max(0, y_min)
            win_x2, win_y2 = min(img_w, x_max + 1), min(img_h, y_max + 1)

            window_gray = gray_img[win_y1:win_y2, win_x1:win_x2]
            offset = np.array([[win_x1, win_y1]])

            # 3. Máscaras en coords de ROI
            cont_orig_roi = cont_coords - offset
            cont_exp_roi = cont_expanded - offset

            mask_orig = np.zeros(window_gray.shape, dtype=np.uint8)
            mask_exp = np.zeros(window_gray.shape, dtype=np.uint8)

            cv2.drawContours(mask_orig, [cont_orig_roi], -1, self.white, cv2.FILLED)
            cv2.drawContours(mask_exp, [cont_exp_roi], -1, self.white, cv2.FILLED)

            # 4. Anillo
            mask_ring = mask_exp & ~mask_orig
            ring_pixels = window_gray[mask_ring > 0]

            if ring_pixels.size == 0:
                continue

            window_mean = np.median(ring_pixels)

            # 5. Si cumple → pintar el EXPANDIDO (esto es una MODIFICACIÓN real)
            if window_mean > isolation_range:
                modified_ids.add(int(idx))

                contours_list.append(cont_expanded)
                cv2.drawContours(gray_img, [cont_expanded], -1, self.white, cv2.FILLED)

                contours_list.append(cont_coords)
                cv2.drawContours(gray_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)

        return gray_img, contours_list, modified_ids

    def compare_areas(self, cont_coords_list: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]], metrics: np.ndarray[Any, np.dtype[np.float32]] , grey_img: np.ndarray[Any, Any]):

        angle_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_cont: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        lines_correct: int = 0

        # Normaliza ángulos: si width < height, suma 90
        angle_norm = np.where(metrics[:, 2] < metrics[:, 3], metrics[:, 4] + 90, metrics[:, 4])
        angle_norm = angle_norm % 180.0

        # Si quieres actualizar la matriz:
        metrics[:, 4] = angle_norm
        aspect_ratio = (np.maximum(metrics[:, 2], metrics[:, 3]) / np.minimum(metrics[:, 2], metrics[:, 3])).astype(np.float32)
        
        # Corregir: usar & en lugar de and, y paréntesis correctos
        mask_lines = aspect_ratio > self.aspect_ratio_range[1]
        lines = np.compress(mask_lines, metrics[:, 0])
        
        # Corregir: comprimir sobre metrics, no sobre lines
        mask_deskew = (aspect_ratio > self.aspect_ratio_range[0]) & (metrics[:, 4] < self.angle_threshold)
        deskew = np.compress(mask_deskew, metrics[:, 0])

        mask_vertical = (aspect_ratio > self.aspect_ratio_range[1]) & (np.abs(metrics[:, 4] - 90) < self.angle_threshold)
        vertical = np.compress(mask_vertical, metrics[:, 0])

        # mask_solidity = metrics[:, 1] / metrics[:, 7]
        # solidity = np.compress(mask_solidity > 0.901, metrics[:, 0])

        # Extraer índices (primera columna) y convertir a set
        lines_indices: Set[int] = set(lines.astype(np.int32)) if len(lines) > 0 else set()
        deskew_indices: Set[int] = set(deskew.astype(np.int32)) if len(deskew) > 0 else set()
        vertical_indices: Set[int] = set(vertical.astype(np.int32)) if len(vertical) > 0 else set()
        # solidity_indices: Set[int] = set(solidity.astype(np.int32)) if len(solidity) > 0 else set()

        for idx, cont_coords in cont_coords_list:
            if idx in lines_indices:
                cv2.drawContours(grey_img, [cont_coords], -1, self.white, thickness=cv2.FILLED)
                lines_cont.append(cont_coords)
                lines_correct += 1
            
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

        logger.debug(f"Rayas eliminados: {lines_correct}")
                
        return lines_cont, angle_cont, grey_img
        
    def fill_gaps(self, gray_img: np.ndarray[Any, Any], metrics: np.ndarray[Any, Any], cont_coords_list: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]]):
        img_h, img_w = gray_img.shape
        min_areas = np.compress(self.min_area > metrics[:, 1], metrics, 0)
        all_ind: Set[int] = set(min_areas[:, 0].astype(np.int32))
        idx: Set[int] = set([c[0] for c in cont_coords_list])
        filtered = all_ind.intersection(idx)
        kernel_shape = np.array([1, 1]).astype(np.uint8)
        white = 0
        black = 0

        white_gaps: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        black_gaps: List[np.ndarray[Any, np.dtype[np.int32]]] = []

        for id in filtered:
            # Busca el contorno correspondiente
            cont = next((c[1] for c in cont_coords_list if c[0] == id), None)
            if cont is not None:
                cont_expanded = dilate_contour(cont, kernel_shape)
                x_min, x_max = cont_expanded[:, 0].min(), cont_expanded[:, 0].max()
                y_min, y_max = cont_expanded[:, 1].min(), cont_expanded[:, 1].max()
                
                win_x1, win_y1 = max(0, x_min), max(0, y_min)
                win_x2, win_y2 = min(img_w, x_max + 1), min(img_h, y_max + 1)
                
                window_gray = gray_img[win_y1:win_y2, win_x1:win_x2]
                offset = np.array([[win_x1, win_y1]])
                
                # 3. Máscaras en coords de ROI
                cont_orig_roi = cont - offset
                cont_exp_roi = cont_expanded - offset
                
                mask_orig = np.zeros(window_gray.shape, dtype=np.uint8)
                mask_exp = np.zeros(window_gray.shape, dtype=np.uint8)
                
                cv2.drawContours(mask_orig, [cont_orig_roi], -1, self.white, cv2.FILLED)
                cv2.drawContours(mask_exp, [cont_exp_roi], -1, self.white, cv2.FILLED)
                
                # 4. Anillo
                mask_ring = mask_exp & ~mask_orig
                ring_pixels = window_gray[mask_ring > 0]
                
                if ring_pixels.size == 0:
                    continue
                
                val_white = np.count_nonzero(ring_pixels > 127)
                val_black = np.count_nonzero(127 >= ring_pixels)
                if val_white < val_black:
                # Rellena el contorno en la imagen
                    cv2.drawContours(gray_img, [cont], -1, self.black, thickness=cv2.FILLED)
                    cv2.drawContours(gray_img, [cont_expanded], -1, self.black, thickness=cv2.FILLED)
                    black += 1
                    black_gaps.append(cont_expanded)
                    
                else:
                    cv2.drawContours(gray_img, [cont], -1, self.white, thickness=cv2.FILLED)
                    cv2.drawContours(gray_img, [cont_expanded], -1, self.white, thickness=cv2.FILLED)
                    white += 1
                    white_gaps.append(cont_expanded)
                    
        logger.info(f"Contornos pintado de blanco: {white} y Ngero: {black}")
        return gray_img, white_gaps, black_gaps
    
    def refine_text_quality(self, bin_img: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any], image_name: str):
        """
        Aplica limpieza adaptativa según tamaño de imagen.
        """
        # Combina contornos + CC
        cont_array_list, cont_array = extract_contours_metrics(bin_img, False)

        X = cont_array[:, 1:].astype(np.float32)

        n_samples = X.shape[0]
        n_neighbors = min(5, n_samples)  # No pedir más vecinos que muestras disponibles
        
        if n_samples < 2:
            logger.warning(f"Insuficientes contornos para análisis de vecindad: {n_samples}")
            return

        # Crear el modelo de vecinos más cercanos
        nearest = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nearest.fit(X)
        
        # Obtener distancias y índices de los vecinos más cercanos
        distances, indices = nearest.kneighbors(X)
        
        # Calcular distancia promedio al k-ésimo vecino más cercano
        # (excluyendo el primer vecino que es el mismo punto con distancia 0)
        if n_neighbors > 1:
            avg_distances = distances[:, 1:]
            logger.info(f"Distancia promedio a vecinos: min={avg_distances.min():.2f}, "
                    f"max={avg_distances.max():.2f}, mean={avg_distances.mean():.2f}")
        else:
            logger.warning("No hay suficientes vecinos para calcular distancias")
    
    # Opcional: encontrar vecinos dentro de un radio específico

        radio_search, distances = nearest.radius_neighbors(X, radius=250, return_distance=True, sort_results=True)

        # Encontrar contornos con pocos vecinos (posible ruido)
        neighbors_per_blob = [len(d) for d in distances]

        for idx, count in zip(cont_array[:, 0].astype(int), neighbors_per_blob):
            logger.debug(f"Blob {idx} tiene {count} vecinos dentro del radio.")


        for i, (idx, dist) in enumerate(zip(radio_search, distances)):
            if len(dist) > 0:
                max_dist = dist[-1]
                max_neighbor_idx = idx[-1]
                # logger.info(f"Blob {i}: vecino más lejano dentro del radio es {max_neighbor_idx} a distancia {max_dist}")

        neighbor_counts = np.array([len(neighbors) for neighbors in radio_search])
        isolated_mask = neighbor_counts < 2
        isolated_indices = cont_array[isolated_mask, 0].astype(np.int32)

        # logger.info(f"Contornos aislados detectados: {neighbor_counts}")
    
        isolated_set = set(isolated_indices)
    
    # Mapear índices aislados a sus contornos correspondientes
        isolated_contours: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = []
        isolated_coords_only: List[np.ndarray[Any, np.dtype[np.int32]]] = []
        
        for idx, cont in cont_array_list:
            if idx in isolated_set:
                isolated_contours.append((idx, cont))
                isolated_coords_only.append(cont)            

        worker_name = context.get("worker_name") or "inker"
        output_paths = context["output_paths"]
        image_id = f"nearest_{image_name}_{worker_name}"
        
        save_shapes(image_name, image_id, bin_img, output_paths, isolated_coords_only, contours2=[])
    
    def _decolorate(self, full_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        Elimina colores (rayones, resaltados, etc.) de la imagen, dejando solo blanco y negro.
        """
        # Máscara para píxeles negros (todos los canales <= threshold_black)
        mask_black = np.all(full_img < self.threshold_black, axis=2)
        
        # Máscara para píxeles blancos (todos los canales >= threshold_white)
        mask_white = np.all(full_img > self.threshold_white, axis=2)
        
        # Máscara de píxeles válidos (negro o blanco)
        mask_valid = mask_black | mask_white
        
        # Reemplaza los píxeles de color (no válidos) por blanco
        full_img[~mask_valid] = self.white

        # Convierte a escala de grises para continuar el flujo normal
        gray = normalice_image(full_img.copy())
        
        if gray is not None:
            return gray
            
        else:
            logger.warning("Normalice IMG devolvío imagen, Imagen en grises de cv2")
            return cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)