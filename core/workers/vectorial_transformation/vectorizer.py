# PerfectOCR/core/workflow/vectorial_transformation/vectorizer.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons
from services.output_service import save_table_values

logger = logging.getLogger(__name__)

class Vectorizer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.output = config.get("features", False)
        self.image_features = config.get("image_features", False)

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.perf_counter()
        try:
            logger.debug("Comienza Vectorizer")

            vectorice = context["vectorice"]
            if not vectorice:
                logger.info(f"Vectorización omitida, por tabla detectada anteriormente: {vectorice}")
                context["all_features"] = np.empty((1, 1))
                return True
                # Si no hay intervalo, se prosigue con la vectorización normal
            all_features = self._vectorize_text(manager)
            if all_features is None:
                logger.error(f"No se pudo realizar vectorización")
                return False
                
            context["all_features"] = all_features

            if self.output:
                all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
                line_id = np.array([id.lineal_id for id in all_lines.values()], np.str_)
                features_to_ind = all_features[:, 1:].astype(np.str_)
                features_id = np.column_stack([line_id, features_to_ind])

                file_name: str = manager.workflow.metadata.image_name if manager.workflow else ""
                worker_name = context.get("worker_name") or "vectorizer_features"
                output_paths = context["output_paths"]
                image_features = self.image_features
                save_table_values(file_name, features_id, output_paths, worker_name, image_features)
                
            logger.debug(f"Vectorización completada en {time.perf_counter() - start_time:.6f}s. Líneas válidas: {len(all_features)}")
                
            return True

        except Exception as e:
            logger.error(f"Error en vectorización: {e}", exc_info=True)
        return False
            
    def _vectorize_text(self, manager: DataFormatter) -> Optional[np.ndarray[Any, Any]]:
        try:
            all_lines_dict = manager.workflow.all_lines if manager.workflow else {}
            
            # Ordenar líneas
            sorted_line_keys = sorted(all_lines_dict.keys())
            sorted_lines = [all_lines_dict[k] for k in sorted_line_keys]
            
            # Obtener array de features con line_index en primera columna
            features_array = self._calculate_features(sorted_lines, manager)
            
            if features_array.size == 0:
                return None
                            
            return features_array
                            
        except Exception as e:
            logger.error(f"Error vectorizando lineas: {e}", exc_info=True)
            return None
        
    def _calculate_features(self, sorted_lines: List[AllLines], manager: DataFormatter) -> np.ndarray[Any, Any]:
        """
        Calcula features geométricos + alineación tabular por cada línea.
        """
        try:
            t0 = time.perf_counter()

            # t2 = time.perf_counter()
            geoline_features: np.ndarray[Any, Any] = self._calculate_geometric_line_features(sorted_lines)
            # logger.debug(f"Features geometricas calculadas en {time.perf_counter() - t2:.7f}s")
            # logger.debug(f"Features geometricas: {geoline_features}")
            
            # t3 = time.perf_counter()
            global_stats = self.calculate_global_stats(geoline_features)
            # logger.info(f"Features globales calculadas en {time.perf_counter() - t3:.7f}s")
            # logger.debug("Features globales:"
            # "\n"f"{global_stats}"
            # "\n"f"SHAPE:{global_stats.shape}")
            
            # t4 = time.perf_counter()
            all_features = self.calculate_all_features(sorted_lines, geoline_features, global_stats, manager)
            # logger.debug(f"Features completas calculadas en {time.perf_counter() - t4:.7f}s")
            # logger.debug("Features completas:"
            # "\n"f"{all_features}"
            # "\n"f"SHAPE:{all_features.shape}")
            
            # t1 = time.perf_counter()
            # line_indices = np.array([line.line_index for line in sorted_lines], dtype=np.int32)
            textual_features = self._calculate_textual_line_features(sorted_lines, manager)
            # logger.debug(f"Features textuales calculadas en {time.perf_counter() - t1:.7f}s")
            # logger.debug(f"Features textuales shape: {textual_features.shape}"
            #             "\n"f"{textual_features}")
        
            # 5. Agregar features textuales
            all_lines_features = np.ascontiguousarray(np.column_stack([all_features, textual_features]))
            # logger.debug(f"{all_lines_features}")
            logger.info(f"TODAS LAS FEATURES calculadas en {time.perf_counter() - t0:.7f}s")

            return all_lines_features

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}", exc_info=True)
            return np.zeros(0)

    def _calculate_textual_line_features(self, sorted_lines: List[AllLines], manager: DataFormatter) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Devuelve features textuales ajustadas a la lógica de vectorize.py (-1.0/1.0 y conteos correctos).
        """
        try:
            polygons_dict: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            index_to_id_map = {p.poly_index: p.polygon_id for p in polygons_dict.values()}

            features_list: List[List[int]] = []
            for line_data in sorted_lines:
                sc_count = 0
                kf_total = 0
                # Cuenta tokens numéricos por línea
                poly_ids_line = line_data.polygons_index
                for pid_idx in poly_ids_line:
                    pid_str = index_to_id_map.get(pid_idx)
                    if pid_str and pid_str in polygons_dict:
                        poly = polygons_dict[pid_str]
                        sc = polygons_dict[pid_str].semantic_clasification
                        sc_count += self.count_numeric_tokens(sc)
                        kf = poly.key_field
                        if kf is not None:
                            if isinstance(kf, list):
                                kf_total += len(kf)
                            else:
                                kf_total += 1
                
                dcount = line_data.t_cuant
                features_list.append([sc_count, dcount, kf_total])

            features = np.array(features_list, np.float32)
            
            if features.shape[0] == 0:
                return np.zeros(0, dtype=np.float32)

            # all_numerics = np.sum(features[:, 0], 0)
            max_numerics, max_digit, max_kf = np.max(features, axis=0)
            
            # Evitar división por cero
            num_count_norm = np.divide(features[:, 0], max_numerics, out=np.zeros_like(features[:, 0]), where=max_numerics!=0)
            num_mean = np.mean(features[:, 0])
            
            kf_abs = (1.0 - 2.0 * (features[:, 2] / max_kf)) if max_kf > 0 else np.zeros(features.shape[0], dtype=np.float32)
                        
            # Lógica matching vectorize.py: 1.0 si >= mean, else -1.0
            num_above = np.where(features[:, 0] > num_mean, 1.0, -1.0)
            
            # Lógica matching vectorize.py: 1.0 si > 1.0, else -1.0
            has_digit = np.where(features[:, 1] > 1.0, 1.0, -1.0)
            
            digit_char_frec = np.divide(features[:, 1], max_digit, out=np.zeros_like(features[:, 1]), where=max_digit!=0)
            
            has_numeric = np.where(features[:, 0] > 0, 1.0, -1.0)

            if max_numerics > 0:
               dig_margin = (features[:, 1] - num_mean) / (max_numerics / 2.0)
               dig_margin = np.clip(dig_margin, -1.0, 1.0)
            else:
               dig_margin = np.zeros_like(features[:, 1])
                
            textual_features = np.column_stack([
                    dig_margin,
                    has_numeric,
                    num_count_norm,
                    num_above,
                    digit_char_frec,
                    has_digit,
                    kf_abs
                    ])
                    
            return np.array(textual_features, dtype=np.float32)
        
        except Exception as e:
            logger.warning(f"Error en features de lineas: {e}", exc_info=True)
            return np.zeros((len(sorted_lines), 6), dtype=np.float32)

    def _calculate_geometric_line_features(self, sorted_lines: List[AllLines]) -> np.ndarray[Any, Any]:
        line_index = np.array([lid.line_index for lid in sorted_lines], np.int16)
        geometry = [lid.line_geometry for lid in sorted_lines]
        bbox = np.array([geo.line_bbox for geo in geometry], np.float32)
        centroid = np.array([geo.line_centroid for geo in geometry], np.float32)
        width = (bbox[:, 2] - bbox[:, 0])
        height = (bbox[:, 3] - bbox[:, 1])
        area = (width * height)
        perimeter = 2 * (width + height)
        aspect_ratio = (height / width) * 100
        diagonal = np.sqrt((width**2.0) + (height**2.0))
        angle = np.degrees(np.arctan2(height, width))
        slope = (width / height)

        return np.column_stack([
            line_index.astype(np.int8),     # [0] line_index
            width,          # [1] width
            height,         # [2] height
            area,           # [3] area
            perimeter,      # [4] perimeter
            aspect_ratio,   # [5] aspect_ratio
            diagonal,       # [6] diagonal
            angle,          # [7] angle
            slope,          # [8] slope
            centroid[:, 0],             # [9] centroide x
            centroid[:, 1]  # [10] centroide y
        ])
            
    def calculate_global_stats(self, geoline_features: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
        MAX_COLS = [1, 3, 4, 5, 6]
        MEDIAN_COLS = [1, 2, 3, 4, 5, 6, 7, 8] 

        max_vals = np.max(geoline_features[:, MAX_COLS], axis=0, keepdims=True)
        median_vals = np.median(geoline_features[:, MEDIAN_COLS], axis=0, keepdims=True)

        return np.column_stack([max_vals, median_vals])
        # global_stats = np.column_stack([
        #     np.max(geoline_features[:, 1]),      # [0] Ancho máximo de los bounding boxes de las líneas 
        #     np.max(geoline_features[:, 3]),      # [1] Área máxima cubierta por los bounding boxes de las líneas
        #     np.max(geoline_features[:, 4]),      # [2] Perímetro máximo entre los bounding boxes de las líneas
        #     np.max(geoline_features[:, 5]),      # [3] Máxima razón de aspecto (alto/ancho * 100) de las líneas
        #     np.max(geoline_features[:, 6]),      # [4] Longitud diagonal máxima entre los bounding boxes de las líneas
            
        #     np.median(geoline_features[:, 1]),   # [5] Mediana del ancho de los bounding boxes de las líneas
        #     np.median(geoline_features[:, 2]),   # [6] Mediana del alto de los bounding boxes de las líneas
        #     np.median(geoline_features[:, 3]),   # [7] Mediana del área de los bounding boxes de las líneas
        #     np.median(geoline_features[:, 4]),   # [8] Mediana del perímetro de los bounding boxes de las líneas
        #     np.median(geoline_features[:, 5]),   # [9] Mediana de la razón de aspecto (alto/ancho * 100) de las líneas
        #     np.median(geoline_features[:, 6]),   # [10] Mediana de la longitud diagonal de los bounding boxes de las líneas
        #     np.median(geoline_features[:, 7]),   # [11] Mediana del ángulo de inclinación de las líneas
        #     np.median(geoline_features[:, 8])    # [12] Mediana del valor de pendiente de las líneas
        # ])
        # logger.info(f"SHAPES: {global_stats.shape}")
        # return global_stats

    def calculate_all_features(self, sorted_lines: List[AllLines], geoline_features: np.ndarray[Any, Any], global_stats: np.ndarray[Any, np.dtype[np.float32]], manager: DataFormatter)-> np.ndarray[Any, Any]:
        img_dims: List[int] = manager.workflow.metadata.img_dims if manager.workflow else []
            
        total_width = img_dims[1] or 0.0
        total_height = img_dims[0] or 0.0
        total_size = total_width * total_height
        
        # Funciones helpers para división segura igualando la lógica de "if x != 0 else 0.0"
        def safe_div(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]):
            return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        
        def safe_dif(val: np.ndarray[Any, Any], med: np.ndarray[Any, Any]):
            return np.where(med != 0, 1 - np.abs(val - med) / med, 0.0)

        # Reemplazos con división segura
        bbox_height_inv = safe_div(geoline_features[:, 2], global_stats[:, 6])
        bbox_h_dif = safe_dif(geoline_features[:, 2], global_stats[:, 6])
        
        bbox_width_inv = safe_div(geoline_features[:, 1], global_stats[:, 5])
        bbox_w_dif = safe_dif(geoline_features[:, 1], global_stats[:, 5])
        
        norm_wid = safe_div(geoline_features[:, 1], global_stats[:, 0])
        width_rel = safe_div(geoline_features[:, 1], total_width) #type: ignore
        
        area_norm = safe_div(geoline_features[:, 3], global_stats[:, 1])
        ratio_area = safe_div(geoline_features[:, 3], total_size) #type: ignore
        
        area_inv = safe_div(geoline_features[:, 3], global_stats[:, 7])
        area_dif = safe_dif(geoline_features[:, 3], global_stats[:, 7])
        
        max_ratio = safe_div(global_stats[:, 1], total_size) #type: ignore
        ratio_area_norm = safe_div(ratio_area, max_ratio)
        
        # aspect_ratio = geoline_features[:, 5]
        aspcrat_inv_norm = 1 - safe_div(np.abs(geoline_features[:, 5]), global_stats[:, 3]) # Nota: vectorize usa abs(ar/max)
        
        perimeter_norm = safe_div(geoline_features[:, 4], global_stats[:, 2])
        perimeter_inv = safe_div(geoline_features[:, 4], global_stats[:, 8])
        perimeter_dif = safe_dif(geoline_features[:, 4], global_stats[:, 8])
        
        diag_inv = safe_div(geoline_features[:, 6], global_stats[:, 10])
        diag_dif = safe_dif(geoline_features[:, 6], global_stats[:, 10])
        
        angle_inv = safe_div(geoline_features[:, 7], global_stats[:, 11])
        diag_norm = safe_div(geoline_features[:, 6], global_stats[:, 4])
        
        compact = safe_div((geoline_features[:, 4] ** 2), geoline_features[:, 3]) / 100.0
        
        # slope_inv = safe_div(geoline_features[:, 8], global_stats[:, 12])
        # slope_dif = safe_dif(geoline_features[:, 8], global_stats[:, 12])
        
        cw: float = (total_width / 2.0)  # centro horizontal de la imagen
        ch: float = (total_height / 2.0)  # centro vertical de la imagen
        main_centroid = np.tile([cw, ch], (len(sorted_lines), 1))
        
        # Extraer todos los bboxes y centroides (usando 'sorted_lines')
        all_bboxes = np.array([ld.line_geometry.line_bbox for ld in sorted_lines], dtype=np.float32)  # (N, 4)
        all_centroids = np.array([ld.line_geometry.line_centroid for ld in sorted_lines], dtype=np.float32)

        # Coordenadas prev/next mediante slicing con padding NaN
        prev_bboxes = np.vstack([np.full((1, 4), np.nan), all_bboxes[:-1]])
        next_bboxes = np.vstack([all_bboxes[1:], np.full((1, 4), np.nan)])
        prev_centroids = np.vstack([np.full((1, 2), np.nan), all_centroids[:-1]])
        next_centroids = np.vstack([all_centroids[1:], np.full((1, 2), np.nan)])

        # Coordenadas xmin/xmax actuales
        current_xmin = all_bboxes[:, 0]
        current_xmax = all_bboxes[:, 2]

        def _compute_bbox_align(curr_coord: np.ndarray[Any, np.dtype[np.float32]], other_bbox: np.ndarray[Any, np.dtype[np.float32]], idx: int) -> np.ndarray[Any, np.dtype[np.float32]]:
            """
            Versión vectorizada que copia matemáticamente bbox_alignment antiguo.
            Ignora curr_coord[y] y usa 0.0, calculando el coseno con el vector (diferencia_x, other_bbox[y]).
            """
            if other_bbox.shape[0] == 0 or np.all(np.isnan(other_bbox[:, 0])):
                return np.ones_like(curr_coord)
            
            # Calculamos diferencias al estilo ref_point = [curr_coord, 0.0]
            dx = other_bbox[:, idx] - curr_coord
            dy = other_bbox[:, 1] - 0.0  # El y de la otra caja menos 0.0 (fiel a alineación original)
            
            # Vector: [diferencia en X, diferencia en Y desde 0]
            vec = np.column_stack([dx, dy])
            
            # Norma del vector (magnitud para dividir en el coseno)
            norms = np.linalg.norm(vec, axis=1)
            
            # El ref_vec es siempre [1.0, 0.0], por lo que el dot product de vec y ref_vec 
            # siempre es igual a vec[:, 0] (el componente X). La norma de [1,0] es 1.
            with np.errstate(divide='ignore', invalid='ignore'):
                cosine = np.where(norms > 0, vec[:, 0] / norms, 0.0)
            
            result = 1.0 - np.abs(cosine)
            
            # Donde other_bbox es NaN (no existe línea), devolver 1.0 como indicaba "if not prev_bbox else 1.0"
            return np.where(np.isnan(other_bbox[:, idx]), 1.0, result.astype(np.float32))

        # Pasar current_ymin a las funciones de alineación
        prev_xmin_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmin, prev_bboxes, 0)
        prev_xmax_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmax, prev_bboxes, 2)
        next_xmin_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmin, next_bboxes, 0)
        next_xmax_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmax, next_bboxes, 2)

        def _compute_centroid_align(ref_c: np.ndarray[Any, np.dtype[np.float32]], other_c: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
            """
            Versión vectorizada que copia matemáticamente el alignment antiguo.
            Asume el punto de referencia como [ref_x, 0.0].
            """
            if other_c.shape[0] == 0 or np.all(np.isnan(other_c[:, 0])):
                return np.ones(ref_c.shape[0], np.float32)
            
            # Vector al estilo ref_point = [ref_c[0], 0.0]
            dx = other_c[:, 0] - ref_c[:, 0]
            dy = other_c[:, 1] - 0.0
            
            vec = np.column_stack([dx, dy])
            
            # Norma del vector
            norms = np.linalg.norm(vec, axis=1)
            
            # Al igual que arriba, el dot product contra [1, 0] es igual al componente X.
            with np.errstate(divide='ignore', invalid='ignore'):
                cosine = np.where(norms > 0, vec[:, 0] / norms, 0.0)
            
            result = 1.0 - np.abs(cosine)
            
            # Donde other_c es NaN (no existe línea), devolver 1.0
            return np.where(np.isnan(other_c[:, 0]), 1.0, result.astype(np.float32))

        # Aplicar corrección a centroides
        align_prev = _compute_centroid_align(all_centroids, prev_centroids)
        align_next = _compute_centroid_align(all_centroids, next_centroids)
        center_align = _compute_centroid_align(all_centroids, main_centroid)
    
        line_ind = geoline_features[:, 0]

        all_features = np.column_stack([
            line_ind.astype(np.int32), # [0] Índice de línea
            bbox_height_inv,     # [1] height normalized/inverse median
            bbox_h_dif,          # [2] diferencia de height vs mediana
            bbox_width_inv,      # [3] width normalized/inverse median
            bbox_w_dif,          # [4] diferencia de width vs mediana
            norm_wid,            # [5] ancho normalizado respecto a máximo
            width_rel,           # [6] ancho relativo al total de imagen
            area_norm,           # [7] area normalizada al máximo

            area_inv,            # [9] area normalizada/inversa de la mediana
            area_dif,            # [10] diferencia de area vs mediana
            center_align,        # [11] alineación con el centroide del documento
            ratio_area_norm,     # [12] ratio relativo a máximo ratio

            aspcrat_inv_norm,    # [14] cuán diferente aspect_ratio vs máximo
            perimeter_norm,      # [15] perímetro normalizado al máximo
            perimeter_inv,       # [16] perímetro inversa/mediana
            perimeter_dif,       # [17] diferencia de perímetro vs mediana
            diag_inv,            # [18] diagonal inversa/mediana
            diag_dif,            # [19] diferencia de diagonal vs mediana
            angle_inv,           # [20] ángulo inversa/mediana
            diag_norm,           # [21] diagonal normalizada al máximo
            compact,             # [22] medida de compactación 
            # slope_inv,          # [23] slope inverso/mediana
            # slope_dif,                     # [24] diferencia de slope vs mediana
            prev_xmin_align,     # [25] 
            prev_xmax_align,     # [26] 
            next_xmin_align,     # [27] 
            next_xmax_align,     # [28] 
            align_prev,          # [29] 
            align_next           # [30]
        ])
        return all_features
                
    def count_numeric_tokens(self, semantic_clasification: List[int]) -> int:
        sc = np.asarray(semantic_clasification, dtype=np.int8)
        mask = (sc > 1) & (sc < 3) | (sc == -2) # cuenta 1 y 2
        return int(np.count_nonzero(mask))
