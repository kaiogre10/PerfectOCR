# PerfectOCR/core/workflow/vectorial_transformation/vectorizer.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons
from core.utils.text_validator import get_char_num
from core.utils.math_utils import alignment, bbox_alignment, density_cluster

logger = logging.getLogger(__name__)

class Vectorizer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('vectorizer', {})
        self.keywords_interval_enabled = worker_config.get('keywords_interval_enabled', True)
        self.exclude_types =  worker_config['exclude_types']
        self.output = config.get("table_lines", False)
        self.second_output = config.get("encoded_lines", False)
        self.features_output = config.get("features", False)
        self.image_features = config.get("image_features", False)
        self.char_num = get_char_num()

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.perf_counter()
        try:
            logger.debug("Comienza Vectorizer")
            logger.debug(f"Estado de Keywords_interval: {self.keywords_interval_enabled}")

            table_line_ids = self._get_keywords_interval(manager) if self.keywords_interval_enabled else None
            if table_line_ids is not None:
                
                logger.warning(f"Intervalo tabular detectado, se omite vectorización")
                context["all_features"] = None
                logger.debug(f"Intervalo detectado en:{time.perf_counter()-start_time:.7f}")

                if manager.save_tabular_lines(table_line_ids):
                    logger.debug("Líneas guardadas en el manager desde Vectorizer")
                    return True
                else:
                    logger.warning("No se pudieron guardar las líneas tabulares en el manager")
                    return True
            else:
                # Si no hay intervalo, se prosigue con la vectorización normal
                all_features = self._vectorize_text(manager, context)
                if all_features.size != 0:

                    if self.features_output:
                        from services.output_service import save_table_values
                        file_name: str = manager.workflow.metadata.image_name if manager.workflow else ""
                        worker_name = context.get("worker_name") or "vectorizer_features"
                        output_paths = context.get("output_paths", [])
                        image_features = self.image_features
                        save_table_values(file_name, all_features, output_paths, worker_name, image_features)
                        
                    logger.info(f"Vectorización completada en {time.perf_counter() - start_time:.6f}s. Líneas válidas: {len(all_features)}")
                    context["all_features"] = all_features
                    logger.debug(f"Features guardadas en el contexto")
                        
                    return True
                else:
                    logger.error(f"No se pudo realizar vectorización")
                    return False

        except Exception as e:
            logger.error(f"Error en vectorización: {e}", exc_info=True)
            return False
            
    def _vectorize_text(self, manager: DataFormatter, context: Dict[str, Any]) -> Optional[np.ndarray[Any, np.dtype[np.float32]]]:
        try:
            t0 = time.perf_counter()
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            
            t3 = time.perf_counter()
            features_by_line = self._calculate_features(all_lines, manager, context)
            logger.info(f"All_Feautures calculadas en {time.perf_counter() - t3:.7f}s")
            return features_by_line
            
                
            all_features: Dict[str, Dict[str, float]] = {}
            table_headers: List[str] = []
            table_rows: List[List[str]] = []
            id_col_width = 10

            for line_id, _ in all_lines:
                features = features_by_line.get(line_id)
                if not features:
                    continue

                all_features[line_id] = features
                    
                # Inicializar headers solo la primera vez
                if not table_headers:
                    table_headers = list(features.keys())
                
                # Agregar cada fila (convirtiendo todo a string para la tabla)
                row_values = [line_id] + [f"{features.get(k, 0.0):.6f}" if isinstance(features.get(k), float) else str(features.get(k, '')) for k in table_headers]
                table_rows.append(row_values)
                
                # Actualizar ancho de columna ID
                if len(line_id) > id_col_width:
                    id_col_width = len(line_id)
        
            # Construir tabla FUERA del bucle
            if all_features and table_rows:
                col_widths = [int(id_col_width)] + [
                    max(len(str(h)), max(len(f"{row[i+1]:.4f}") if isinstance(row[i+1], float) else len(str(row[i+1])) for row in table_rows))
                    + 2 for i, h in enumerate(table_headers)
                ]
                table: List[str] = []
                table.append("+" + "+".join("-" * w for w in col_widths) + "+")
                header_row: str = "|" + "line_id".center(col_widths[0]) + "|" + "|".join(
                    str(h).center(w) for h, w in zip(table_headers, col_widths[1:])
                ) + "|"
                table.append(header_row)
                # Línea separadora
                table.append("+" + "+".join("-" * w for w in col_widths) + "+")
                # Filas de datos
                for row in table_rows:
                    value_row = "|" + str(row[0]).center(col_widths[0]) + "|" + "|".join(
                        (f"{v:.4f}" if isinstance(v, float) else str(v)).center(w)
                        for v, w in zip(row[1:], col_widths[1:])
                    ) + "|"
                    table.append(value_row)
                # Línea inferior
                table.append("+" + "+".join("-" * w for w in col_widths) + "+")
                table_str = "\n".join(table)
                logger.debug(f"\nTabla unificada características:\n{table_str}")
                logger.warning(f"Vectorización completada en: {time.perf_counter() - t0:.7f}s")
                return all_features
            else:
                logger.warning("No se pudieron calcular features para ninguna línea")
                return None
                    
        except Exception as e:
            logger.error(f"Error vectorizando lineas: {e}", exc_info=True)
            return None
        
    def _calculate_features(self, all_lines: Dict[str, AllLines] , manager: DataFormatter, context: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Calcula features geométricos + alineación tabular por cada línea.
        Retorna un diccionario con features por cada línea.
        """
        try:
            t1 = time.perf_counter()            
            textual_features = self._calculate_textual_line_features(all_lines, manager)
            logger.debug(f"Features textuales calculadas en {time.perf_counter() - t1:.7f}s")
            logger.info(f"Features textuales shape: {textual_features.shape}"
                        "\n"f"{textual_features.round(0)}")

            t2 = time.perf_counter()
            geoline_features: np.ndarray[Any, Any] = self._calculate_geometric_line_features(all_lines, manager)
            logger.debug(f"Features geometricas calculadas en {time.perf_counter() - t2:.7f}s")
            logger.info(f"Features geometricas: {geoline_features}")
            
            t3 = time.perf_counter()
            global_stats = self.calculate_global_stats(geoline_features.copy())
            logger.debug(f"Features globales calculadas en {time.perf_counter() - t3:.7f}s")
            logger.info("Features globales:"
            "\n"f"{global_stats}"
            "\n"f"SHAPE:{global_stats.shape}")
            
            t3 = time.perf_counter()
            all_features = self.calculate_all_features(all_lines, geoline_features, global_stats, manager)
            logger.debug(f"Features completas calculadas en {time.perf_counter() - t3:.7f}s")
            logger.info("Features completas:"
            "\n"f"{all_features}"
            "\n"f"SHAPE:{all_features.shape}")
           
            all_lines_features = np.column_stack([all_features, textual_features])
            logger.debug(f"{all_lines_features.round(0)}")

            if self.second_output:
                from services.output_service import save_raw_json
                file_name: str = manager.workflow.metadata.image_name if manager.workflow else ""
                worker_name = context.get("worker_name") or "vectorizer"
                output_paths = context["output_paths"]
                save_raw_json(output_paths, worker_name, all_lines_features, file_name)

            return all_lines_features

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}", exc_info=True)
            return []

    def _calculate_textual_line_features(self, all_lines: Dict[str, AllLines], manager: DataFormatter) -> np.ndarray[Any, np.dtype[np.int32]]:
        """
        Devuelve un array (n_lines, 3) con las features textuales por línea:
        [count_numeric_tokens, count_digits, longitud_texto]
        """
        try:
            polygons_dict: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            index_to_id_map = {p.poly_index: p.polygon_id for p in polygons_dict.values()}

            features = []
            for line_data in all_lines.values():
                # Cuenta tokens numéricos por línea
                sc_count = 0
                line_index = line_data.line_index
                poly_ids_line = getattr(line_data, "polygon_ids", [])
                for pid_idx in poly_ids_line:
                    pid_str = index_to_id_map.get(pid_idx)
                    if pid_str and pid_str in polygons_dict:
                        sc = polygons_dict[pid_str].semantic_clasification
                        sc_count += self.count_numeric_tokens(sc)
                # Cuenta dígitos en el texto de la línea
                line_text = getattr(line_data, "text", "")
            #    chars = len([c for c in line_text if not c.isspace()])
                dcount = sum(1 for ch in line_text if ch in self.char_num)
       #         dig_mean = dcount / chars
                features.append([sc_count, dcount])

            features = np.array(features)
            
        #    num_lines = np.max(features[:, 0])
            all_numerics = np.sum(features[:, 0])
            max_numerics = np.max(features[:, 0])
            max_digit = np.max(features[:, 1])
            
            num_count_norm = features[:, 0] / max_numerics 
            
            num_mean = (all_numerics/features.shape[0])
            mask = np.zeros((features.shape[0]))
            num_above = np.where(features[:, 0] >= num_mean, mask, 1-mask)
            
            has_digit = np.where(features[:,1] > 0, mask, 1-mask)
            digit_char_frec = features[:,1] / max_digit
            has_numeric = np.where(features[:, 0] > 0, mask, 1 - mask)

            # Normaliza la diferencia entre los tokens numéricos de la línea y el promedio global al rango [-1, 1]; cerca de 1 significa mucho más numérica que la media, cerca de -1 significa mucho menos numérica.
            #if max_numerics > 0:
            #    num_margin = (features[:, 1] - num_mean) / (max_numerics / np.array(2))*1  # Normaliza la diferencia a [-1, 1]; valores extremos significan líneas atípicas respecto a lo numérico.
              #  num_margin = np.max(np.array(-1.0), np.min(np.array(1.0), num_margin))*1
            #else:
              ## num_margin = 0.0
                
            return np.column_stack([
                    has_numeric,

                    num_count_norm,
                    num_above,
                    digit_char_frec,
                    has_digit
                 # num_margin
                    ])
                     
        except Exception as e:
            logger.warning(f"Error en features de lineas: {e}", exc_info=True)
            return np.zeros((len(all_lines), 3), dtype=np.int32)

    def _calculate_geometric_line_features(self, all_lines: Dict[str, AllLines], manager: DataFormatter) -> np.ndarray[Any, Any]:
        try:
            if not manager.workflow.all_lines if manager.workflow else {}:
                return []

            line_index = np.array([lid.line_index for lid in all_lines.values()], np.int32)
            geometry = [lid.line_geometry for lid in all_lines.values()]
            
            bbox = np.array([geo.line_bbox for geo in geometry], np.float32)
            centroid = np.array([geo.line_centroid for geo in geometry], np.float32)
            width = (bbox[:, 2] - bbox[:, 0])
            height = (bbox[:, 3] - bbox[:, 1])
            area = (width * height)
            perimeter = 2 * (width + height)
            aspect_ratio = (height / width) * 100
            diagonal = np.sqrt(width**2.0 + height**2.0)
            angle = np.degrees(np.arctan2(height, width))
            slope = (width / height)

            return np.column_stack([
                line_index,     # [0] line_index
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
            
        except Exception as e:
            logger.warning(f"Error calculando features geometricas: {e}", exc_info=True)
            
    def calculate_global_stats(self, geoline_features: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.column_stack([
            np.max(geoline_features[:, 1]),      # [0] Ancho máximo de los bounding boxes de las líneas 
            np.max(geoline_features[:, 3]),      # [1] Área máxima cubierta por los bounding boxes de las líneas
            np.max(geoline_features[:, 4]),      # [2] Perímetro máximo entre los bounding boxes de las líneas
            np.max(geoline_features[:, 5]),      # [3] Máxima razón de aspecto (alto/ancho * 100) de las líneas
            np.max(geoline_features[:, 6]),      # [4] Longitud diagonal máxima entre los bounding boxes de las líneas
            
            np.median(geoline_features[:, 1]),   # [5] Mediana del ancho de los bounding boxes de las líneas
            np.median(geoline_features[:, 2]),   # [6] Mediana del alto de los bounding boxes de las líneas
            np.median(geoline_features[:, 3]),   # [7] Mediana del área de los bounding boxes de las líneas
            np.median(geoline_features[:, 4]),   # [8] Mediana del perímetro de los bounding boxes de las líneas
            np.median(geoline_features[:, 5]),   # [9] Mediana de la razón de aspecto (alto/ancho * 100) de las líneas
            np.median(geoline_features[:, 6]),   # [10] Mediana de la longitud diagonal de los bounding boxes de las líneas
            np.median(geoline_features[:, 7]),   # [11] Mediana del ángulo de inclinación de las líneas
            np.median(geoline_features[:, 8])    # [12] Mediana del valor de pendiente de las líneas
        ])

    def calculate_all_features(self, all_lines: Dict[str, AllLines], geoline_features: np.ndarray[Any, Any], global_stats: np.ndarray[Any, np.dtype[np.float32]], manager: DataFormatter):
        img_dims: Dict[str, int] = {}
        if manager.workflow and hasattr(manager.workflow, "metadata") and hasattr(manager.workflow.metadata, "img_dims"):
            img_dims = dict(getattr(manager.workflow.metadata, "img_dims", {}))
            
        total_size = img_dims.get("size") or 0.0
        total_width = img_dims.get("width") or 0.0
        total_height = img_dims.get("height") or 0.0
        
        bbox_height_inv = geoline_features[:, 2] / global_stats[:, 6]
        bbox_h_dif = 1 - abs(geoline_features[:, 2] - global_stats[:, 6]) / global_stats[:, 6]
        bbox_width_inv = geoline_features[:, 1] / global_stats[:, 5]
        bbox_w_dif = 1 - abs(geoline_features[:, 1] - global_stats[:, 5]) / global_stats[:, 5]
        norm_wid = geoline_features[:, 1] / global_stats[:, 0]  # ancho normalizado respecto al máximo ancho observado
        width_rel = geoline_features[:, 1] / total_width  # ancho relativo al ancho total de la imagen
        area_norm = geoline_features[:, 3] / global_stats[:, 1]  # área de la línea normalizada al máximo área
        ratio_area = geoline_features[:, 3] / total_size   # área de línea respecto al área total de la imagen
        area_inv = geoline_features[:, 3] / global_stats[:, 7]
        area_dif = 1 - abs(geoline_features[:, 3] - global_stats[:, 7]) / global_stats[:, 7]
        max_ratio = global_stats[:, 1] / total_size   # máxima proporción de área de línea respecto al área total
        ratio_area_norm = ratio_area / max_ratio  # relación entre ratio de área de la línea y el máximo ratio posible
        aspect_ratio = geoline_features[:, 5]  # ya viene como (alto/ancho * 100.0) de la línea
        aspcrat_inv_norm = 1 - abs(geoline_features[:, 5] / global_stats[:, 3]) # cuán diferente es el aspect_ratio respecto al máximo observado (más cerca de 1, más "promedio")
        perimeter_norm = geoline_features[:, 4] / global_stats[:, 2]  # perímetro normalizado al máximo
        perimeter_inv = geoline_features[:, 4] / global_stats[:, 8]
        perimeter_dif = 1 - abs(geoline_features[:, 4] - global_stats[:, 8]) / global_stats[:, 8]  # Normalización respecto a la mediana, igual que diag_inv
        diag_inv = geoline_features[:, 6] / global_stats[:, 10]
        diag_dif = 1 - abs(geoline_features[:, 6] - global_stats[:, 10]) / global_stats[:, 10]
        angle_inv = geoline_features[:, 7] / global_stats[:, 11]
        diag_norm = geoline_features[:, 6] / global_stats[:, 4]
        compact = ((geoline_features[:, 4] ** 2) / geoline_features[:, 3]) / 100.0  # medida de compactación
        slope_inv = geoline_features[:, 8] / global_stats[:, 12]
        slope_dif = 1 - abs(geoline_features[:, 8] - global_stats[:, 12]) / global_stats[:, 12]
        cw: float = (total_width / 2.0)  # centro horizontal de la imagen
        ch: float = (total_height / 2.0)  # centro vertical de la imagen
        main_centroid: List[float] = cw, ch # type: ignore  # coordenadas (x, y) del centro de la imagen
        
        # Extraer todos los bboxes y centroides en arrays (esto ya lo tienes parcialmente)
        all_bboxes = np.array([ld.line_geometry.line_bbox for ld in all_lines.values()], dtype=np.float32)  # (N, 4)
        all_centroids = np.array([ld.line_geometry.line_centroid for ld in all_lines.values()], dtype=np.float32)  # (N, 2)
        n_lines = len(all_bboxes)

        # Coordenadas prev/next mediante slicing con padding NaN
        prev_bboxes = np.vstack([np.full((1, 4), np.nan), all_bboxes[:-1]])      # (N, 4)
        next_bboxes = np.vstack([all_bboxes[1:], np.full((1, 4), np.nan)])       # (N, 4)
        prev_centroids = np.vstack([np.full((1, 2), np.nan), all_centroids[:-1]])  # (N, 2)
        next_centroids = np.vstack([all_centroids[1:], np.full((1, 2), np.nan)])   # (N, 2)

        # Coordenadas xmin/xmax actuales
        current_xmin = all_bboxes[:, 0]  # (N,)
        current_xmax = all_bboxes[:, 2]  # (N,)

        # --- Alineacion bbox (prev_xmin, prev_xmax, next_xmin, next_xmax) ---
        # Formula: 1 - |cos(angulo)| donde vector = (other_coord - current, other_y)

        def _compute_bbox_align(curr_coord, other_bbox, idx):
            """Operacion matricial para bbox_alignment"""
            vec = np.column_stack([other_bbox[:, idx] - curr_coord, other_bbox[:, 1]])
            norms = np.linalg.norm(vec, axis=1)
            cosine = np.where(norms > 0, vec[:, 0] / norms, 0.0)
            result = 1.0 - np.abs(cosine)
            return np.where(np.isnan(other_bbox[:, 0]), 1.0, result)  # NaN -> 1.0

        prev_xmin_align = _compute_bbox_align(current_xmin, prev_bboxes, 0)
        prev_xmax_align = _compute_bbox_align(current_xmax, prev_bboxes, 2)
        next_xmin_align = _compute_bbox_align(current_xmin, next_bboxes, 0)
        next_xmax_align = _compute_bbox_align(current_xmax, next_bboxes, 2)

        # --- Alineacion centroide (align_prev, align_next, center_align) ---
        # Formula: 1 - |cos(angulo)| donde vector = (other_x - ref_x, other_y)

        def _compute_centroid_align(ref_c, other_c):
            """Operacion matricial para alignment"""
            vec = np.column_stack([other_c[:, 0] - ref_c[:, 0], other_c[:, 1]])
            norms = np.linalg.norm(vec, axis=1)
            cosine = np.where(norms > 0, vec[:, 0] / norms, 0.0)
            result = 1.0 - np.abs(cosine)
            return np.where(np.isnan(other_c[:, 0]), 1.0, result)

        align_prev = _compute_centroid_align(all_centroids, prev_centroids)
        align_next = _compute_centroid_align(all_centroids, next_centroids)

        # center_align: todos contra main_centroid (centro de imagen)
        main_centroid_arr = np.tile(main_centroid, (n_lines, 1))  # (N, 2)
        center_align = _compute_centroid_align(all_centroids, main_centroid_arr)
                
        line_ind = geoline_features[:, 0]

        all_features = np.column_stack([
         #   line_ind,                   # [0] Índice de línea
          #  geoline_features[:, 1],    # [1] width
         #   geoline_features[:, 2],    # [2] height
         ##  geoline_features[:, 3],    # [3] area
          #  geoline_features[:, 4],    # [4] perimeter
        #    geoline_features[:, 5],    # [5] aspect_ratio
         #   geoline_features[:, 6],    # [6] diagonal
          #  geoline_features[:, 7],    # [7] angle
        #    geoline_features[:, 8],    # [8] slope
        #    geoline_features[:, 9],    # [9] centroid x
        ##   geoline_features[:, 10],   # [10] centroid y

            bbox_height_inv,     # [11] height normalized/inverse median
            bbox_h_dif,          # [12] diferencia de height vs mediana
            bbox_width_inv,      # [13] width normalized/inverse median
            bbox_w_dif,          # [14] diferencia de width vs mediana
            norm_wid,            # [15] ancho normalizado respecto a máximo
            width_rel,           # [16] ancho relativo al total de imagen
            area_norm,           # [17] area normalizada al máximo
            ratio_area,          # [18] area relativo al total de imagen
            area_inv,            # [19] area normalizado/inversa de la mediana
            area_dif,            # [20] diferencia de area vs mediana
            center_align,        # [21] alineación con el centroide del documento
            ratio_area_norm,     # [22] ratio relativo a máximo ratio
            aspect_ratio,        # [23] proporción de aspecto (alto/ancho * 100)
            aspcrat_inv_norm,    # [24] cuán diferente aspect_ratio vs máximo
            perimeter_norm,      # [25] perímetro normalizado al máximo
            perimeter_inv,       # [26] perímetro inversa/mediana
            perimeter_dif,       # [27] diferencia de perímetro vs mediana
            diag_inv,            # [28] diagonal inversa/mediana
            diag_dif,            # [29] diferencia de diagonal vs mediana
            angle_inv,           # [30] ángulo inversa/mediana
            diag_norm,           # [31] diagonal normalizada al máximo
            compact,             # [32] medida de compactación 
            slope_inv,           # [33] slope inverso/mediana
            slope_dif,           # [34] diferencia de slope vs mediana
            prev_xmin_align, 
            prev_xmax_align, 
            next_xmin_align,
            next_xmax_align,
            align_prev,
            align_next
        ])

        return all_features
                
    def _get_keywords_interval(self, manager: DataFormatter) -> Optional[List[int]]:
        
        all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}

        # Verificar existencia de header_line
        header_line_id = [lid.line_index for lid in all_lines.values() if getattr(lid, "header_line", None) is not None]
        header_line_id = header_line_id[0] if header_line_id else None
        if not header_line_id:
            logger.debug("No se encontró encabezado de tabla")
            return None
            
        footer_line_id = [lid.line_index for lid in all_lines.values() if getattr(lid, "footer_line", None) is not None]
        footer_line_id = footer_line_id[0] if footer_line_id else None

        if not footer_line_id:
            logger.warning("No se encontró pie de tabla")
            return None

        # Obtener el intervalo de líneas tabulares (excluyendo header y footer)
        all_line_ids = list(lid.line_index for lid in all_lines.values())
        header_pos = all_line_ids.index(header_line_id)
        footer_pos = all_line_ids.index(footer_line_id)
        
        if header_pos > footer_pos - 1:
            return None

        logger.debug(f"Encabezado: {all_line_ids[header_pos]}, footer: {all_line_ids[footer_pos]}")

        tabular_line_ids = all_line_ids[header_pos + 1:footer_pos]
        logger.debug(f"Tabular lines desde vectorizer: {tabular_line_ids}")
        return tabular_line_ids

    def count_numeric_tokens(self, semantic_clasification: int | List[int]) -> int:
        """
        Cuenta cuántos tokens en un polígono son numéricos o cuantitativos, respetando exclusiones.
        """
        sc_array = np.array(semantic_clasification, np.int8)
        mask = (np.all(sc_array >= 1)) & (not np.all(sc_array==0))
        return np.count_nonzero(mask)
        
        # semantic_map = manager.get_semmantic_types()
        # exclude_ints = {semantic_map.get(et) for et in self.exclude_types if semantic_map.get(et) is not None}

        # classifications = semantic_clasification if isinstance(semantic_clasification, list) else [semantic_clasification]

        # # Si algún token tiene un tipo excluido, el polígono entero se considera no numérico.
        # if any(sc in exclude_ints for sc in classifications):
        #     return 0
        
        # # De lo contrario, cuenta los tokens que son numéricos (1) o cuantitativos (2).
        # return sum(1 for sc in classifications if sc in [1, 2])
