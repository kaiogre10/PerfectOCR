# PerfectOCR/core/workflow/vectorial_transformation/vectorizer.py
import numpy as np
import time
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons, SemanticClassification
from core.utils.cosine_similarity import alignment

logger = logging.getLogger(__name__)

class Vectorizer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('vectorizer', {})
        self.keywords_interval_enabled = self.worker_config.get('keywords_interval_enabled', True)
        self.exclude_types =  self.worker_config.get('exclude_types', [])
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)
        self.output = self.enabled_outputs.get("features", False)
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time: float = time.time()
            logger.debug("Calculando Features")
            logger.warning(f"Estado de Keywords_interval: {self.keywords_interval_enabled}")

            table_line_ids = self._get_keywords_interval(manager) if self.keywords_interval_enabled else None
            if table_line_ids is not None:
                
                logger.warning(f"Intervalo tabular detectado, se omite vectorización")
                context["all_features"] = None

                if manager.save_tabular_lines(table_line_ids):
                    logger.debug("Líneas guardadas en el manager desde Vectorizer")
                    return True
                else:
                    logger.warning("No se pudieron guardar las líneas tabulares en el manager")
                    return True

            else:
                # Si no hay intervalo, se prosigue con la vectorización normal
                all_features = self._vectorize_text(manager)
                if all_features:
                    total_time = time.time() - start_time
                    logger.debug(f"Vectorización completada en {total_time:.6f}s. Líneas válidas: {len(all_features)}")
                    context["all_features"] = all_features
                    logger.debug(f"Features guardadas en el contexto")
                    if self.output:
                        self._save_debug_table(manager, context, all_features)
                        
                    return True
                else:
                    logger.error(f"No se pudo realizar vectorización")
                    return False

        except Exception as e:
            logger.error(f"Error en vectorización: {e}", exc_info=True)
            return False
            
    def _vectorize_text(self, manager: DataFormatter) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """
        try:            
            all_lines: Dict[str, AllLines] = {}
            if manager.workflow and hasattr(manager.workflow, "all_lines"):
                all_lines = getattr(manager.workflow, "all_lines", {})

            sorted_lines: List[Tuple[str, AllLines]] = sorted(
                all_lines.items(),
                key=lambda kv: kv[0]
            )
            
            # sc.descriptive or sc.rfc or sc.code or sc.umd
            line_features: Dict[str, float] = self._calculate_textual_line_featrues(sorted_lines, manager)
            geoline_features: Dict[str, float] = self._calculate_geometric_line_featrues(sorted_lines, manager)            
            features_by_line = self._calculate_features(sorted_lines, manager, line_features, geoline_features)
            if not features_by_line:
                logger.warning("No se pudieron calcular las características de ninguna línea.")
                return None
                
            all_features: Dict[str, Dict[str, float]] = {}
            table_headers: List[str] = []
            table_rows: List[List[str]] = []
            id_col_width = 10

            for line_id, _ in sorted_lines:
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
                logger.info(f"\nTabla unificada de características:\n{table_str}")
                logger.debug(f"Se calcularon features para {len(all_features)} líneas")
                # logger.debug(f"Features: {all_features} líneas")
                return all_features
            else:
                logger.warning("No se pudieron calcular features para ninguna línea")
                return None
                    
        except Exception as e:
            logger.error(f"Error vectorizando lineas: {e}", exc_info=True)
            return None
        
    def _calculate_features(self, sorted_lines: List[Tuple[str, AllLines]] , manager: DataFormatter, line_features: Dict[str, float], geoline_features: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calcula features geométricos + alineación tabular por cada línea.
        Retorna un diccionario con features por cada línea.
        """
        try:
            img_dims: Dict[str, int] = {}
            if manager.workflow and hasattr(manager.workflow, "metadata") and hasattr(manager.workflow.metadata, "img_dims"):
                img_dims = dict(getattr(manager.workflow.metadata, "img_dims", {}))

            if not line_features:
                return {}
            
            all_lines_features: Dict[str, Dict[str, float]] = {}
            encoded_lines = manager.get_encode_lines()
            # Conteo inline: cantidad de elementos por línea y máximo global (sin función adicional)
            max_count_by_line: Dict[str, int] = {
                lid: (len(vals) if vals is not None else 0) # type: ignore
                for lid, vals in (encoded_lines or {}).items()
            }
            global_max_encoded: int = max(max_count_by_line.values()) if max_count_by_line else 0
            num_lines = len(sorted_lines)

            for i, (line_id, line_data) in enumerate(sorted_lines):
                line_values = encoded_lines.get(line_id, [])
                if not line_values:
                    logger.warning(f"Línea {line_id} sin codificación o sin features geométricos; será ignorada.")
                    continue

                if not line_data:
                    logger.warning(f"No se encontraron datos para la línea {line_id}; será ignorada.")
                    continue
                
                numeric_count = 0.0

                poly_ids_line = getattr(line_data, "polygon_ids", []) or []
                if manager.workflow and manager.workflow.polygons and poly_ids_line:
                    polygons_dict: Dict[str, Polygons] = manager.workflow.polygons
                    for pid in poly_ids_line:
                        if pid in polygons_dict:
                            sc = polygons_dict[pid].semantic_clasification
                            if self._is_numeric_polygon(sc):
                                numeric_count += 1.0
                
                all_numerics = line_features.get("total_numerics_global") or 0.0  # Total de claisificación numerica o quantitativa en todas las líneas (global).
                max_numeric_count = line_features.get("max_numeric_count_global") or 0.0  # Cantidad máxima de claisificación numerica o quantitativa encontrados en una sola línea (global).
                median_numeric = line_features.get("median_numeric_global") or 0.0
                max_digit_count = line_features.get("max_digit_count_global") or 0.0  # Máxima cantidad de caracteres dígitos (0-9) en una sola línea (global).
                line_text = getattr(line_data, "text", "") or ""  # El texto bruto de la línea actual.

                has_numeric = 1.0 if numeric_count > 1.0 else - 1.0
                num_count_norm = numeric_count / max_numeric_count if numeric_count > 0 else 0.0  # Proporción de tokens numéricos en la línea respecto al máximo global; mide "cuán numérica" es la línea respecto a la más numérica.
                num_median_norm = numeric_count / median_numeric if numeric_count else 0.0
                num_mean: float = all_numerics / num_lines if num_lines > 0 else 0.0  # Promedio global de tokens numéricos por línea; sirve como referencia.
                num_above: float = 1.0 if numeric_count > num_mean else -1.0  # Indicador (1/0) si la línea supera el promedio global de tokens numéricos.
                digit_char_count = sum(ch.isdigit() for ch in line_text)  # Total de caracteres dígito ("0-9") en el texto de la línea.
                has_digit = 1.0 if digit_char_count > 1.0 else -1.0
                digit_char_frec: float = digit_char_count / max_digit_count if max_digit_count > 0 else 0.0  # Proporción de caracteres dígito en la línea respecto al máximo global visto.

                # Normaliza la diferencia entre los tokens numéricos de la línea y el promedio global al rango [-1, 1]; cerca de 1 significa mucho más numérica que la media, cerca de -1 significa mucho menos numérica.
                if max_numeric_count > 0:
                    num_margin = (numeric_count - num_mean) / (max_numeric_count / 2)  # Normaliza la diferencia a [-1, 1]; valores extremos significan líneas atípicas respecto a lo numérico.
                    num_margin = max(-1.0, min(1.0, num_margin))
                else:
                    num_margin = 0.0
                
                bbox: List[float] = line_data.line_geometry.line_bbox
                centroid: List[float] = line_data.line_geometry.line_centroid

                if len(bbox) < 4.0 or len(centroid) < 2.0:
                    continue

                # Proporción del área respecto del total
                if not img_dims:
                    continue
                
                total_size = img_dims.get("size") or 0.0
                total_width = img_dims.get("width") or 0.0
                total_height = img_dims.get("height") or 0.0
                max_width = geoline_features.get("max_count_width") or 0.0
                max_area = geoline_features.get("max_count_area") or 0.0
                max_perimeter = geoline_features.get("max_count_perimeter") or 0.0
                max_asptrat = geoline_features.get("max_count_aspcrat") or 0.0
                max_diagonal = geoline_features.get("max_count_diagonal") or 0.0
                diagonal_median = geoline_features.get("diagonal_median") or 0.0
                bbox_width_median = geoline_features.get("bbox_width_median") or 0.0
                area_median = geoline_features.get("area_median") or 0.0
                perimeter_median = geoline_features.get("perimeter_median") or 0.0
                angle_median = geoline_features.get("angle_median") or 0.0

                bbox_height: float = bbox[3] - bbox[1]  # alto del bbox de la línea
                bbox_width: float = float(bbox[2] - bbox[0])  # ancho del bbox de la línea
                bbox_width_inv = bbox_width / bbox_width_median if bbox_width_median != 0 else 0.0
                bbox_w_dif = 1 - abs(bbox_width - bbox_width_median)/bbox_width_median if bbox_width_median != 0 else 0.0
                norm_wid = (bbox_width / max_width)  # ancho normalizado respecto al máximo ancho observado
                width_rel = bbox_width / total_width  # ancho relativo al ancho total de la imagen
                cw: float = (total_width / 2.0)  # centro horizontal de la imagen
                ch: float = (total_height / 2.0)  # centro vertical de la imagen
                main_centroid: List[float] = cw, ch # type: ignore  # coordenadas (x, y) del centro de la imagen
                line_area: float = float(bbox_width * bbox_height)  # área del bbox de la línea
                area_norm: float = (line_area / max_area)  # área de la línea normalizada al máximo área
                ratio_area: float = (line_area / float(total_size))  # área de línea respecto al área total de la imagen
                area_inv = line_area / area_median if area_median !=0 else 0.0
                area_dif = 1 - abs(line_area - area_median)/area_median if area_median !=0 else 0.0
                max_ratio: float = (max_area / total_size)  # máxima proporción de área de línea respecto al área total
                ratio_area_norm = float(ratio_area / max_ratio )  # relación entre ratio de área de la línea y el máximo ratio posible; mide "qué tan grande es esta línea respecto al máximo esperado"
                aspect_ratio = ((bbox_height / bbox_width) * 100.0)  # proporción de aspecto (alto/ancho) en porcentaje de la línea
                aspcrat_inv_norm = 1 - abs(aspect_ratio / max_asptrat)  # cuán diferente es el aspect_ratio respecto al máximo observado (más cerca de 1, más "promedio")
                perimeter: float = 2 * (bbox_width + bbox_height)  # perímetro del bbox
                perimeter_norm: float = (perimeter / max_perimeter)  # perímetro normalizado al máximo
                perimeter_inv = perimeter / perimeter_median if perimeter_median != 0 else 0.0 
                perimeter_dif = 1 - abs(perimeter - perimeter_median)/perimeter_median if perimeter_median != 0 else 0.0  # Normalización respecto a la mediana, igual que diag_inv
                diagonal = float(np.hypot(bbox_width, bbox_height))  # diagonal del bbox (distancia máxima)
                diag_inv = diagonal / diagonal_median if diagonal_median != 0 else 0.0
                diag_dif = 1 - abs(diagonal - diagonal_median)/diagonal_median if diagonal_median != 0 else 0.0
                angle = math.degrees(math.atan2(bbox_height, bbox_width))
                angle_inv = angle/ angle_median if angle_median != 0 else 0.0
                diag_norm = float(diagonal / max_diagonal)  # diagonal normalizada al máximo
                compact = ((perimeter**2) / line_area) / 100.0  # medida de compactación (perímetro al cuadrado sobre área), valores más bajos = formas más cuadradas/compactas

                def _calculate_line_coords(sorted_lines: List[Tuple[str, AllLines]], current_index: int) -> Tuple[List[float], List[float], List[float], List[float]]:
                    """Calcula coordenadas de líneas anterior y siguiente, retorna listas vacías si no existen"""
                    lines_num = len(sorted_lines)
                    
                    # Línea anterior
                    if current_index > 0:
                        prev_bbox = sorted_lines[current_index-1][1].line_geometry.line_bbox
                        prev_centroid = sorted_lines[current_index-1][1].line_geometry.line_centroid
                    else:
                        prev_bbox = []
                        prev_centroid = []
                    
                    # Línea siguiente
                    if current_index < lines_num - 1:
                        next_bbox = sorted_lines[current_index+1][1].line_geometry.line_bbox
                        next_centroid = sorted_lines[current_index+1][1].line_geometry.line_centroid
                    else:
                        next_bbox = []
                        next_centroid = []
                    
                    return prev_bbox, next_bbox, prev_centroid, next_centroid            

                def _bbox_alignment(current_coord: float, other_bbox: List[float], coord_idx: int) -> Optional[float]:
                    """
                    Mide alineación usando similitud coseno.
                    Punto de referencia: [current_coord, 0] en el eje X
                    Vector hacia otra línea: [other_coord - current_coord, other_y - 0]
                    """
                    try:
                        if other_bbox:
                            # Punto de referencia en el eje X
                            ref_point = np.array([current_coord, 0.0])

                            # Coordenada de la otra línea
                            other_coord = other_bbox[coord_idx]  # Acceso correcto por índice
                            other_y = other_bbox[1]  # Coordenada Y de la otra línea

                            # Vector desde el punto de referencia hacia la otra línea
                            vec_to_other = np.array([other_coord - current_coord, other_y - ref_point[1]])
                            
                            # Vector de referencia (eje X positivo)
                            ref_vec = np.array([1, 0])
                            
                            # Similitud coseno
                            if np.linalg.norm(vec_to_other) == 0:
                                return 1.0
                            
                            cosine_sim = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec))
                            return 1.0 - abs(float(cosine_sim))   

                        else:
                            return 1.0

                    except Exception as e:
                        logger.error(f"Error calculando similitud coseno: {e}", exc_info=True)
                        return 1.0

                # Alineación ortogonal para xmin y xmax con prev y next
                current_xmin = bbox[0]
                current_xmax = bbox[2]

                prev_bbox, next_bbox, prev_centroid, next_centroid = _calculate_line_coords(sorted_lines, i)
                
                prev_xmin_align: Optional[float] = _bbox_alignment(current_xmin, prev_bbox, 0) if prev_bbox else 1.0
                prev_xmax_align: Optional[float] = _bbox_alignment(current_xmax, prev_bbox, 2) if prev_bbox else 1.0
                next_xmin_align: Optional[float] = _bbox_alignment(current_xmin, next_bbox, 0) if next_bbox else 1.0
                next_xmax_align: Optional[float] = _bbox_alignment(current_xmax, next_bbox, 2) if next_bbox else 1.0

                align_prev: Optional[float] = alignment(centroid, prev_centroid)
                align_next: Optional[float] = alignment(centroid, next_centroid)
                
                center_aling: float = alignment(centroid, main_centroid)
                
                # max_size_num_vals: float = 114.0
                numeric_values: List[float] = [float(x) for x in line_values] # Líneas codificadas
                
                # Calcular estadísticos básicos
                count: float = float(len(numeric_values))
                mean: float = sum(numeric_values) / count
                std_dev = np.std(numeric_values, ddof=1).astype(float)

                # Calcular etadisticos especiales
                mean_rel: float = mean / global_max_encoded if mean > 0.0 else 0.0
                if mean > 0.0:
                    mean_ref: float = float(global_max_encoded / 2.0)
                    mean_margin: float = (mean - mean_ref) / mean_ref
                    mean_margin = max(-1.0, min(1.0, mean_margin))
                else:
                    mean_margin = 0.0
                    
                # Calcular skewness
                skewness: float = 0.0
                if std_dev > 0:
                    moment3: float = sum(((x - mean) / std_dev) ** 3 for x in numeric_values)
                    skewness = moment3 / count
                        
                # Anida el diccionario de características para que coincida con el tipo de retorno esperado.
                line_all_features: Dict[str, float] = {
                    'mean_rel': mean_rel,
                    'mean_margin': mean_margin,
                    'skewness': skewness,
                    "has_numeric": has_numeric,
                    "num_median_norm": num_median_norm,
                    "numeric_count_norm": num_count_norm,
                    "num_above": num_above,
                    "num_margin": num_margin,
                    "digit_char_frec": digit_char_frec,
                    "has_digit": has_digit,
                    "area_norm": area_norm,
                    "norm_wid": norm_wid,
                    "width_rel": width_rel,
                    "bbox_width_inv": bbox_width_inv,
                    "bbox_w_dif": bbox_w_dif,
                    "ratio_area_norm": ratio_area_norm,
                    "area_inv": area_inv,
                    "area_dif": area_dif,
                    "aspcrat_inv_norm": aspcrat_inv_norm,
                    "perimeter_norm": perimeter_norm,
                    "perimeter_inv": perimeter_inv,
                    "perimeter_dif": perimeter_dif,
                    "diag_inv": diag_inv,
                    "diag_dif": diag_dif,
                    "diag_norm": diag_norm,
                    "angle_inv": angle_inv,
                    "compact": compact,
                    "prev_xmin_align": prev_xmin_align if prev_xmin_align is not None else 1.0,
                    "prev_xmax_align": prev_xmax_align if prev_xmax_align is not None else 1.0,
                    "next_xmin_align": next_xmin_align if next_xmin_align is not None else 1.0,
                    "next_xmax_align": next_xmax_align if next_xmax_align is not None else 1.0,
                    "align_prev": align_prev,
                    "align_next": align_next,
                    "center_aling": center_aling,
                }
                all_lines_features[line_id] = line_all_features

            return all_lines_features

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}", exc_info=True)
            return {}

    def _calculate_textual_line_featrues(self, sorted_lines: List[Tuple[str, AllLines]],  manager: DataFormatter) -> Dict[str, float]:
        try:
            if not manager.workflow.all_lines if manager.workflow else {}:
                return {}

            line_features: Dict[str, float] = {}
            # Cálculo global de numerics (promedio y máximo)
            digit_count_by_line: Dict[str, float] = {}
            numeric_counts_by_line: Dict[str, float] = {}
            for line_id, line_data in sorted_lines:
                dcount = 0.0
                ncount = 0.0
                poly_ids_line = getattr(line_data, "polygon_ids", []) or []
                if manager.workflow and manager.workflow.polygons and poly_ids_line:
                    polygons_dict: Dict[str, Polygons] = manager.workflow.polygons
                    for pid in poly_ids_line:
                        if pid in polygons_dict:
                            sc = polygons_dict[pid].semantic_clasification
                            if self._is_numeric_polygon(sc):
                                ncount += 1.0
                numeric_counts_by_line[line_id] = ncount
                
                line_text = getattr(line_data, "text", "") or ""
                dcount = sum(ch.isdigit() for ch in line_text)
                dcount += 1.0
                digit_count_by_line[line_id] = dcount

            if numeric_counts_by_line:
                # Para obtener la mediana correctamente, convierte los valores a una lista antes de pasarlos a np.median
                numeric_counts_list = list(numeric_counts_by_line.values())
                total_numerics_global: float = float(sum(numeric_counts_list)) if numeric_counts_list else 0.0
                max_numeric_count_global = max(numeric_counts_list) if numeric_counts_list else 0.0
                median_numeric_global = float(np.median(numeric_counts_list)) if numeric_counts_list else 0.0
                digit_counts_list = list(digit_count_by_line.values())
                max_digit_count_global = max(digit_counts_list) if digit_counts_list else 0.0
                
                line_features = {
                    "total_numerics_global": total_numerics_global,
                    "max_numeric_count_global": max_numeric_count_global,
                    "max_digit_count_global": max_digit_count_global,
                    "median_numeric_global": median_numeric_global
                }

            return line_features

        except Exception as e:
            logger.debug(f"Error en feaures de lineas: {e}", exc_info=True)
            return {}

    def _calculate_geometric_line_featrues(self, sorted_lines: List[Tuple[str, AllLines]],  manager: DataFormatter) -> Dict[str, float]:
        try:
            if not manager.workflow.all_lines if manager.workflow else {}:
                return {}

            geoline_features: Dict[str, float] = {}

            width_count_by_line: Dict[str, float] = {}
            area_count_by_line: Dict[str, float] = {}
            perimeter_count_by_line: Dict[str, float] = {}
            asprat_count_by_line: Dict[str, float] = {}
            diagonal_count_by_line: Dict[str, float] = {}
            angle_count_by_line: Dict[str, float] = {}
            
            for line_id, line_data in sorted_lines:
                line_geometry = getattr(line_data, "line_geometry", None)
                if line_geometry and hasattr(line_geometry, "line_bbox") and len(line_geometry.line_bbox) == 4:
                    bbox = line_geometry.line_bbox
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height: float = bbox[3] - bbox[1]
                    area = bbox_width * bbox_height
                    perimeter: float = 2 * (bbox_width + bbox_height)
                    aspect_ratio = ((bbox_height / bbox_width) * 100)
                    diagonal = float(np.sqrt((bbox_width**2.0) + (bbox_height**2.0)))
                    angle = math.degrees(math.atan2(bbox_height, bbox_width))
                    width_count_by_line[line_id] = bbox_width
                    area_count_by_line[line_id] = area
                    perimeter_count_by_line[line_id] = perimeter
                    asprat_count_by_line[line_id] = aspect_ratio
                    diagonal_count_by_line[line_id] = diagonal
                    angle_count_by_line[line_id] = angle
                    
            if width_count_by_line:
                max_count_width = max(width_count_by_line.values()) if width_count_by_line else 0.0      # Ancho máximo de los bounding boxes de las líneas
                max_count_area = max(area_count_by_line.values()) if area_count_by_line else 0.0         # Área máxima cubierta por los bounding boxes de las líneas
                max_count_perimeter = max(perimeter_count_by_line.values()) if perimeter_count_by_line else 0.0 # Perímetro máximo entre los bounding boxes de las líneas
                max_count_aspcrat = max(asprat_count_by_line.values()) if asprat_count_by_line else 0.0  # Máxima razón de aspecto (alto/ancho * 100) de las líneas
                max_count_diagonal = max(diagonal_count_by_line.values()) if diagonal_count_by_line else 0.0    # Longitud diagonal máxima entre los bounding boxes de las líneas
                diagonal_values = list(diagonal_count_by_line.values())
                diagonal_median = float(np.median(diagonal_values)) if diagonal_values else 0.0
                bbox_width_values= list(width_count_by_line.values())
                bbox_width_median = float(np.median(bbox_width_values)) if bbox_width_values else 0.0
                area_values= list(area_count_by_line.values())
                area_median = float(np.median(area_values)) if area_values else 0.0
                perimeter_values = list(perimeter_count_by_line.values())
                perimeter_median = float(np.median(perimeter_values)) if perimeter_values else 0.0
                angle_values = list(angle_count_by_line.values())
                angle_median = float(np.median(angle_values)) if angle_values else 0.0

                geoline_features = {
                    "max_count_width": max_count_width,
                    "max_count_area": max_count_area,
                    "max_count_perimeter": max_count_perimeter,
                    "max_count_aspcrat": max_count_aspcrat,
                    "max_count_diagonal": max_count_diagonal,
                    "diagonal_median": diagonal_median,
                    'bbox_width_median': bbox_width_median,
                    'area_median': area_median,
                    'perimeter_median': perimeter_median,
                    "angle_median": angle_median
                }

            return geoline_features
        except Exception as e:
                logger.debug(f"Error en feaures de lineas: {e}", exc_info=True)
                return {}

    def _get_keywords_interval(self, manager: DataFormatter) -> Optional[List[str]]:
        
        polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
        all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}

        # Verificar existencia de header_line
        header_line_ids = [lid for lid, l in all_lines.items() if getattr(l, "header_line", None) is not None]
        header_line_id = header_line_ids[0] if header_line_ids else None
        if not header_line_id:
            return None

        # Buscar los poly_id de los posibles footers
        footer_poly_ids: List[str] = [
            pid for pid, p in polygons.items()
            if getattr(p, "key_field", None) in ("TotalProductos", "MontoTotalDocumento")
        ]
        if not footer_poly_ids:
            return None

        # Mapear poly_id a line_id (por ejemplo, si existe un campo line_id en el polígono)
        polyid_to_lineid: Dict[str, str] = {}
        for pid in footer_poly_ids:
            for line_id, line_obj in all_lines.items():
                if pid in line_obj.polygon_ids:
                    polyid_to_lineid[pid] = line_id
                    break

        # Elegir el footer más cercano al header_line_id
        header_idx = list(all_lines.keys()).index(header_line_id)
        min_distance = None
        winner_line_id = None
        for pid, line_id in polyid_to_lineid.items():
            if line_id in all_lines:
                idx = list(all_lines.keys()).index(line_id)
                distance = abs(idx - header_idx)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    winner_line_id = line_id

        if winner_line_id is None:
            return None

        # Obtener el intervalo de líneas tabulares (excluyendo header y footer)
        all_line_ids = list(all_lines.keys())
        header_pos = all_line_ids.index(header_line_id)
        footer_pos = all_line_ids.index(winner_line_id)

        logger.debug(f"Encabezado: {all_line_ids[header_pos]}, footer: {all_line_ids[footer_pos]}")

        # Asegurar que el header esté antes que el footer
        if header_pos >= footer_pos - 1:
            return None

        tabular_line_ids = all_line_ids[header_pos + 1:footer_pos]
        logger.debug(f"Tabular lines desde vectorizeier: {tabular_line_ids}")
        return tabular_line_ids

    def _is_numeric_polygon(self, semantic_clasification: SemanticClassification) -> bool:
        """
        Determina si un polígono debe contarse como numérico basado en su clasificación semántica.
        Usa la configuración del YAML para determinar qué tipos excluir.
        """
        # Verificar si alguno de los tipos excluidos está activo
        for exclude_type in self.exclude_types:
            if getattr(semantic_clasification, exclude_type, False):
                return False
        return True
        
    def _save_debug_table(self, manager: DataFormatter, context: Dict[str, Any], all_features: Dict[str, Dict[str, float]]):
        from services.output_service import save_table
        import os
        import pandas as pd #type: ignore
        import matplotlib.pyplot as plt
        # Crear DataFrame desde all_features
        
        df = pd.DataFrame.from_dict(all_features, orient='index')
        df.index.name = 'line_id'
        
        # Resetear índice para que line_id sea una columna
        df = df.reset_index()
        
        # Obtener nombre del archivo
        file_name: str = manager.workflow.metadata.image_name
        output_paths = context.get("output_paths", [])
        
        for path in output_paths:
            output_dir: str = os.path.join(path, "vectorizer")
            table_file_name = f"{os.path.splitext(file_name)[0]}.csv"
            save_table(df, output_dir, table_file_name, list(df.columns))
        
        # Preparar datos (excluir line_id)
        features_data = df.drop('line_id', axis=1)
        feature_names = features_data.columns.tolist()
        
        # Crear la figura
        plt.figure(figsize=(12, 8))
        
        # Plotear cada línea del documento con valores originales
        for idx, row in features_data.iterrows():
            line_id = df.iloc[idx]['line_id']
            plt.plot(feature_names, row.values, label=f'Línea {line_id}', alpha=0.7, linewidth=1)
        
        # Configurar la gráfica
        plt.xlabel('Features')
        plt.ylabel('Valores de Features')
        plt.title(f'Comportamiento de Features por Línea - {os.path.splitext(file_name)[0]}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Calcular los límites del eje Y y poner los ticks de 1 en 1
        if not features_data.empty:
            ymin = features_data.min().min()
            ymax = features_data.max().max()
            ymin_tick = int(np.floor(ymin))
            ymax_tick = int(np.ceil(ymax))
            plt.yticks(np.arange(ymin_tick, ymax_tick + 1, 1))
        
        # Limitar leyenda si hay muchas líneas
        if len(df) > 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Guardar la gráfica
        plot_filename = f"{os.path.splitext(file_name)[0]}_features_plot.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfica de features guardada en: {plot_path}")
        
        