# PerfectOCR/core/workflow/vectorial_transformation/vectorizer.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons
from core.utils.cosine_similarity import alignment

logger = logging.getLogger(__name__)

class Vectorizer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('vectorizer', {})
        self.keywords_interval_enabled = self.worker_config.get('keywords_interval_enabled', True)
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time: float = time.time()
            logger.debug("Calculando Features")
            logger.debug(f"Estado de Keywords_interval: {self.keywords_interval_enabled}")

            table_line_ids = self._get_keywords_interval(manager) if self.keywords_interval_enabled else None
            if table_line_ids is not None:
                
                # Si se detecta un intervalo claro, se omite la vectorización
                logger.debug(f"Intervalo tabular detectado, se omite vectorización")
                if self.output:
                    file_name: str = context.get("image_name", "")
                    self._save_output(context, table_line_ids, file_name, manager)

                context["all_features"] = None
                success: bool = manager.save_tabular_lines(table_line_ids)
                if success:
                    logger.debug("Líneas guardadas en el manager desde Vectorizer")
                    return True
                else:
                    logger.warning("No se pudieron guardar las líneas tabulares en el manager")
                    return False

            else:
                # Si no hay intervalo, se prosigue con la vectorización normal
                all_features = self._vectorize_text(manager)
                if all_features:
                    total_time = time.time() - start_time
                    logger.debug(f"Vectorización completada en {total_time:.6f}s. Líneas válidas: {len(all_features)}")
                    context["all_features"] = all_features
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
                key=lambda kv: kv[1].line_geometry.line_centroid[1]
            )
            
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
                row_values = [line_id] + [f"{features.get(k, 0.0):.4f}" if isinstance(features.get(k), float) else str(features.get(k, '')) for k in table_headers]
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
                logger.debug(f"\nTabla unificada de características:\n{table_str}")
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
                lid: (len(vals) if vals is not None else 0)
                for lid, vals in (encoded_lines or {}).items()
            }
            global_max_encoded: int = max(max_count_by_line.values()) if max_count_by_line else 0

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
                            semantic = getattr(polygons_dict[pid], "semantic_type", "") or ""
                            if semantic == "numeric":
                                numeric_count += 1.0
                
                all_numerics = line_features.get("total_numerics_global")
                if all_numerics is None:
                    continue
                    
                max_numeric_count = line_features.get("max_numeric_count_global")
                if max_numeric_count is None:
                    continue

                numeric_mean = line_features.get("numeric_mean_global")
                if numeric_mean is None:
                    continue

                max_digit_count = line_features.get("max_digit_count_global")
                if max_digit_count is None:
                    continue

                max_char_count = line_features.get("max_char_count_global")
                if max_char_count is None:
                    continue

                line_text = getattr(line_data, "text", "") or ""
                
                numeric_count_norm = numeric_count / max_numeric_count if max_numeric_count > 0 else 0.0
                numeric_frec_rel: float = numeric_count - max_numeric_count 
                numeric_ratio_frec: float = numeric_count_norm + numeric_frec_rel
                num_above: float = 1.0 if numeric_count > numeric_mean else 0.0
                digit_char_count = sum(ch.isdigit() for ch in line_text)
                digit_char_frec: float = digit_char_count / max_digit_count if digit_char_count > 0 else 0.0

                # Normaliza num_margin en el intervalo [-1, 1] usando el promedio global como base 0.
                if max_numeric_count > 0:
                    num_margin = (numeric_count - numeric_mean) / (max_numeric_count / 2)
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
                
                total_size = img_dims.get("size")
                total_width = img_dims.get("width")
                total_height = img_dims.get("height") 
                max_width = geoline_features.get("max_count_width")
                max_area = geoline_features.get("max_count_area")
                max_perimeter = geoline_features.get("max_count_perimeter") 
                max_asptrat = geoline_features.get("max_count_aspcrat")
                max_diagonal = geoline_features.get("max_count_diagonal")

                bbox_height: float = bbox[3] - bbox[1]
                bbox_width: float = float(bbox[2] - bbox[0])
                norm_wid = (bbox_width / max_width) if max_width is not None else 0.0
                width_rel = bbox_width / total_width if total_width is not None else 0.0
                cw: float = (total_width / 2.0) if total_width is not None else 0.0
                ch: float = (total_height / 2.0) if total_height is not None else 0.0
                main_centroid: List[float] = cw, ch if ch or cw > 0.0 else 0.0 # type: ignore
                line_area: float = float(bbox_width * bbox_height) if bbox_height or bbox_width > 0.0  else 0.0
                area_norm: float = (line_area / max_area) if max_area is not None else 0.0
                ratio_area: float = (line_area / float(total_size)) if total_size is not None else 0.0
                max_ratio: float = (max_area / total_size) if total_size is not None and max_area is not None else 0.0
                ratio_area_norm = float(ratio_area / max_ratio )
                aspect_ratio = ((bbox_height / bbox_width) * 100.0) if bbox_width or bbox_height > 0 else 0.0
                aspcrat_inv_norm = 1 - abs(aspect_ratio / max_asptrat) if max_asptrat is not None else 0
                perimeter: float = 2 * (bbox_width + bbox_height) if bbox_width or bbox_height > 0 else 0.0
                perimeter_norm: float = (perimeter / max_perimeter) if max_perimeter is not None else 0.0
                diagonal = float(np.sqrt((bbox_width**2.0) + (bbox_height**2.0)))
                diagonal_norm = float(diagonal / max_diagonal) if max_diagonal is not None else 0.0 
                compact = ((perimeter**2) / line_area) / 100.0 if line_area > 0 else 0.0

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
                numeric_values: List[float] = [float(x) for x in line_values]
                if len(numeric_values) < 2:
                    continue
                
                # Calcular estadísticos básicos
                count: float = float(len(numeric_values))
                mean: float = sum(numeric_values) / count
                std_dev = np.std(numeric_values, ddof=1).astype(float)

                # Calcular etadisticos especiales
                mean_rel: float = mean / global_max_encoded if mean > 0.0 else 0.0
                count_rel: float = count / global_max_encoded if count or global_max_encoded > 0.0 else 0.0
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
                    'count_rel': count_rel,
                    'mean_rel': mean_rel,
                    'mean_margin': mean_margin,
                    'skewness': skewness,
                    "numeric_count_norm": numeric_count_norm,
                    "numeric_ratio_frec": numeric_ratio_frec,
                    "num_above": num_above,
                    "num_margin": num_margin,
                    "digit_char_frec": digit_char_frec,
                    "area_norm": area_norm,
                    "norm_wid": norm_wid,
                    "width_rel": width_rel,
                    "ratio_area": ratio_area,
                    "ratio_area_norm": ratio_area_norm,
                    "aspcrat_inv_norm": aspcrat_inv_norm,
                    "perimeter_norm": perimeter_norm,
                    "diagonal_norm": diagonal_norm,
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
            char_count_by_line: Dict[str, float] = {}
            digit_count_by_line: Dict[str, float] = {}
            numeric_counts_by_line: Dict[str, float] = {}
            for line_id, line_data in sorted_lines:
                chcount = 0.0
                dcount = 0.0
                ncount = 0.0
                poly_ids_line = getattr(line_data, "polygon_ids", []) or []
                if manager.workflow and manager.workflow.polygons and poly_ids_line:
                    polygons_dict: Dict[str, Polygons] = manager.workflow.polygons
                    for pid in poly_ids_line:
                        if pid in polygons_dict and getattr(polygons_dict[pid], "semantic_type", "") == "numeric":
                            ncount += 1.0
                numeric_counts_by_line[line_id] = ncount
                
                line_text = getattr(line_data, "text", "") or ""

                chcount = len(line_text)                
                char_count_by_line[line_id] = chcount

                dcount = sum(ch.isdigit() for ch in line_text)
                dcount += 1.0
                digit_count_by_line[line_id] = dcount

            if numeric_counts_by_line:
                total_numerics_global = float(sum(numeric_counts_by_line.values()))
                total_lines_global = float(len(numeric_counts_by_line))
                numeric_mean_global = total_numerics_global / total_lines_global if total_lines_global > 0 else 1.0
                max_numeric_line_id = max(numeric_counts_by_line, key=numeric_counts_by_line.get) # type: ignore
                max_numeric_count_global = float(numeric_counts_by_line[max_numeric_line_id])
                max_digit_count_global = max(digit_count_by_line.values()) if digit_count_by_line else 0.0
                max_char_count_global = max(char_count_by_line.values()) if char_count_by_line else 0.0

                line_features = {
                    "total_numerics_global": total_numerics_global,
                    "numeric_mean_global": numeric_mean_global,
                    "max_numeric_count_global": max_numeric_count_global,
                    "max_digit_count_global": max_digit_count_global,
                    "max_char_count_global": max_char_count_global,
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
                    width_count_by_line[line_id] = bbox_width
                    area_count_by_line[line_id] = area
                    perimeter_count_by_line[line_id] = perimeter
                    asprat_count_by_line[line_id] = aspect_ratio
                    diagonal_count_by_line[line_id] = diagonal
                    
            if width_count_by_line:
                max_count_width = max(width_count_by_line.values()) if width_count_by_line else 0.0
                max_count_area = max(area_count_by_line.values()) if area_count_by_line else 0.0
                max_count_perimeter = max(perimeter_count_by_line.values()) if perimeter_count_by_line else 0.0
                max_count_aspcrat = max(asprat_count_by_line.values()) if asprat_count_by_line else 0.0
                max_count_diagonal = max(diagonal_count_by_line.values()) if diagonal_count_by_line else 0.0


                geoline_features = {
                    "max_count_width": max_count_width,
                    "max_count_area": max_count_area,
                    "max_count_perimeter": max_count_perimeter,
                    "max_count_aspcrat": max_count_aspcrat,
                    "max_count_diagonal": max_count_diagonal,
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

        # Buscar los poly_id de los posibles footers
        footer_poly_ids: List[str] = [
            pid for pid, p in polygons.items()
            if getattr(p, "key_field", None) in ("TotalProductos", "MontoTotalDocumento")
        ]

        # Mapear poly_id a line_id (por ejemplo, si existe un campo line_id en el polígono)
        polyid_to_lineid: Dict[str, str] = {}
        for pid in footer_poly_ids:
            for line_id, line_obj in all_lines.items():
                if pid in line_obj.polygon_ids:
                    polyid_to_lineid[pid] = line_id
                    break

        # Verificar condiciones: debe haber header y al menos un footer
        if not header_line_id or not polyid_to_lineid:
            return None

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

    def _save_output(self, context: Dict[str, Any], expanded_line_ids: List[str], file_name: str, manager: DataFormatter):
        from services.output_service import save_tabjson
        import os
        project_root = self.project_root
        output_file = context.get("output_paths", [])
        for path in output_file:
            output_dir: str = os.path.join(path, "dbscan")
            json_file_name = f"{os.path.splitext(file_name)[0]}.json"
            output_file = save_tabjson(expanded_line_ids, manager, output_dir, json_file_name, project_root)
        if output_file:
            logger.debug(f"OCR Raw results para '{file_name}' guardado en {len(output_file)} ubicaciones.")