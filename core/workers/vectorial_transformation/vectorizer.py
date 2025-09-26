# PerfectOCR/core/workflow/vectorial_transformation/vectorizer.py
import math
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons

logger = logging.getLogger(__name__)

class Vectorizer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('vectorizer', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)        
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:   
            start_time: float = time.time()
            logger.info("Calculando Features")
            
            img_dims: Dict[str, int] = {}
            if manager.workflow and hasattr(manager.workflow, "metadata") and hasattr(manager.workflow.metadata, "img_dims"):
                img_dims = dict(getattr(manager.workflow.metadata, "img_dims", {}))

            all_lines: Dict[str, AllLines] = {}
            if manager.workflow and hasattr(manager.workflow, "all_lines"):
                all_lines = getattr(manager.workflow, "all_lines", {})

            encoded_lines = manager.get_encode_lines()

            all_features = self._vectorize_text(encoded_lines, all_lines, img_dims, manager)
            if all_features:
                total_time = time.time() - start_time
                logger.info(f"Vectorización completada en {total_time:.6f}s. Líneas válidas: {len(all_features)}")
                context["all_features"] = all_features
                
                return True
            else:
                logger.error(f"No se pudo realizar vectorización")
                return False
        except Exception as e:
            logger.error(f"Error en vectorización: {e}", exc_info=True)
            return False
            
    def _vectorize_text(self, encoded_lines: Dict[str, List[int]], all_lines: Dict[str, AllLines], img_dims: Dict[str, int], manager: DataFormatter) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """
        try:            
            feature_keys: List[str] = [
                'count_rel',
                'mean_rel',
                'mean_margin',
                'std_dev',
                'p25', 
                'p50',
                # 'p75',
                # 'iqr',
                'skewness',
                'numeric_count_norm',
                "numeric_frec_rel",
                "numeric_ratio_frec",
                "num_above",
                "num_margin",
                "digit_char_frec",
                'line_area',
                'width_rel',
                'ratio_area',
                'aspect_ratio',
                'height',
                'perimeter',
                'diagonal',
                'compact',
                'prev_xmin_align',
                'prev_xmax_align',
                'next_xmin_align',
                'next_xmax_align',
                'align_prev',
                'align_next',
                ]
            
            line_features: Dict[str, float] = self._calculate_line_featrues(all_lines, manager)
            if not line_features:
                logger.warning("No se pudieron calcular las características globales de líneas")
                return None
                
            all_features: Dict[str, Dict[str, float]] = {}
            tabla_headers: List[str] = []
            tabla_rows: List[List[str]] = []
            id_col_width = 10
            
            for line_id, line_data in all_lines.items():
                line_values = encoded_lines.get(line_id, [])
                if not line_values:
                    logger.warning(f"Línea {line_id} sin codificación o sin features geométricos; será ignorada.")
                    continue

                if not line_data:
                    logger.warning(f"No se encontraron datos para la línea {line_id}; será ignorada.")
                    continue
                
                features = self._calculate_features(line_id, line_data, all_lines, img_dims, manager, line_features, line_values)
                if features and 'aggregate_stats' in features:
                    agg = features.get('aggregate_stats', {})
                    all_features[line_id] = features['aggregate_stats']
                    
                    # Inicializar headers solo la primera vez
                    if not tabla_headers:
                        tabla_headers = list(agg.keys())
                    
                    # Agregar cada fila
                    tabla_rows.append([line_id] + [agg[k] for k in tabla_headers]) 
                    
                    # Actualizar ancho de columna ID
                    if len(line_id) > id_col_width:
                        id_col_width = len(line_id)
        
            # Construir tabla FUERA del bucle
            if all_features and tabla_rows:
                col_widths = [int(id_col_width)] + [
                    max(len(str(h)), max(len(f"{row[i+1]:.4f}") if isinstance(row[i+1], float) else len(str(row[i+1])) for row in tabla_rows))
                    + 2 for i, h in enumerate(tabla_headers)
                ]
                tabla: List[str] = []
                tabla.append("+" + "+".join("-" * w for w in col_widths) + "+")
                header_row: str = "|" + "line_id".center(col_widths[0]) + "|" + "|".join(
                    str(h).center(w) for h, w in zip(tabla_headers, col_widths[1:])
                ) + "|"
                tabla.append(header_row)
                # Línea separadora
                tabla.append("+" + "+".join("-" * w for w in col_widths) + "+")
                # Filas de datos
                for row in tabla_rows:
                    value_row = "|" + str(row[0]).center(col_widths[0]) + "|" + "|".join(
                        (f"{v:.4f}" if isinstance(v, float) else str(v)).center(w)
                        for v, w in zip(row[1:], col_widths[1:])
                    ) + "|"
                    tabla.append(value_row)
                # Línea inferior
                tabla.append("+" + "+".join("-" * w for w in col_widths) + "+")
                tabla_str = "\n".join(tabla)
                logger.info(f"\nTabla unificada de características:\n{tabla_str}")
                logger.info(f"Se calcularon features para {len(all_features)} líneas")
                # logger.info(f"Features: {all_features} líneas")
                return all_features
            else:
                logger.warning("No se pudieron calcular features para ninguna línea")
                return None
                    
        except Exception as e:
            logger.error(f"Error vectorizando lineas: {e}", exc_info=True)
            return None

    def _calculate_line_featrues(self, all_lines: Dict[str, AllLines],  manager: DataFormatter) -> Dict[str, float]:
        try:
            if not manager.workflow.all_lines if manager.workflow else {}:
                return {}

            # polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}

            sorted_lines: List[Tuple[str, AllLines]] = sorted(
                all_lines.items(),
                key=lambda kv: kv[1].line_geometry.line_centroid[1]
            )

            line_features: Dict[str, float] = {}
            # Cálculo global de numerics (promedio y máximo)
            char_count_by_line: Dict[str, float] = {}
            digit_count_by_line: Dict[str, float] = {}
            numeric_counts_by_line: Dict[str, float] = {}
            for i, (line_id, line_data) in enumerate(sorted_lines):
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
            logger.info(f"Error en feaures de lineas: {e}", exc_info=True)
                    
    def _calculate_features(self, line_id: str, line_data: AllLines, all_lines: Dict[str, AllLines], img_dims: Dict[str, int], manager: DataFormatter, line_features: Dict[str, float], line_values: List[int]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calcula features geométricos + alineación tabular por cada línea.
        Retorna un diccionario con features por cada línea.
        """
        try:
            if not all_lines or not line_data:
                return {}

            if not line_features:
                return {}
                
            sorted_lines: List[Tuple[str, AllLines]] = sorted(
                all_lines.items(),
                key=lambda kv: kv[1].line_geometry.line_centroid[1]
            )
            
            # Encontrar el índice de la línea actual en la lista ordenada
            current_index = -1
            for i, (lid, _) in enumerate(sorted_lines):
                if lid == line_id:
                    current_index = i
                    break
            
            if current_index == -1:
                logger.error(f"No se pudo encontrar la línea {line_id} en las líneas ordenadas.")
                return {}

            numeric_count = 0.0

            poly_ids_line = getattr(line_data, "polygon_ids", []) or []
            if manager.workflow and manager.workflow.polygons and poly_ids_line:
                polygons_dict = manager.workflow.polygons
                for pid in poly_ids_line:
                    if pid in polygons_dict:
                        semantic = getattr(polygons_dict[pid], "semantic_type", "") or ""
                        if semantic == "numeric":
                            numeric_count += 1.0
            
            all_numerics = line_features.get("total_numerics_global")
            if all_numerics is None:
                return {}
                
            max_numeric_count = line_features.get("max_numeric_count_global")
            if max_numeric_count is None:
                return {}

            numeric_mean = line_features.get("numeric_mean_global")
            if numeric_mean is None:
                return {}

            max_digit_count = line_features.get("max_digit_count_global")
            if max_digit_count is None:
                return {}

            max_char_count = line_features.get("max_char_count_global")
            if max_char_count is None:
                return {}

            line_text = getattr(line_data, "text", "") or ""
            
            numeric_count_norm = numeric_count / max_numeric_count if max_numeric_count > 0 else 0.0
            numeric_frec_rel: float = numeric_count - max_numeric_count 
            numeric_ratio_frec: float = numeric_count_norm + numeric_frec_rel
            num_above: float = 1.0 if numeric_count > numeric_mean else 0.0
            digit_char_count: float = float(sum(ch.isdigit() for ch in line_text))
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
                return {}

            # Proporción del área respecto del total
            if not img_dims:
                return {}
            
            total_size = img_dims.get("size")
            if not total_size:
                return {}

            total_width = img_dims.get("width")
            if not total_width:
                return {} 

            bbox_height: float = bbox[3] - bbox[1]
            bbox_width: float = float(bbox[2] - bbox[0])
            width_rel = bbox_width / total_width if bbox_width > 0.0 else 0.0
            line_area: float = float(bbox_width * bbox_height)
            ratio_area: float = (line_area / float(total_size)) if line_area > 0.0 else 0.0
            aspect_ratio = (bbox_height / bbox_width)  if bbox_width or bbox_height > 0 else 0.0
            perimeter: float = 2 * (bbox_width + bbox_height)
            diagonal = float(np.sqrt((bbox_width**2.0) + (bbox_height**2.0)))
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

            def _alignment(ref_c: List[float], other_c: List[float]) -> float:
                """
                Mide alineación usando similitud coseno, tomando como referencia (X,0) del centroide.
                Vector desde (ref_c[0], 0) hacia (other_c[0], other_c[1]).
                """
                if not other_c:
                    return 1.0
                ref_point = np.array([ref_c[0], 0.0])
                vec_to_other = np.array([other_c[0] - ref_point[0], other_c[1] - ref_point[1]]).astype(np.float32)
                ref_vec = np.array([1, 0]).astype(np.float32)  # eje X positivo
                if np.linalg.norm(vec_to_other) == 0.0:
                    return 1.0
                cosine = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec))
                return 1.0 - abs(float(cosine))

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

            # Alineación ortogonal para xmin y xmax con prev y next
            current_xmin = bbox[0]
            current_xmax = bbox[2]

            prev_bbox, next_bbox, prev_centroid, next_centroid = _calculate_line_coords(sorted_lines, current_index)
            
            prev_xmin_align: Optional[float] = _bbox_alignment(current_xmin, prev_bbox, 0) if prev_bbox else 1.0
            prev_xmax_align: Optional[float] = _bbox_alignment(current_xmax, prev_bbox, 2) if prev_bbox else 1.0
            next_xmin_align: Optional[float] = _bbox_alignment(current_xmin, next_bbox, 0) if next_bbox else 1.0
            next_xmax_align: Optional[float] = _bbox_alignment(current_xmax, next_bbox, 2) if next_bbox else 1.0

            align_prev: Optional[float] = _alignment(centroid, prev_centroid)
            align_next: Optional[float] = _alignment(centroid, next_centroid)
            
            max_size_num_vals: float = 114.0
            numeric_values: List[float] = [float(x) for x in line_values]
            if len(numeric_values) < 2:
                return {}
            
            # Calcular estadísticos básicos
            count: float = float(len(numeric_values))
            mean: float = sum(numeric_values) / count
            variance: float = sum((x - mean) ** 2 for x in numeric_values) / (count - 1) if count > 1 else 0.0
            std_dev: float = math.sqrt(variance)

            # Calcular etadisticos especiales
            mean_rel: float = mean / max_size_num_vals if mean > 0.0 else 0.0
            count_rel: float = count / max_char_count if count > 0.0 else 0.0
            if mean > 0.0:
                mean_ref: float = max_size_num_vals / 2
                mean_margin: float = (mean - mean_ref) / (mean_ref)
                mean_margin = max(-1.0, min(1.0, mean_margin))
            else: 
                mean_margin = 0.0

            # Calcular percentiles
            sorted_values: List[float] = sorted(numeric_values)
                
            def _percentile(p: int) -> float:
                index: float = (p / 100) * (int(count) - 1)
                lower: int = int(index)
                upper: int = min(lower + 1, int(count) - 1)
                weight: float = float(index - lower)
                return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
            
            p25: float = _percentile(25)
            p50: float = _percentile(50)
            # p75: float = _percentile(75)
            # iqr: float = p75 - p25
                
                # Calcular skewness
            skewness: float = 0.0
            if std_dev > 0:
                moment3: float = sum(((x - mean) / std_dev) ** 3 for x in numeric_values)
                skewness = moment3 / count
                    
            # Anida el diccionario de características para que coincida con el tipo de retorno esperado.
            all_features: Dict[str, float] = {
                'count_rel': count_rel,
                'mean_rel': mean_rel,
                'std_dev': std_dev,
                'mean_margin': mean_margin,
                'p25': p25,
                'p50': p50,
                # 'p75': p75,
                # 'iqr': iqr,
                'skewness': skewness,
                "numeric_count_norm": numeric_count_norm,
                "numeric_frec_rel": numeric_frec_rel,
                "numeric_ratio_frec": numeric_ratio_frec,
                "num_above": num_above,
                "num_margin": num_margin,
                "digit_char_frec": digit_char_frec,
                "line_area": line_area,
                "width_rel": width_rel,
                "ratio_area": ratio_area,
                "aspect_ratio": aspect_ratio,
                "bbox_height": bbox_height,
                "perimeter": perimeter,
                "diagonal": diagonal,
                "compact": compact,
                "prev_xmin_align": prev_xmin_align if prev_xmin_align is not None else 1.0,
                "prev_xmax_align": prev_xmax_align if prev_xmax_align is not None else 1.0,
                "next_xmin_align": next_xmin_align if next_xmin_align is not None else 1.0,
                "next_xmax_align": next_xmax_align if next_xmax_align is not None else 1.0,
                "align_prev": align_prev,
                "align_next": align_next,
            }
            return {"aggregate_stats": all_features}

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}", exc_info=True)
        return {}