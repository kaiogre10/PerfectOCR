# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import math
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('cos_sim', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)        
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time: float = time.time()
            logger.info("Calculando matriz de similitud")
            
            img_dims = manager.workflow.metadata.img_dims if manager.workflow and hasattr(manager.workflow.metadata, "img_dims") else {}
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}

            header_line_id = self._find_header_line_id(polygons, all_lines)
            if header_line_id is not None:
                manager.update_header(header_line_id)
                logger.info(f"Linea de encabezado actualizada desde Coseno")
                encoded_lines = manager.get_encode_lines()

                # pasar img_dims en la posición correcta
                table_detection_result = self._detect_by_cosine_similarity(encoded_lines, all_lines, img_dims, header_line_id)

                if table_detection_result.get("table_lines"):
                    total_time = time.time() - start_time
                    logger.info(f"Detección de tablas completada coseno en {total_time:.6f}s")
                    return True
                
                line_ids: List[str] = table_detection_result.get("table_lines", [])
                if line_ids:
                    success: bool = manager.save_tabular_lines(line_ids)
                    if success:
                        logger.info("Lineas guardadas en el manager desde COSENO")
                        return True
                    else:
                        logger.info("Error al guardar líneas tabulares en el workflow")
                        return False
                else:
                    logger.info("COSENO no detecto tablas en el documento")
                    return False
            else:
                logger.info("Enviando Lineas a DBSCAN")
                return True
        except Exception as e:
            logger.info(f"Error en matriz de similitud coseno: {e}", exc_info=True)
            return False
    
    def _detect_by_cosine_similarity(self, encoded_lines: Dict[str, List[int]], all_lines: Dict[str, AllLines], img_dims: Dict[str, int], header_line_id: str) -> Dict[str, Any]:
        """Estrategia de similitud al encabezado (Plan A) usando Similitud Coseno."""
        min_cluster: int = self.worker_config.get("min_cluster", {})
        try:
            line_ids: List[str] = list(encoded_lines.keys())
            line_analyses: List[Optional[Dict[str, Dict[str, float]]]] = []
            
            all_geometric_features: Optional[Dict[str, Dict[str, float]]] = self._calculate_geometric_features(all_lines, img_dims)
            if not all_geometric_features:
                logger.warning("No se pudieron calcular las características geométricas para ninguna línea.")
                return {"status": "error", "table_lines": []}

            for line_id in line_ids:
                line_values: List[int] = encoded_lines[line_id]
                geometric_features: Optional[Dict[str, float]] = all_geometric_features.get(line_id)
                
                line_obj = all_lines.get(line_id)
                line_bbox = line_obj.line_geometry.line_bbox if line_obj else []
                line_centroid = line_obj.line_geometry.line_centroid if line_obj else []

                if len(line_values) >= min_cluster and line_bbox and line_centroid and geometric_features:
                    analysis: Optional[Dict[str, Dict[str, float]]] = self._analyze_encoded_line(line_id, line_values, geometric_features)
                    line_analyses.append(analysis)
                else:
                    line_analyses.append(None)
                                
            valid_analyses: List[Dict[str, Dict[str, float]]] = [a for a in line_analyses if a is not None]
            valid_indices: List[int] = [i for i, a in enumerate(line_analyses) if a is not None]

            if len(valid_analyses) < min_cluster:
                logger.warning("No hay suficientes líneas válidas para ")
                return None

            table_indices: List[int] = self._calculate_cosine_matrix(valid_analyses, valid_indices, header_line_id, encoded_lines)
            
            if table_indices:
                consecutive_indices: List[int] = self._expand_to_consecutive_interval(table_indices)
                table_line_ids: List[str] = [line_ids[i] for i in consecutive_indices if i < len(line_ids)]
            else:
                consecutive_indices = []
                table_line_ids = []
            
            return {
                "table_lines": table_line_ids,
            }
            
        except Exception as e:
            logger.error(f"Error detectando tablas: {e}", exc_info=True)
            return None
        
    def _calculate_cosine_matrix(self, valid_analyses: List[Dict[str, Dict[str, float]]], valid_indices: List[int], header_line_id: str, encoded_lines: Dict[str, List[int]]) -> Optional[List[int]]:
        try:
            feature_keys = [
                'count','mean','std_dev','iqr','p50','skewness',
                'line_area','bbox_width','align_prev', 'align_next', 'ratio_area',
                'aspect_ratio','perimeter','xmin_align','xmax_align'
            ]

            # convertir header_line_id string a índice numérico
            line_ids: List[str] = list(encoded_lines.keys())
            try:
                header_idx = line_ids.index(header_line_id)
                hdr_pos = valid_indices.index(header_idx)
            except ValueError:
                logger.error(f"Header {header_line_id} no encontrado en valid_indices")
                return None

            # construir vector header
            header_analysis = valid_analyses[hdr_pos]
            header_vec = np.asarray([float(header_analysis['aggregate_stats'].get(k, 0.0)) for k in feature_keys], dtype=np.float32).reshape(1, -1)
            if np.linalg.norm(header_vec) == 0:
                return 1.0

            logger.debug(f"Header idx={header_idx}, pos={hdr_pos}, vector shape={header_vec.shape}")

            # candidatos: TODAS las líneas válidas EXCEPTO el header
            candidate_analyses = [valid_analyses[i] for i in range(len(valid_analyses)) if i != hdr_pos]
            candidate_indices = [valid_indices[i] for i in range(len(valid_indices)) if i != hdr_pos]
            
            if not candidate_analyses:
                return None

            logger.debug(f"Candidatos: {len(candidate_analyses)} líneas, índices: {candidate_indices}")

            # construir matriz densa (N_candidates x F)
            mat = np.asarray(
                [[float(a['aggregate_stats'].get(k, 0.0)) for k in feature_keys] for a in candidate_analyses],
                dtype=np.float32
            )

            # Imprimir valores de manera legible y estructurada
            # Mostrar la matriz de candidatos en formato de tabla: filas = líneas, columnas = features
            header = ["Idx", "Tabla idx", "Line ID"] + feature_keys
            row_format = "{:<5} {:<10} {:<20} " + " ".join(["{:>10.4f}"] * len(feature_keys))
            logger.info("Valores de la matriz de candidatos (cada fila es una línea, columnas = features):")
            logger.info("{:<5} {:<10} {:<20} ".format("Idx", "Tabla idx", "Line ID") + " ".join([f"{k:>10}" for k in feature_keys]))
            for idx, (orig_idx, row) in enumerate(zip(candidate_indices, mat)):
                line_id = line_ids[orig_idx] if orig_idx < len(line_ids) else f"idx_{orig_idx}"
                logger.info(row_format.format(idx, orig_idx, line_id, *row))

            # Combinar header y candidatos para un escalado consistente
            full_mat = np.vstack([header_vec, mat])

            # Escalar características para que tengan media 0 y desviación estándar 1
            scaler = MinMaxScaler()
            scaled_mat = scaler.fit_transform(full_mat)

            # Log de los valores escalados, mismo formato que el log anterior
            logger.info("Valores ESCALADOS de la matriz de candidatos (cada fila es una línea, columnas = features):")
            logger.info("{:<5} {:<10} {:<20} ".format("Idx", "Tabla idx", "Line ID") + " ".join([f"{k:>10}" for k in feature_keys]))
            # El primer elemento de scaled_mat es el header, los siguientes son los candidatos
            for idx, (orig_idx, row) in enumerate(zip(candidate_indices, scaled_mat[1:])):
                line_id = line_ids[orig_idx] if orig_idx < len(line_ids) else f"idx_{orig_idx}"
                logger.info(row_format.format(idx, orig_idx, line_id, *row))


            
            # Separar header y candidatos escalados
            scaled_header_vec = scaled_mat[0].reshape(1, -1)
            scaled_candidates_mat = scaled_mat[1:]

            similarity_threshold = float(self.worker_config.get("similarity_threshold", 0.90))
            max_gap_tolerance = self.worker_config.get("interval", 2)

            sims = cosine_similarity(scaled_header_vec, scaled_candidates_mat)[0]
            logger.info(f"Similitudes vs header (excluyendo header): {sims.tolist()}")

            # aplicar lógica de gap tolerance solo para líneas DESPUÉS del header
            matched_original_indices: List[int] = []
            consecutive_failures = 0
            
            for i, (sim, orig_idx) in enumerate(zip(sims, candidate_indices)):
                # solo procesar líneas que vienen DESPUÉS del header
                if orig_idx > header_idx:
                    if sim >= similarity_threshold:
                        matched_original_indices.append(orig_idx)
                        consecutive_failures = 0
                        logger.info(f"Línea idx={orig_idx}: Similitud = {sim:.4f} ✓")
                    else:
                        consecutive_failures += 1
                        logger.info(f"Línea idx={orig_idx}: Similitud = {sim:.4f} (fallo {consecutive_failures}/{max_gap_tolerance})")
                        
                        if consecutive_failures > max_gap_tolerance:
                            logger.info(f"Gap de {consecutive_failures} líneas excede tolerancia. Rompiendo tabla.")
                            break
                else:
                    logger.debug(f"Línea idx={orig_idx}: antes del header, ignorada")

            # incluir el header en los índices finales
            if matched_original_indices:
                all_table_indices = [header_idx] + matched_original_indices
                return all_table_indices

            return None
        except Exception as e:
            logger.error(f"Error en detección por similitud de encabezado: {e}", exc_info=True)
            return None
    
    def _calculate_geometric_features(self, all_lines: Dict[str, AllLines], img_dims: Dict[str, int]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calcula features geométricos + alineación tabular por cada línea.
        Retorna un diccionario con features por cada línea.
        """
        try:
            if not all_lines:
                return None

            sorted_lines: List[Tuple[str, AllLines]] = sorted(
                all_lines.items(),
                key=lambda kv: kv[1].line_geometry.line_centroid[1]
            )

            all_geometric_features: Dict[str, Dict[str, float]] = {}
            for i, (line_id, line_data) in enumerate(sorted_lines):
                bbox: List[float] = line_data.line_geometry.line_bbox
                centroid: List[float] = line_data.line_geometry.line_centroid

                if len(bbox) < 4 or len(centroid) < 2:
                    continue

                # Proporción del área respecto del total
                if not img_dims or "size" not in img_dims:
                    return None
                
                total_size = img_dims.get("size")
                if not total_size:
                    return None
                
                line_area: float = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                ratio_area = (100.0 / float(total_size)) * line_area
                aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
                # El perímetro de un rectángulo se calcula como 2*(ancho + alto)
                width: float = bbox[2] - bbox[0]
                height: float = bbox[3] - bbox[1]
                perimeter: float = 2 * (width + height)
                bbox_width: float = float(bbox[2] - bbox[0])                
                prev_bbox: Optional[List[float]] = sorted_lines[i-1][1].line_geometry.line_bbox if i > 0 else None
                # vecinos arriba/abajo
                prev_centroid: Optional[List[float]] = sorted_lines[i-1][1].line_geometry.line_centroid if i > 0 else None
                next_centroid: Optional[List[float]] = sorted_lines[i-1][1].line_geometry.line_centroid if i > 0 else None

                # función auxiliar para similitud coseno con eje X
                def alignment(ref_c: List[float], other_c: Optional[List[float]]) -> Optional[float]:
                    if other_c is None: 
                        return None
                    vec = np.array([other_c[0] - ref_c[0], other_c[1] - ref_c[1]])
                    axis = np.array([1, 0])  # eje X
                    if np.linalg.norm(vec) == 0: 
                        return 0.0
                    return float(np.dot(vec, axis) / (np.linalg.norm(vec) * np.linalg.norm(axis)))

                def bbox_alignment(current_coord: float, prev_bbox: Optional[List[float]], coord_idx: int) -> Optional[float]:
                    if prev_bbox is None or len(prev_bbox) < 4:
                        return None
                    prev_coord = prev_bbox[coord_idx]
                    diff = abs(current_coord - prev_coord)
                    # Normalizar: menor diferencia = mayor alineación (0-1)
                    max_tolerance = bbox_width if bbox_width > 0 else 100.0
                    alignment_score = max(0.0, 1.0 - (diff / max_tolerance))
                    return float(alignment_score)

                # Alineación ortogonal para xmin y xmax
                current_xmin = bbox[0]
                current_xmax = bbox[2]
                
                xmin_align: Optional[float] = bbox_alignment(current_xmin, prev_bbox, 0)
                xmax_align: Optional[float] = bbox_alignment(current_xmax, prev_bbox, 2)

                align_prev: Optional[float] = alignment(centroid, prev_centroid)
                align_next: Optional[float] = alignment(centroid, next_centroid)

                # varianza entre alineaciones válidas
                # align_values: List[float] = [v for v in [align_prev, align_next] if v is not None]
                # var_alignment: float = float(np.var(align_values)) if len(align_values) > 1 else 0.0

                all_geometric_features[line_id] = {
                    "line_area": line_area,
                    "bbox_width": bbox_width,
                    "align_prev": align_prev if align_prev is not None else 0.0,
                    "align_next": align_next if align_next is not None else 0.0,
                    # "var_alignment": var_alignment,
                    "ratio_area": ratio_area,
                    "aspect_ratio": aspect_ratio,
                    "perimeter": perimeter,
                    "xmin_align": xmin_align if xmin_align is not None else 0.0,
                    "xmax_align": xmax_align if xmax_align is not None else 0.0,
                }
            return all_geometric_features

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}", exc_info=True)
            return None
        
    def _analyze_encoded_line(self, line_id: str, line_values: List[int], geometric_features: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
        """Analiza una línea codificada y retorna estadísticas."""
        try:
            # Las características geométricas ahora se reciben como parámetro.
            line_area: float = geometric_features.get("line_area", 0.0)
            bbox_width: float = geometric_features.get("bbox_width", 0.0)
            align_prev: float = geometric_features.get("align_prev", 0.0)
            align_next: float = geometric_features.get("align_next", 0.0)
            # var_alignment: float = geometric_features.get("var_alignment", 0.0)
            ratio_area: float = geometric_features.get("ratio_area", 0.0) 
            aspect_ratio: float = geometric_features.get("aspect_ratio", 0.0)
            perimeter: float = geometric_features.get("perimeter", 0.0)
            xmin_align: float = geometric_features.get("xmin_align", 0.0)
            xmax_align: float = geometric_features.get("xmax_align", 0.0)
            
            # Convertir valores codificados a numéricos
            numeric_values: List[float] = [float(x) for x in line_values]
            if len(numeric_values) < 2:
                return None
            
            # Calcular estadísticos básicos
            count: float = float(len(numeric_values))
            mean: float = sum(numeric_values) / len(numeric_values)
            variance: float = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1) if len(numeric_values) > 1 else 0.0
            std_dev: float = math.sqrt(variance)
            
            # Calcular percentiles
            sorted_values: List[float] = sorted(numeric_values)
            n: int = len(sorted_values)
            
            def percentile(p: float) -> float:
                index: float = (p / 100) * (n - 1)
                lower: int = int(index)
                upper: int = min(lower + 1, n - 1)
                weight: float = index - lower
                return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
            
            p25: float = percentile(25)
            p50: float = percentile(50)
            p75: float = percentile(75)
            iqr: float = p75 - p25
            
            # Calcular skewness
            skewness: float = 0.0
            if std_dev > 0:
                moment3: float = sum(((x - mean) / std_dev) ** 3 for x in numeric_values)
                skewness = moment3 / n
                
            # Anida el diccionario de características para que coincida con el tipo de retorno esperado.
            feature_dict: Dict[str, float] = {
                'count': count,
                'mean': mean,
                'std_dev': std_dev,
                'iqr': iqr,
                'p50': p50,
                'skewness': skewness,
                "line_area": line_area,
                "bbox_width": bbox_width,
                "align_prev": align_prev,
                "align_next": align_next,
                # "var_alignment": var_alignment,
                "ratio_area": ratio_area,
                "aspect_ratio": aspect_ratio,
                "perimeter": perimeter,
                "xmin_align": xmin_align,
                "xmax_align": xmax_align,
            }

            return {"aggregate_stats": feature_dict}
            
        except Exception as e:
            logger.error(f"Error analizando línea {line_id}: {e}", exc_info=True)
            return None

    def _expand_to_consecutive_interval(self, indices: List[int]) -> List[int]:
        """
        Expande lista de índices a intervalo consecutivo.
        """
        interval_expand = self.worker_config.get("interval", {})

        if not indices:
            return []
        
        start: int = min(indices)
        end: int = max(indices)
        return list(range(start, end + interval_expand))

    def _find_header_line_id(self, polygons: Dict[str, Polygons], all_lines: Dict[str, AllLines]) -> Optional[str]:
        """Localiza la line_id del encabezado basada en HeaderWords."""
        try:
            hdr_poly_ids = [pid for pid, p in polygons.items() if getattr(p, "key_field", None) == "HeaderWords"]
            if not hdr_poly_ids: 
                return None
            
            hdr_set = set(hdr_poly_ids)
            counts = {lid: len(set(lobj.polygon_ids).intersection(hdr_set)) for lid, lobj in all_lines.items() if lobj.polygon_ids}
            
            if not counts: 
                return None
        
            header_line_id = max(counts, key=counts.get)
        
            if not header_line_id:
                return None
            else:
                logger.info(f"Header_line_id={header_line_id} (via HeaderWords)")
                return header_line_id
        
        except Exception as e:
            logger.error(f"No hubo encabezado textual por similitud de encabezado: {e}", exc_info=True)