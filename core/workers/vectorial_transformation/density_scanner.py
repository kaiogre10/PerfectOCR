# PerfectOCR/core/workflow/vectorial_transformation/density_scanner.py
from sklearn.cluster import DBSCAN #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
import math
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines

logger = logging.getLogger(__name__)

class DensityScanner(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('dbscan', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)        
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time: float = time.time()
            logger.info("DBSCScanner iniciado")
            
            img_dims = manager.workflow.metadata.img_dims if manager.workflow and hasattr(manager.workflow.metadata, "img_dims") else {}                
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            encoded_lines = manager.get_encode_lines()
            
            table_detection_result = self._detect_tables_from_encoded_lines(encoded_lines, all_lines, img_dims, manager)
                                        
            if table_detection_result.get("table_lines"):
                total_time = time.time() - start_time
                logger.info(f"Detección de tablas completada en {total_time:.4f}s")
            
            line_ids: List[str] = table_detection_result.get("table_lines", [])
            if line_ids:
                success: bool = manager.save_tabular_lines(line_ids)
                if success:
                    logger.info("Lineas guardadas en el manager desde DBSCAN")
                    return True
                else:
                    logger.error("Error al guardar líneas tabulares en el workflow")
                    return False
            else:
                logger.info("DBSCAN no detecto tablas en el documento")
                return False
            
        except Exception as e:
            logger.error(f"Error en DensityScanner: {e}", exc_info=True)
            return False

    def _detect_tables_from_encoded_lines(self, encoded_lines: Dict[str, List[int]], all_lines: Dict[str, AllLines], img_dims: Dict[str, int], manager: DataFormatter) -> Dict[str, Any]:
        """Detecta tablas usando DBSCAN en líneas ya codificadas."""
        try:
            line_ids: List[str] = list(encoded_lines.keys())
            line_analyses: List[Optional[Dict[str, Dict[str, float]]]] = []
            
            all_geometric_features: Optional[Dict[str, Dict[str, float]]] = self._calculate_geometric_features(all_lines, img_dims, manager)
            if not all_geometric_features:
                logger.warning("No se pudieron calcular las características geométricas para ninguna línea.")
                return {"status": "error", "table_lines": []}

            for line_id in line_ids:
                line_values: List[int] = encoded_lines[line_id]
                geometric_features: Optional[Dict[str, float]] = all_geometric_features.get(line_id)
                
                line_obj = all_lines.get(line_id)
                line_bbox = line_obj.line_geometry.line_bbox if line_obj else []
                line_centroid = line_obj.line_geometry.line_centroid if line_obj else []

                if len(line_values) >= 2 and line_bbox and line_centroid and geometric_features:
                    analysis: Optional[Dict[str, Dict[str, float]]] = self._analyze_encoded_line(line_id, line_values, geometric_features)
                    line_analyses.append(analysis)
                else:
                    line_analyses.append(None)
            
            valid_analyses: List[Dict[str, Dict[str, float]]] = [a for a in line_analyses if a is not None]
            valid_indices: List[int] = [i for i, a in enumerate(line_analyses) if a is not None]
            
            if len(valid_analyses) < 2:
                logger.warning("No hay suficientes líneas válidas para clustering.")
                return {"status": "insufficient_data", "table_lines": []}
            
            table_indices: List[int] = self._apply_dbscan_clustering(valid_analyses, valid_indices)
            
            if table_indices:
                consecutive_indices: List[int] = self._expand_to_consecutive_interval(table_indices)
                table_line_ids: List[str] = [line_ids[i] for i in consecutive_indices if i < len(line_ids)]
            else:
                consecutive_indices = []
                table_line_ids = []
            
            return {
                "status": "success",
                "table_lines": table_line_ids,
            }
            
        except Exception as e:
            logger.error(f"Error detectando tablas: {e}", exc_info=True)
            return {"status": "error", "table_lines": []}
        
    def _analyze_encoded_line(self, line_id: str, line_values: List[int], geometric_features: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
        """Analiza una línea codificada y retorna estadísticas."""
        try:
            # Las características geométricas ahora se reciben como parámetro.
            line_area: float = geometric_features.get("line_area", 0.0)
            bbox_width: float = geometric_features.get("bbox_width", 0.0)
            align_prev: float = geometric_features.get("align_prev", 1.0)
            align_next: float = geometric_features.get("align_next", 1.0)
            ratio_area: float = geometric_features.get("ratio_area", 0.0) 
            aspect_ratio: float = geometric_features.get("aspect_ratio", 0.0)
            perimeter: float = geometric_features.get("perimeter", 0.0)
            prev_xmin_align: float = geometric_features.get("prev_xmin_align", 1.0)
            prev_xmax_align: float = geometric_features.get("prev_xmax_align", 1.0)
            next_xmin_align: float = geometric_features.get("next_xmin_align", 1.0)
            next_xmax_align: float = geometric_features.get("next_xmax_align", 1.0)
            numeric_count: float = geometric_features.get("numeric_count", 0.0)
            numeric_ratio: float = geometric_features.get("numeric_ratio", 0.0)
            
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
                "ratio_area": ratio_area,
                "aspect_ratio": aspect_ratio,
                "perimeter": perimeter,
                "prev_xmin_align": prev_xmin_align,
                "prev_xmax_align": prev_xmax_align, 
                "next_xmin_align": next_xmin_align ,
                "next_xmax_align": next_xmax_align ,
                "numeric_count": numeric_count,
                "numeric_ratio": numeric_ratio,
            }

            return {"aggregate_stats": feature_dict}
            
        except Exception as e:
            logger.error(f"Error analizando línea {line_id}: {e}", exc_info=True)
            return None
    
    def _calculate_geometric_features(self, all_lines: Dict[str, AllLines], img_dims: Dict[str, int], manager: DataFormatter) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calcula features geométricos + alineación tabular por cada línea.
        Retorna un diccionario con features por cada línea.
        """
        try:
            if not manager.workflow.all_lines if manager.workflow else {}:
                return None

            sorted_lines: List[Tuple[str, AllLines]] = sorted(
                all_lines.items(),
                key=lambda kv: kv[1].line_geometry.line_centroid[1]
            )

            all_geometric_features: Dict[str, Dict[str, float]] = {}
            for i, (line_id, line_data) in enumerate(sorted_lines):
                numeric_count = 0.0
                code_count = 0.0
                descriptive_count = 0.0

                poly_ids_line = getattr(line_data, "polygon_ids", []) or []
                if manager.workflow and manager.workflow.polygons and poly_ids_line:
                    polygons_dict = manager.workflow.polygons
                    for pid in poly_ids_line:
                        if pid in polygons_dict:
                            semantic = getattr(polygons_dict[pid], "semantic_type", "") or ""
                            if semantic == "numeric":
                                numeric_count += 1.0
                            elif semantic == "code":
                                code_count += 1.0
                            else:
                                descriptive_count += 1.0

                total_polygons: float = numeric_count + code_count + descriptive_count
                numeric_ratio: Optional[float] = (numeric_count / total_polygons) * 100.0
                desc_ratio: Optional[float] = (descriptive_count / total_polygons) * 100.0
                    
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
                height: float = bbox[3] - bbox[1]
                bbox_width: float = float(bbox[2] - bbox[0])
                perimeter: float = 2 * (bbox_width + height)
                prev_bbox: Optional[List[float]] = sorted_lines[i-1][1].line_geometry.line_bbox if i > 0 else None
                next_bbox: Optional[List[float]] = sorted_lines[i+1][1].line_geometry.line_bbox if i < len(sorted_lines) - 1 else None
                prev_centroid: Optional[List[float]] = sorted_lines[i-1][1].line_geometry.line_centroid if i > 0 else None
                next_centroid: Optional[List[float]] = sorted_lines[i+1][1].line_geometry.line_centroid if i < len(sorted_lines) - 1 else None

                # función auxiliar para similitud coseno con eje X
                def alignment(ref_c: List[float], other_c: Optional[List[float]]) -> Optional[float]:
                    if other_c is None: 
                        return 1.0
                    vec = np.array([other_c[0] - ref_c[0], other_c[1] - ref_c[1]])
                    axis = np.array([1, 0])  # eje X
                    if np.linalg.norm(vec) == 0: 
                        return 1.0
                    return float(np.dot(vec, axis) / (np.linalg.norm(vec) * np.linalg.norm(axis)))

                def bbox_alignment(current_coord: float, other_bbox: Optional[List[float]], coord_idx: int) -> Optional[float]:
                    if other_bbox is None or len(other_bbox) < 4:
                        return 1.0
                    other_coord = other_bbox[coord_idx]
                    diff = abs(current_coord - other_coord)
                    max_tolerance = bbox_width if bbox_width > 0 else 100.0
                    alignment_score = max(0.0, 1.0 - (diff / max_tolerance))
                    return float(alignment_score)

                # Alineación ortogonal para xmin y xmax con prev y next
                current_xmin = bbox[0]
                current_xmax = bbox[2]

                prev_xmin_align: Optional[float] = bbox_alignment(current_xmin, prev_bbox, 0)
                prev_xmax_align: Optional[float] = bbox_alignment(current_xmax, prev_bbox, 2)
                next_xmin_align: Optional[float] = bbox_alignment(current_xmin, next_bbox, 0)
                next_xmax_align: Optional[float] = bbox_alignment(current_xmax, next_bbox, 2)

                align_prev: Optional[float] = alignment(centroid, prev_centroid)
                align_next: Optional[float] = alignment(centroid, next_centroid)

                # varianza entre alineaciones válidas de centroides
                align_values: List[float] = [v for v in [align_prev, align_next] if v is not None]
                var_alignment: float = float(np.var(align_values)) if len(align_values) > 1 else 0.0

                # varianza entre alineaciones válidas de bbox (xmin/xmax con prev/next)
                xmin_align_values: List[float] = [v for v in [prev_xmin_align, next_xmin_align] if v is not None]
                xmax_align_values: List[float] = [v for v in [prev_xmax_align, next_xmax_align] if v is not None]

                all_geometric_features[line_id] = {
                    "line_area": line_area,
                    "bbox_width": bbox_width,
                    "align_prev": align_prev if align_prev is not None else 1.0,
                    "align_next": align_next if align_next is not None else 1.0,
                    "ratio_area": ratio_area,
                    "aspect_ratio": aspect_ratio,
                    "perimeter": perimeter,
                    "prev_xmin_align": prev_xmin_align if prev_xmin_align is not None else 1.0,
                    "prev_xmax_align": prev_xmax_align if prev_xmax_align is not None else 1.0,
                    "next_xmin_align": next_xmin_align if next_xmin_align is not None else 1.0,
                    "next_xmax_align": next_xmax_align if next_xmax_align is not None else 1.0,
                    "numeric_count": numeric_count,
                    "numeric_ratio": numeric_ratio,
                }
            return all_geometric_features

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}", exc_info=True)
            return None
                                                
    def _apply_dbscan_clustering(self, valid_analyses: List[Dict[str, Dict[str, float]]], valid_indices: List[int]) -> List[int]:
        """Aplica DBSCAN para agrupar líneas similares."""
        min_cluster_size = int(self.worker_config.get("min_cluster_size", 2))
        eps = float(self.worker_config.get("eps", 2.0))
        
        features: List[List[float]] = []
        for analysis in valid_analyses:
            aggregate_stats: Dict[str, float] = analysis.get('aggregate_stats', {})
            features.append(list(aggregate_stats.values()))
        
        features_array = np.asarray(features, dtype=np.float64)

        scaler = StandardScaler()
        features_scaled: np.ndarray[Any, np.dtype[np.float64]]  = scaler.fit_transform(features_array)

        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
        labels: np.ndarray[Any, np.dtype[np.uint8]] = clustering.fit_predict(features_scaled)
        
        logger.debug(f"DBSCAN: eps={eps}, min_samples={min_cluster_size}, labels={labels}")
        
        unique_labels: List[int] = [l for l in set(labels) if l != -1]
        if not unique_labels:
            logger.warning("DBSCAN: No se encontraron clusters válidos.")
            return []
        
        cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
        main_cluster = max(cluster_sizes, key=cluster_sizes.get)
        
        logger.debug(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}")
        
        table_indices: List[int] = [valid_indices[i] for i, label in enumerate(labels) if label == main_cluster] 
        
        return table_indices

    def _expand_to_consecutive_interval(self, indices: List[int]) -> List[int]:
        """
        Expande lista de índices a intervalo consecutivo.
        """
        if not indices:
            return []
        
        start: int = min(indices)
        end: int = max(indices)
        return list(range(start, end + 1))

