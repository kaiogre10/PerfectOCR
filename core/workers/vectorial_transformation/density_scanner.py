# PerfectOCR/core/workflow/vectorial_transformation/density_scanner.py
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class DensityScanner(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('dsbscan', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)        
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analiza líneas codificadas para detectar tablas usando DBSCAN.
        """
        try:
            start_time = time.time()
            logger.debug("Scanner iniciado")
            
            # Obtener líneas codificadas del DataFormatter
            encoded_lines: Dict[str, List[int]] = manager.get_encode_lines()
            if not encoded_lines:
                logger.warning("No hay líneas codificadas para analizar.")
                return False
            
            # Obtener geometrías de líneas
            lines_geometries = manager.get_dict_data().get("all_lines", {})
            if not lines_geometries:
                logger.warning("No hay geometrías de líneas disponibles.")
                return False
            
            # Analizar líneas y detectar tablas
            table_detection_result = self._detect_tables_from_encoded_lines(encoded_lines, lines_geometries)
            
            # Guardar resultados si hay tablas detectadas
            if table_detection_result["status"] == "success" and table_detection_result["table_lines"]:
                total_time = time.time() - start_time
                logger.debug(f"Detección de tablas completada en {total_time:.4f}s")
                
                success = manager.save_tabular_lines(table_detection_result)
                if not success:
                    logger.error("Error al guardar líneas tabulares en el workflow_dict")
                    return False
            else:
                logger.debug("No se detectaron tablas en el documento")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en DensityScanner: {e}")
            return False

    def _detect_tables_from_encoded_lines(self, encoded_lines: Dict[str, List[int]], lines_geometries: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta tablas usando DBSCAN en líneas ya codificadas.
        """
        try:
            line_ids = list(encoded_lines.keys())
            line_analyses = []
            
            # Analizar cada línea para obtener estadísticas
            for line_id in line_ids:
                line_values = encoded_lines[line_id]
                line_geometry = lines_geometries.get(line_id, {})
                
                if len(line_values) >= 2 and line_geometry:
                    analysis = self._analyze_encoded_line(line_id, line_values, line_geometry)
                    line_analyses.append(analysis)
                else:
                    line_analyses.append(None)
            
            # Filtrar análisis válidos
            valid_analyses: List[Dict[str, float]] = [a for a in line_analyses if a is not None]
            valid_indices = [i for i, a in enumerate(line_analyses) if a is not None]
            
            if len(valid_analyses) < 2:
                logger.warning("No hay suficientes líneas válidas para clustering.")
                return {"status": "insufficient_data", "table_lines": []}
            
            # Aplicar DBSCAN
            table_indices = self._apply_dbscan_clustering(valid_analyses, valid_indices)
            
            # Expandir a intervalo consecutivo
            if table_indices:
                consecutive_indices = self._expand_to_consecutive_interval(table_indices)
                table_line_ids = [line_ids[i] for i in consecutive_indices if i < len(line_ids)]
            else:
                consecutive_indices = []
                table_line_ids = []
            
            return {
                "status": "success",
                "table_lines": table_line_ids,
                "table_indices": consecutive_indices,
                "total_lines_analyzed": len(line_ids),
                "table_lines_count": len(table_line_ids)
            }
            
        except Exception as e:
            logger.error(f"Error detectando tablas: {e}")
            return {"status": "error", "table_lines": []}
        
    def _analyze_encoded_line(self, line_id: str, line_values: List[int], line_geometry: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Analiza una línea codificada y retorna estadísticas.
        """
        try:
            # Calcular features geométricos
            geometric_features = self._calculate_geometric_features(line_geometry)
            if geometric_features is None:
                logger.warning(f"No se pudieron calcular features geométricos para línea {line_id}")
                return None
            
            bbox_width, centroid_x, centroid_y = geometric_features
            
            # Convertir valores codificados a numéricos
            numeric_values = [float(x) for x in line_values if isinstance(x, (int, float))]
            if len(numeric_values) < 2:
                return None
            
            # Calcular estadísticos básicos
            count = float(len(numeric_values))
            mean = sum(numeric_values) / len(numeric_values)
            variance = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1) if len(numeric_values) > 1 else 0.0
            std_dev = math.sqrt(variance)
            
            # Calcular percentiles
            sorted_values = sorted(numeric_values)
            n = len(sorted_values)
            
            def percentile(p: float) -> float:
                index = (p / 100) * (n - 1)
                lower = int(index)
                upper = min(lower + 1, n - 1)
                weight = index - lower
                return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
            
            p25, p50, p75 = percentile(25), percentile(50), percentile(75)
            iqr = p75 - p25
            
            # Calcular skewness
            skewness = 0.0
            if std_dev > 0:
                moment3 = sum(((x - mean) / std_dev) ** 3 for x in numeric_values)
                skewness = moment3 / n
            
            return {
                'count': count,
                'mean': mean,
                'std_dev': std_dev,
                'iqr': iqr,
                'p50': p50,
                'skewness': skewness,
                'bbox_width': bbox_width,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
            }
            
        except Exception as e:
            logger.error(f"Error analizando línea {line_id}: {e}")
            return None
    
    def _calculate_geometric_features(self, line_geometry: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        """
        Calcula features geométricos de una línea.
        Retorna: (bbox_width, centroid_x, centroid_y)
        """
        try:
            # Obtener bbox y centroide
            line_bbox = line_geometry.get("line_bbox", [])
            line_centroid = line_geometry.get("line_centroid", [])
            
            if len(line_bbox) < 4 or len(line_centroid) < 2:
                logger.warning("Geometría de línea incompleta")
                return None
            
            # Calcular dimensiones del bbox
            # Asumiendo formato [x_min, y_min, x_max, y_max]
            bbox_width = float(line_bbox[2] - line_bbox[0])
            
            # Obtener coordenadas del centroide
            centroid_x = float(line_centroid[0])
            centroid_y = float(line_centroid[1])
            
            return bbox_width, centroid_x, centroid_y
            
        except Exception as e:
            logger.error(f"Error calculando features geométricos: {e}")
            return None
                                                
    def _apply_dbscan_clustering(self, valid_analyses: List[Dict[str, float]], valid_indices: List[int]) -> List[int]:
        """
        Aplica DBSCAN para agrupar líneas similares.
        """
        # Obtener parámetros de configuración
        try:
            min_cluster_size = int(self.worker_config.get("min_cluster_size", 2))
            if min_cluster_size < 1:
                min_cluster_size = 2
        except (TypeError, ValueError):
            min_cluster_size = 2
        
        try:
            eps = float(self.worker_config.get("eps", 1.0))
            if eps <= 0:
                eps = 1.0
        except (TypeError, ValueError):
            eps = 1.0
        
        try:
            # Preparar features para clustering
            features = []
            for analysis in valid_analyses:
                features.append([
                    analysis['count'],
                    analysis['mean'],
                    analysis['std_dev'],
                    analysis['iqr'],
                    analysis['p50'],
                    analysis['skewness'],
                    analysis['bbox_width'],
                ])
            
            features_array = np.array(features)
            
            # Escalar features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # Aplicar DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_cluster_size)
            labels = clustering.fit_predict(features_scaled)
            
            logger.debug(f"DBSCAN: eps={eps}, min_samples={min_cluster_size}, labels={labels}")
            
            # Encontrar cluster principal (excluyendo ruido -1)
            unique_labels = [l for l in set(labels) if l != -1]
            if not unique_labels:
                logger.debug("DBSCAN: No se encontraron clusters válidos (todos son ruido -1)")
                return []
            
            cluster_sizes = {label: list(labels).count(label) for label in unique_labels}
            main_cluster = max(cluster_sizes, key=cluster_sizes.get)
            
            logger.debug(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}")
            
            # Retornar índices de líneas de tabla
            table_indices = [valid_indices[i] for i, label in enumerate(labels) if label == main_cluster]
            
            logger.debug(f"DBSCAN: table_indices={table_indices}")
            return table_indices
            
        except Exception as e:
            logger.error(f"Error en clustering DBSCAN: {e}")
            return []

    def _expand_to_consecutive_interval(self, indices: List[int]) -> List[int]:
        """
        Expande lista de índices a intervalo consecutivo.
        """
        if not indices:
            return []
        
        start = min(indices)
        end = max(indices)
        return list(range(start, end + 1))