# PerfectOCR/core/workflow/vectorial_transformation/density_scanner.py
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
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
            start_time: float = time.time()
            logger.debug("Scanner iniciado")
            
            # Obtener líneas codificadas del DataFormatter
            encoded_lines: Dict[str, List[int]] = manager.get_encode_lines()
            if not encoded_lines:
                logger.warning("No hay líneas codificadas para analizar.")
                return False
            else:
                logger.info(f"Lineas codificadas: {encoded_lines}")
            
            # Obtener geometrías de líneas
            all_lines: Dict[str, Dict[str, Any]] = manager.get_dict_data().get("all_lines", {})
            
            if not all_lines:
                logger.warning("No hay líneas disponibles para analizar.")
                return False
            
            # Verificar que al menos una línea tenga geometría válida
            has_valid_geometry = False
            for line_data in all_lines.values():
                line_geometry = line_data.get("line_geometry", {})
                if line_geometry.get("line_bbox") and line_geometry.get("line_centroid"):
                    has_valid_geometry = True
                    break
            
            if not has_valid_geometry:
                logger.warning("No hay geometrías de líneas válidas disponibles.")
                return False
        
            # Analizar líneas y detectar tablas
            table_detection_result: Dict[str, Any] = self._detect_tables_from_encoded_lines(encoded_lines, all_lines)
            
            # Guardar resultados si hay tablas detectadas
            if table_detection_result["status"] == "success" and table_detection_result["table_lines"]:
                total_time: float = time.time() - start_time
                logger.info(f"Detección de tablas completada en {total_time:.4f}s")
                
                success: bool = manager.save_tabular_lines(table_detection_result)
                if not success:
                    logger.error("Error al guardar líneas tabulares en el workflow_dict")
                    return False
            else:
                logger.info("No se detectaron tablas en el documento")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en DensityScanner: {e}")
            return False

    def _detect_tables_from_encoded_lines(self, encoded_lines: Dict[str, List[int]], all_lines: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detecta tablas usando DBSCAN en líneas ya codificadas.
        """
        try:
            line_ids: List[str] = list(encoded_lines.keys())
            line_analyses: List[Optional[Dict[str, Dict[str, float]]]] = []
            
            # 1. Calcular todas las características geométricas una sola vez.
            all_geometric_features: Optional[Dict[str, Dict[str, float]]] = self._calculate_geometric_features(all_lines)
            if not all_geometric_features:
                logger.warning("No se pudieron calcular las características geométricas para ninguna línea.")
                return {"status": "error", "table_lines": []}

            # 2. Iterar y analizar cada línea, pasando las características precalculadas.
            for line_id in line_ids:
                line_values: List[int] = encoded_lines[line_id]
                
                geometric_features: Optional[Dict[str, float]] = all_geometric_features.get(line_id)
                line_geometry: Dict[str, Any] = all_lines.get(line_id, {}).get("line_geometry", {})

                if len(line_values) >= 2 and line_geometry and geometric_features:
                    analysis: Optional[Dict[str, Dict[str, float]]] = self._analyze_encoded_line(line_id, line_values, geometric_features)
                    line_analyses.append(analysis)
                else:
                    line_analyses.append(None)
            
            # Filtrar análisis válidos
            valid_analyses: List[Dict[str, Dict[str, float]]] = [a for a in line_analyses if a is not None]
            valid_indices: List[int] = [i for i, a in enumerate(line_analyses) if a is not None]
            
            if len(valid_analyses) < 2:
                logger.warning("No hay suficientes líneas válidas para clustering.")
                return {"status": "insufficient_data", "table_lines": []}
            
            # Aplicar DBSCAN
            table_indices: List[int] = self._apply_dbscan_clustering(valid_analyses, valid_indices)
            
            # Expandir a intervalo consecutivo
            if table_indices:
                consecutive_indices: List[int] = self._expand_to_consecutive_interval(table_indices)
                table_line_ids: List[str] = [line_ids[i] for i in consecutive_indices if i < len(line_ids)]
            else:
                consecutive_indices: List[int] = []
                table_line_ids: List[str] = []
            
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
        
    def _analyze_encoded_line(self, line_id: str, line_values: List[int], geometric_features: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Analiza una línea codificada y retorna estadísticas.
        """
        try:
            # Las características geométricas ahora se reciben como parámetro.
            bbox_width: float = geometric_features.get("bbox_width", 0.0)
            align_prev: float = geometric_features.get("align_prev", 0.0)
            align_next: float = geometric_features.get("align_next", 0.0)
            var_alignment: float = geometric_features.get("var_alignment", 0.0)
            
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
                "bbox_width": bbox_width,
                "align_prev": align_prev,
                "align_next": align_next,
                "var_alignment": var_alignment,
            }

            return {"aggregate_stats": feature_dict}
            
        except Exception as e:
            logger.error(f"Error analizando línea {line_id}: {e}")
            return None
    
    def _calculate_geometric_features(self, all_lines: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Calcula features geométricos + alineación tabular por cada línea.
        Retorna un diccionario con features por cada línea.
        """
        try:
            # Ordenar líneas por coordenada Y del centroide
            if not all_lines:
                return None

            sorted_lines: List[Tuple[str, Dict[str, Any]]] = sorted(
                all_lines.items(),
                key=lambda kv: kv[1].get("line_geometry", {}).get("line_centroid", [0, 0])[1]
            )

            all_geometric_features: Dict[str, Dict[str, float]] = {}
            for i, (line_id, line_data) in enumerate(sorted_lines):
                line_geometry: Dict[str, Any] = line_data.get("line_geometry", {}) 
                bbox: List[float] = line_geometry.get("line_bbox", [])
                centroid: List[float] = line_geometry.get("line_centroid", [])

                if len(bbox) < 4 or len(centroid) < 2:
                    continue

                bbox_width: float = float(bbox[2] - bbox[0])
                cx: float = float(centroid[0])
                cy: float = float(centroid[1])

                # vecinos arriba/abajo
                prev_centroid: Optional[List[float]] = sorted_lines[i-1][1]["line_geometry"]["line_centroid"] if i > 0 else None
                next_centroid: Optional[List[float]] = sorted_lines[i+1][1]["line_geometry"]["line_centroid"] if i < len(sorted_lines)-1 else None

                # función auxiliar para similitud coseno con eje X
                def alignment(ref_c: List[float], other_c: Optional[List[float]]) -> Optional[float]:
                    if other_c is None: 
                        return None
                    vec: np.ndarray = np.array([other_c[0] - ref_c[0], other_c[1] - ref_c[1]])
                    axis: np.ndarray = np.array([1, 0])  # eje X
                    if np.linalg.norm(vec) == 0: 
                        return 1.0
                    return float(np.dot(vec, axis) / (np.linalg.norm(vec) * np.linalg.norm(axis)))

                align_prev: Optional[float] = alignment(centroid, prev_centroid)
                align_next: Optional[float] = alignment(centroid, next_centroid)

                # varianza entre alineaciones válidas
                align_values: List[float] = [v for v in [align_prev, align_next] if v is not None]
                var_alignment: float = float(np.var(align_values)) if len(align_values) > 1 else 0.0

                all_geometric_features[line_id] = {
                    "bbox_width": bbox_width,
                    "align_prev": align_prev if align_prev is not None else 0.0,
                    "align_next": align_next if align_next is not None else 0.0,
                    "var_alignment": var_alignment,
                }

            return all_geometric_features

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}")
            return None
                                                
    def _apply_dbscan_clustering(self, valid_analyses: List[Dict[str, Dict[str, float]]], valid_indices: List[int]) -> List[int]:
        """
        Aplica DBSCAN para agrupar líneas similares.
        """
        # Obtener parámetros de configuración
        try:
            min_cluster_size: int = int(self.worker_config.get("min_cluster_size", 2))
            if min_cluster_size < 1:
                min_cluster_size = 2
        except (TypeError, ValueError):
            min_cluster_size = 2
        
        try:
            eps: float = float(self.worker_config.get("eps", 1.0))
            if eps <= 0:
                eps = 1.0
        except (TypeError, ValueError):
            eps = 1.0
        
        try:
            # Preparar features para clustering
            features: List[List[float]] = []
            for analysis in valid_analyses:
                aggregate_stats: Dict[str, float] = analysis.get('aggregate_stats', {})
                features.append([
                    aggregate_stats.get('count', 0.0),
                    aggregate_stats.get('mean', 0.0),
                    aggregate_stats.get('std_dev', 0.0),
                    aggregate_stats.get('iqr', 0.0),
                    aggregate_stats.get('p50', 0.0),
                    aggregate_stats.get('skewness', 0.0),
                    aggregate_stats.get('bbox_width', 0.0),
                    aggregate_stats.get('align_prev', 0.0),
                    aggregate_stats.get('align_next', 0.0),
                    aggregate_stats.get('var_alignment', 0.0),
                ])
            
            features_array: np.ndarray = np.array(features)
            
            # Escalar features
            scaler: StandardScaler = StandardScaler()
            features_scaled: np.ndarray = scaler.fit_transform(features_array)
            
            # Aplicar DBSCAN
            clustering: DBSCAN = DBSCAN(eps=eps, min_samples=min_cluster_size)
            labels: np.ndarray = clustering.fit_predict(features_scaled)
            
            logger.debug(f"DBSCAN: eps={eps}, min_samples={min_cluster_size}, labels={labels}")
            
            # Encontrar cluster principal (excluyendo ruido -1)
            unique_labels: List[int] = [l for l in set(labels) if l != -1]
            if not unique_labels:
                logger.debug("DBSCAN: No se encontraron clusters válidos (todos son ruido -1)")
                return []
            
            cluster_sizes: Dict[int, int] = {label: list(labels).count(label) for label in unique_labels}
            main_cluster = max(cluster_sizes, key=lambda x: cluster_sizes[x])
            
            logger.debug(f"DBSCAN: cluster_sizes={cluster_sizes}, main_cluster={main_cluster}")
            
            # Retornar índices de líneas de tabla
            table_indices: List[int] = [valid_indices[i] for i, label in enumerate(labels) if label == main_cluster]
            
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
        
        start: int = min(indices)
        end: int = max(indices)
        return list(range(start, end + 1))