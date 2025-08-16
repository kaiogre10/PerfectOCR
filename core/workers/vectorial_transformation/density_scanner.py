# PerfectOCR/core/workflow/vectorial_transformation/density_scanner.py
from sklearn.cluster import DBSCAN
import math
import numpy as np
import time
import logging
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
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
            logger.info("Scanner iniciado")
            
            
            # Obtener líneas codificadas del DataFormatter
            encoded_lines = manager.encode_lines()
            if not encoded_lines:
                logger.warning("No hay líneas codificadas para analizar.")
                return False
            
            # Convertir a lista de valores para procesamiento
            lines_values = list(encoded_lines.values())
            line_ids = list(encoded_lines.keys())
            
            # Analizar líneas y detectar tablas
            table_detection_result = self._detect_tables_from_encoded_lines(lines_values, line_ids)
            
            # Llamar directamente al manager para guardar (como LinealReconstructor)
            if table_detection_result["status"] == "success" and table_detection_result["table_lines"]:
                success = manager.save_tabular_lines(table_detection_result)
                if not success:
                    logger.error("Error al guardar líneas tabulares en el workflow_dict")
                    return False
            else:
                logger.info("No se detectaron tablas en el documento")
            
            total_time = time.time() - start_time
            logger.info(f"Detección de tablas completada en {total_time:.4f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en DensityScanner: {e}")
            return False

    def _detect_tables_from_encoded_lines(self, lines_values: List[List[int]], line_ids: List[str]) -> Dict[str, Any]:
        """
        Detecta tablas usando DBSCAN en líneas ya codificadas.
        """
        try:
            # Analizar cada línea para obtener estadísticas
            line_analyses = []
            for line_values in lines_values:
                if len(line_values) >= 2:
                    analysis = self._analyze_encoded_line(line_values)
                    if analysis:
                        line_analyses.append(analysis)
                else:
                    line_analyses.append(None)
            
            # Filtrar análisis válidos
            valid_analyses = [a for a in line_analyses if a is not None]
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
                "total_lines_analyzed": len(lines_values),
                "table_lines_count": len(table_line_ids)
            }
            
        except Exception as e:
            logger.error(f"Error detectando tablas: {e}")
            return {"status": "error", "table_lines": []}

    def _analyze_encoded_line(self, line_values: List[int]) -> Dict[str, float]:
        """
        Analiza una línea codificada y retorna estadísticas.
        """
        try:
            # Filtrar solo valores numéricos
            numeric_values = [x for x in line_values if isinstance(x, (int, float))]
            if len(numeric_values) < 2:
                return None

            mean = sum(numeric_values) / len(numeric_values)
            variance = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
            std_dev = math.sqrt(variance)

            # Calcular percentiles
            sorted_values = sorted(numeric_values)
            n = len(sorted_values)

            def percentile(p):
                index = (p / 100) * (n - 1)
                lower = int(index)
                upper = min(lower + 1, n - 1)
                weight = index - lower
                return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

            p25, p50, p75 = percentile(25), percentile(50), percentile(75)
            iqr = p75 - p25

            # Calcular skewness
            if std_dev > 0:
                moment3 = sum(((x - mean) / std_dev) ** 3 for x in numeric_values)
                skewness = moment3 / n
            else:
                skewness = 0

            return {
                'count': len(numeric_values),
                'mean': mean,
                'std_dev': std_dev,
                'iqr': iqr,
                'p50': p50,
                'skewness': skewness
            }

        except Exception as e:
            logger.error(f"Error analizando línea: {e}")
            return None

    def _apply_dbscan_clustering(self, analyses: List[Dict[str, float]], valid_indices: List[int]) -> List[int]:
        """
        Aplica DBSCAN para agrupar líneas similares.
        """
        # Obtener parámetros de configuración con valores por defecto y casting robusto
        raw_min_samples = (
            self.worker_config.get("min_cluster_size")
            if isinstance(self.worker_config, dict) else None
        )
        if raw_min_samples is None:
            raw_min_samples = self.config.get("min_cluster_size", 2) if hasattr(self, "config") else 2
        try:
            min_cluster_size = int(raw_min_samples)
            if min_cluster_size < 1:
                min_cluster_size = 2  # Usar 2 como en el original
        except (TypeError, ValueError):
            min_cluster_size = 2  # Usar 2 como en el original

        raw_eps = (
            self.worker_config.get("eps")
            if isinstance(self.worker_config, dict) else None
        )
        if raw_eps is None:
            raw_eps = self.config.get("eps", 1.0) if hasattr(self, "config") else 1.0
        try:
            eps = float(raw_eps)
            if eps <= 0:
                eps = 1.0
        except (TypeError, ValueError):
            eps = 1.0
        try:
            # Preparar features para clustering
            features = []
            for analysis in analyses:
                features.append([
                    analysis['count'],
                    analysis['mean'],
                    analysis['std_dev'],
                    analysis['iqr'],
                    analysis['p50'],
                    analysis.get('skewness', 0.0),
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

