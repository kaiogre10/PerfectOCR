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
from sklearn.preprocessing import MinMaxScaler # type: ignore

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
                
                # intentar obtener líneas detectadas por el scanner (DBSCAN)
                tabular_lines = manager.get_tabular_lines()
                # normalizar: si manager devuelve lista de dicts extraer ids
                if isinstance(tabular_lines, list) and tabular_lines:
                    if isinstance(tabular_lines[0], dict):
                        tabular_lines = [t.get('line_id') or t.get('id') or t.get('line') for t in tabular_lines if isinstance(t, dict)]
                encoded_lines = manager.get_encode_lines()


                # Si hay resultado del scanner, VALIDAR usando estrategia all-vs-all en el intervalo header+1 .. última del scanner
                if tabular_lines:
                    logger.info(f"Validando resultado del scanner con validación coseno all-vs-all ({len(tabular_lines)} líneas reportadas)")
                    validated = self._validate_scanner_interval_all_vs_all(encoded_lines, all_lines, img_dims, header_line_id, tabular_lines)
                    if validated:
                        total_time = time.time() - start_time
                        logger.info(f"Validación coseno completada en {total_time:.6f}s. Líneas válidas: {len(validated)}")
                        success: bool = manager.save_tabular_lines(validated)
                        if success:
                            logger.info("Lineas guardadas en el manager desde COSENO (validación all-vs-all)")
                            return True
                        else:
                            logger.error("Error al guardar líneas tabulares validadas en el workflow")
                            return False
                    else:
                        logger.info("Validación coseno rechazó las líneas detectadas por el scanner")
                        return False
                else:
                    # Si no hay líneas del scanner, cae a la detección por similitud tradicional
                    logger.info("No se encontraron líneas del scanner; intentando detección por similitud interna")
                    table_detection_result = self._detect_by_cosine_similarity(encoded_lines, all_lines, img_dims, header_line_id)

                    if table_detection_result and table_detection_result.get("table_lines"):
                        total_time = time.time() - start_time
                        logger.info(f"Detección de tablas completada coseno en {total_time:.6f}s")
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
        min_cluster: int = int(self.worker_config.get("min_cluster", 2))
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
            
            return {
                "table_lines": table_indices,
            }
            
        except Exception as e:
            logger.error(f"Error detectando tablas: {e}", exc_info=True)
            return None
        
    def _calculate_cosine_matrix(self, valid_analyses: List[Dict[str, Dict[str, float]]], valid_indices: List[int], header_line_id: str, encoded_lines: Dict[str, List[int]]) -> Optional[List[int]]:
        try:
            feature_keys = [
                'count','mean','std_dev','iqr','p50','skewness',
                'line_area','bbox_width','align_prev', 'align_next', "var_alignment",'ratio_area',
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
                height: float = bbox[3] - bbox[1]
                bbox_width: float = float(bbox[2] - bbox[0])
                perimeter: float = 2 * (bbox_width + height)
                prev_bbox: Optional[List[float]] = sorted_lines[i-1][1].line_geometry.line_bbox if i > 0 else None
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
                align_values: List[float] = [v for v in [align_prev, align_next] if v is not None]
                var_alignment: float = float(np.var(align_values)) if len(align_values) > 1 else 0.0

                all_geometric_features[line_id] = {
                    "line_area": line_area,
                    "bbox_width": bbox_width,
                    "align_prev": align_prev if align_prev is not None else 0.0,
                    "align_next": align_next if align_next is not None else 0.0,
                    "var_alignment": var_alignment,
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
            var_alignment: float = geometric_features.get("var_alignment", 0.0)
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
                "var_alignment": var_alignment,
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

    def _validate_scanner_interval_all_vs_all(self, encoded_lines: Dict[str, List[int]], all_lines: Dict[str, AllLines], img_dims: Dict[str, int], header_line_id: str, scanned_line_ids: List[str]) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """
        try:
            similarity_threshold = float(self.worker_config.get("similarity_threshold", 0.85))
            min_cluster = int(self.worker_config.get("min_cluster", 2))

            line_ids: List[str] = list(encoded_lines.keys())
            if header_line_id not in line_ids:
                logger.error("Header no encontrado entre las líneas codificadas")
                return []
            header_idx = line_ids.index(header_line_id)
 
            # normalizar scanned_line_ids si vienen como lista de dicts
            if isinstance(scanned_line_ids, list) and scanned_line_ids and isinstance(scanned_line_ids[0], dict):
                scanned_line_ids = [s.get('line_id') or s.get('id') or s.get('line') for s in scanned_line_ids if isinstance(s, dict)]
            # localizar la última línea reportada por el scanner que exista en line_ids
            scanned_indices = [line_ids.index(l) for l in scanned_line_ids if isinstance(l, str) and l in line_ids]
            if not scanned_indices:
                logger.warning("Ninguna de las líneas del scanner se encontró en las líneas codificadas")
                return []
            
            last_scanner_idx = max(scanned_indices)
            
            start_idx = header_idx + 1
            end_idx = last_scanner_idx

            if start_idx > end_idx:
                logger.info("Intervalo para validar vacío (header al final o scanner produjo líneas anteriores al header).")
                return []

            # LOG: Mostrar el intervalo de líneas a validar
            interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
            logger.info(f"Intervalo de validación: líneas {start_idx} a {end_idx} (total: {len(interval_line_ids)} líneas)")
            for i, line_id in enumerate(interval_line_ids):
                line_obj = all_lines.get(line_id)
                line_text = line_obj.text if line_obj else "SIN TEXTO"
                logger.info(f"  [{start_idx + i}] {line_id}: '{line_text}'")

            feature_keys = [
                'count','mean','std_dev','iqr','p50','skewness',
                'line_area','bbox_width','align_prev', 'align_next', "var_alignment",'ratio_area',
                'aspect_ratio','perimeter','xmin_align','xmax_align'
            ]

            all_geometric_features = self._calculate_geometric_features(all_lines, img_dims)
            if not all_geometric_features:
                logger.warning("No se pudieron calcular las características geométricas para la validación.")
                return []

            mat_rows = []
            candidate_indices = []
            candidate_line_ids = []
            for idx in range(start_idx, end_idx + 1):
                lid = line_ids[idx]
                vals = encoded_lines.get(lid, [])
                geom = all_geometric_features.get(lid)
                if not vals or geom is None:
                    logger.debug(f"Línea {lid} en intervalo sin codificación o sin features geométricos; será ignorada.")
                    continue
                analysis = self._analyze_encoded_line(lid, vals, geom)
                if not analysis:
                    logger.debug(f"No se pudo analizar línea {lid}; será ignorada.")
                    continue
                agg = analysis.get('aggregate_stats', {})
                row = [float(agg.get(k, 0.0)) for k in feature_keys]
                mat_rows.append(row)
                candidate_indices.append(idx)
                candidate_line_ids.append(lid)

            n = len(mat_rows)
            if n < min_cluster:
                logger.warning("No hay suficientes líneas válidas en el intervalo para validación coseno.")
                return []

            mat = np.asarray(mat_rows, dtype=np.float32)
            scaler = MinMaxScaler()
            mat_scaled = scaler.fit_transform(mat)

            # matriz todos contra todos (n x n)
            sims_mat = cosine_similarity(mat_scaled)
            
            # Para facilitar la lectura del log y saber a qué línea corresponde cada valor de la matriz,
            # se imprime la matriz junto con los IDs de línea en el mismo orden de las filas/columnas.
            logger.info(
                "Matriz de similitud (cosine_similarity):\nFilas/Columnas (en orden): %s\n%s",
                candidate_line_ids,
                np.array2string(sims_mat, precision=4, suppress_small=True)
            )

            # para cada fila, calcular similitud media con las demás (excluir self)
            mean_sims = []
            for i in range(n):
                if n == 1:
                    mean_sims.append(1.0)
                else:
                    mean_val = (np.sum(sims_mat[i]) - 1.0) / (n - 1)
                    mean_sims.append(float(mean_val))

            matched_original_indices: List[int] = []
            consecutive_failures = 0
            # iterar en el orden original del intervalo
            for mean_sim, orig_idx, lid in zip(mean_sims, candidate_indices, candidate_line_ids):
                if mean_sim >= similarity_threshold:
                    matched_original_indices.append(orig_idx)
                    consecutive_failures = 0
                    logger.debug(f"Línea {lid} idx={orig_idx}: mean_sim={mean_sim:.4f} ✓")

            if not matched_original_indices:
                logger.info("Validación all-vs-all no encontró líneas consistentes en el intervalo.")
                return []

            # incluir header (aunque no participó en la similitud)
            final_indices = [header_idx] + matched_original_indices
            table_line_ids = [line_ids[i] for i in final_indices if i < len(line_ids)]
            return table_line_ids
        except Exception as e:
            logger.error(f"Error validando intervalo del scanner (all-vs-all): {e}", exc_info=True)
            return []