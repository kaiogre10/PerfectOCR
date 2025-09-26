# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import math
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines, Polygons
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, normalize # type: ignore

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
                tabular_lines: List[str] = manager.get_tabular_lines()
                # normalizar: si manager devuelve lista de dicts extraer ids
                if isinstance(tabular_lines, list) and tabular_lines:
                    if isinstance(tabular_lines[0], dict):
                        tabular_lines = [t.get('line_id') or t.get('id') or t.get('line') for t in tabular_lines if isinstance(t, dict)]
                encoded_lines = manager.get_encode_lines()

                # Si hay resultado del scanner, VALIDAR usando estrategia all-vs-all en el intervalo header+1 .. última del scanner
                if tabular_lines:
                    logger.info(f"Validando resultado del scanner con validación coseno all-vs-all ({len(tabular_lines)} líneas reportadas)")
                    validated = self._validate_scanner_interval_all_vs_all(encoded_lines, all_lines, img_dims, header_line_id, tabular_lines, manager)
                    if validated:
                        total_time = time.time() - start_time
                        logger.info(f"Validación coseno completada en {total_time:.6f}s. Líneas válidas: {len(validated)}")
                        success = manager.save_tabular_lines(validated)
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
                logger.info("Enviando Lineas a DBSCAN")
                return True
        except Exception as e:
            logger.info(f"Error en matriz de similitud coseno: {e}", exc_info=True)
            return False
            
    def _calculate_line_featrues(self, all_lines: Dict[str, AllLines],  manager: DataFormatter) -> Optional[Dict[str, float]]:
        try:
            if not manager.workflow.all_lines if manager.workflow else {}:
                return None
            
            sorted_lines: List[Tuple[str, AllLines]] = sorted(
                all_lines.items(),
                key=lambda kv: kv[1].line_geometry.line_centroid[1]
            )
    
            line_features: Dict[str, float] = {}
            # Cálculo global de numerics (promedio y máximo)
            numeric_counts_by_line: Dict[str, float] = {}
            for i, (line_id, line_data) in enumerate(sorted_lines):
                ncount = 0.0
                poly_ids_line = getattr(line_data, "polygon_ids", []) or []
                if manager.workflow and manager.workflow.polygons and poly_ids_line:
                    polygons_dict = manager.workflow.polygons
                    for pid in poly_ids_line:
                        if pid in polygons_dict and getattr(polygons_dict[pid], "semantic_type", "") == "numeric":
                            ncount += 1.0
                numeric_counts_by_line[line_id] = ncount

            if numeric_counts_by_line:
                total_numerics_global = float(sum(numeric_counts_by_line.values()))
                total_lines_global = float(len(numeric_counts_by_line))
                numeric_mean_global = total_numerics_global / total_lines_global if total_lines_global > 0 else 0.0
                max_numeric_line_id = max(numeric_counts_by_line, key=numeric_counts_by_line.get)
                max_numeric_count_global = float(numeric_counts_by_line[max_numeric_line_id])
                
                line_features = {
                    "total_numerics_global": total_numerics_global,
                    "numeric_mean_global": numeric_mean_global,
                    "max_numeric_count_global": max_numeric_count_global,
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
                return None

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
                return None

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
                return None # O un valor por defecto apropiado
                
            max_numeric_count = line_features.get("max_numeric_count_global")
            if max_numeric_count is None:
                return None # O un valor por defecto apropiado

            numeric_mean = line_features.get("numeric_mean_global")
            if numeric_mean is None:
                return None # O un valor por defecto apropiado
            
            line_text = getattr(line_data, "text", "") or ""
            
            numeric_count_norm = numeric_count / max_numeric_count if max_numeric_count > 0 else 0.0
            numeric_frec_rel: float = max_numeric_count - numeric_count 
            numeric_ratio_frec: float = numeric_count_norm + numeric_frec_rel
            num_above: float = 1.0 if numeric_count > numeric_mean else 0.0
            # Normaliza num_margin en el intervalo [-1, 1] usando el promedio global como base 0.
            if max_numeric_count > 0:
                num_margin = (numeric_count - numeric_mean) / (max_numeric_count / 2)
                # Limita el valor al rango [-1, 1]
                num_margin = max(-1.0, min(1.0, num_margin))
            else:
                num_margin = 0.0
            
            digit_char_count: float = float(sum(ch.isdigit() for ch in line_text))
            
            bbox: List[float] = line_data.line_geometry.line_bbox
            centroid: List[float] = line_data.line_geometry.line_centroid

            if len(bbox) < 4 or len(centroid) < 2:
                return None

            # Proporción del área respecto del total
            if not img_dims or "size" not in img_dims:
                return None
            
            total_size = img_dims.get("size")
            if not total_size:
                return None
            
            line_area: float = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            bbox_width: float = float(bbox[2] - bbox[0])
            ratio_area: float = line_area / float(total_size) 
            aspect_ratio = ((bbox[2] - bbox[0]) / (bbox[3] - bbox[1])) / 100 if (bbox[3] - bbox[1]) > 0 else 0
            height: float = bbox[3] - bbox[1]
            perimeter: float = 2 * (bbox_width + height)
            diagonal = float(np.sqrt((bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2))
            compact = ((perimeter**2) / line_area) / 100 if line_area > 0 else 0
            
            prev_bbox: Optional[List[float]] = sorted_lines[current_index-1][1].line_geometry.line_bbox if current_index > 0 else None
            next_bbox: Optional[List[float]] = sorted_lines[current_index+1][1].line_geometry.line_bbox if current_index < len(sorted_lines) - 1 else None
            prev_centroid: Optional[List[float]] = sorted_lines[current_index-1][1].line_geometry.line_centroid if current_index > 0 else None
            next_centroid: Optional[List[float]] = sorted_lines[current_index+1][1].line_geometry.line_centroid if current_index < len(sorted_lines) - 1 else None
            
            # función auxiliar para similitud coseno con eje X
            def alignment(ref_c: List[float], other_c: Optional[List[float]]) -> Optional[float]:
                if other_c is None: 
                    return 1.0
                vec = np.array([other_c[0] - ref_c[0], other_c[1] - ref_c[1]])
                axis = np.array([1, 0])  # eje X
                if np.linalg.norm(vec) == 0: 
                    return 1.0
                return float(np.dot(vec, axis) / (np.linalg.norm(vec) * np.linalg.norm(axis)))

            def bbox_alignment(current_coord: float, other_bbox: Optional[List[float]], coord_idx: int) -> float:
                """
                Mide alineación usando similitud coseno.
                Punto de referencia: [current_coord, 0] en el eje X
                Vector hacia otra línea: [other_coord - current_coord, other_y - 0]
                """
                if other_bbox is None:
                    return 1.0
                
                # Punto de referencia en el eje X
                ref_point = np.array([current_coord, 0])
                
                # Coordenada de la otra línea
                other_coord = other_bbox[coord_idx]
                other_y = other_bbox[1]  # Coordenada Y de la otra línea
                
                # Vector desde el punto de referencia hacia la otra línea
                vec_to_other = np.array([other_coord - current_coord, other_y - ref_point[1]])
                
                # Vector de referencia (eje X positivo)
                ref_vec = np.array([1, 0])
                
                # Similitud coseno
                if np.linalg.norm(vec_to_other) == 0:
                    return 1.0
                
                cosine_sim = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec))
                
                return cosine_sim

            # Alineación ortogonal para xmin y xmax con prev y next
            current_xmin = bbox[0]
            current_xmax = bbox[2]
            
            prev_xmin_align: Optional[float] = bbox_alignment(current_xmin, prev_bbox, 0) if prev_bbox else 1.0
            prev_xmax_align: Optional[float] = bbox_alignment(current_xmax, prev_bbox, 2) if prev_bbox else 1.0
            next_xmin_align: Optional[float] = bbox_alignment(current_xmin, next_bbox, 0) if next_bbox else 1.0
            next_xmax_align: Optional[float] = bbox_alignment(current_xmax, next_bbox, 2) if next_bbox else 1.0

            align_prev: Optional[float] = alignment(centroid, prev_centroid)
            align_next: Optional[float] = alignment(centroid, next_centroid)
            
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
            all_features: Dict[str, float] = {
                'count': count,
                'mean': mean,
                'std_dev': std_dev,
                'p25': p25,
                'p50': p50,
                'p75': p75,
                'iqr': iqr,
                'skewness': skewness,
                "numeric_count_norm": numeric_count_norm,
                "numeric_frec_rel": numeric_frec_rel,
                "numeric_ratio_frec": numeric_ratio_frec,
                "num_above": num_above,
                "num_margin": num_margin,
                "digit_char_count": digit_char_count,
                "line_area": line_area,
                "bbox_width": bbox_width,
                "ratio_area": ratio_area,
                "aspect_ratio": aspect_ratio,
                "height": height,
                "perimeter": perimeter,
                "diagonal": diagonal,
                "compact": compact,
                "prev_xmin_align": prev_xmin_align,
                "prev_xmax_align": prev_xmax_align,
                "next_xmin_align": next_xmin_align,
                "next_xmax_align": next_xmax_align,
                "align_prev": align_prev if align_prev is not None else 1.0,
                "align_next": align_next if align_next is not None else 1.0,
            }
            return {"aggregate_stats": all_features}

        except Exception as e:
            logger.error(f"Error calculando tabular features: {e}", exc_info=True)
            return None
        
    def _validate_scanner_interval_all_vs_all(self, encoded_lines: Dict[str, List[int]], all_lines: Dict[str, AllLines], img_dims: Dict[str, int], header_line_id: str, tabular_lines: List[str], manager: DataFormatter) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """
        try:
            similarity_threshold = float(self.worker_config.get("similarity_threshold", 0.85))
            min_cluster = int(self.worker_config.get("min_cluster", {}))

            line_ids: List[str] = list(encoded_lines.keys())
            if header_line_id not in line_ids:
                logger.error("Header no encontrado entre las líneas codificadas")
                return []
            header_idx = line_ids.index(header_line_id)

            # normalizar scanned_line_ids si vienen como lista de dicts
            if isinstance(tabular_lines, list) and tabular_lines and isinstance(tabular_lines[0], dict):
                tabular_lines = [s.get('line_id') or s.get('id') or s.get('line') for s in tabular_lines if isinstance(s, dict)]
            # localizar la última línea reportada por el scanner que exista en line_ids
            scanned_indices = [line_ids.index(l) for l in tabular_lines if isinstance(l, str) and l in line_ids]
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
                logger.info(f"[{start_idx + i}] {line_id}: '{line_text}'")

            # BLOQUEAR líneas que contienen key_field
            # MontoTotalDocumento, Subtotal, TotalProductos, MontoIVAGeneral, RFCProveedor, FolioDocumento, FechaDocumento, NombreCliente
     
            blocked_line_ids: Set[str] = set()
            if manager.workflow and manager.workflow.polygons:
                logger.debug(f"Total de polígonos disponibles: {len(manager.workflow.polygons)}")
                
                # Obtener IDs de polígonos con key_field (incluyendo HeaderWords)
                blocked_polygon_ids: Set[str] = set()
                for poly_id, polygon in manager.workflow.polygons.items():
                    if polygon.key_field:
                        blocked_polygon_ids.add(poly_id)
                        logger.debug(f"Polígono {poly_id}: key_field='{polygon.key_field}'")
                
                # Buscar líneas que contengan estos polígonos
                for line_id, line_obj in all_lines.items():
                    if line_obj.polygon_ids:
                        line_polygon_ids = set(line_obj.polygon_ids)
                        if line_polygon_ids.intersection(blocked_polygon_ids):
                            blocked_line_ids.add(line_id)
                            logger.info(f"Línea {line_id} bloqueada por contener polígonos con key_field")
            
            logger.info(f"Líneas bloqueadas por key_field: {sorted(blocked_line_ids)}")
            
            # Filtrar el intervalo excluyendo líneas bloqueadas
            blocked_in_interval = [idx for idx in range(start_idx, end_idx + 1) if line_ids[idx] in blocked_line_ids]
            if blocked_in_interval:
                first_blocked = min(blocked_in_interval)
                new_end = first_blocked - 1
                logger.info(f"Cortando intervalo en la primera línea bloqueada: bloqueada_idx={first_blocked}, nuevo end_idx={new_end}")
                end_idx = new_end

            # reconstruir intervalo y comprobar si quedó vacío tras el corte
            if start_idx > end_idx:
                logger.info("Intervalo quedó vacío tras cortar por líneas bloqueadas.")
                return []

            filtered_interval_indices: List[int] = [i for i in range(start_idx, end_idx + 1)]
            logger.info(f"Intervalo filtrado: {len(filtered_interval_indices)} líneas (de {end_idx - start_idx + 1} originales)")

            feature_keys: List[str] = [
                'count',
                'mean',
                'std_dev',
                'p25', 
                'p50',
                'p75',
                'iqr',
                'skewness',
                'numeric_count_norm',
                "numeric_frec_rel",
                "numeric_ratio_frec",
                "num_above",
                "num_margin",
                "digit_char_count",
                'line_area',
                'bbox_width',
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
                
            mat_rows: List[List[float]] = []
            candidate_indices: List[str] = []
            candidate_line_ids: List[str] = []
            try:
                for idx in filtered_interval_indices:
                    lid = line_ids[idx]
                    line_values = encoded_lines.get(lid, [])
                    if not line_values:
                        logger.warning(f"Línea {lid} en intervalo sin codificación o sin features geométricos; será ignorada.")
                        continue

                    line_data = all_lines.get(lid)
                    if not line_data:
                        logger.warning(f"No se encontraron datos para la línea {lid}; será ignorada.")
                        continue

                    try:
                        analysis = self._calculate_features(lid, line_data, all_lines, img_dims, manager, line_features, line_values)
                        
                        agg = analysis.get('aggregate_stats', {})
                        logger.debug(f"Features por línea (aggregate_stats): {len(analysis.get('aggregate_stats', {}))}, {agg}")

                    except Exception as e:
                        logger.info(f"Error en similitud: {e}", exc_info=True)

            except Exception as e:
                logger.info(f"Error en similitud: {e}", exc_info=True)
                
                row: List[float] = [float(agg.get(k, 0.0)) for k in feature_keys]
                mat_rows.append(row)
                candidate_indices.append(idx)
                candidate_line_ids.append(lid)

            n = len(mat_rows)
            if n < min_cluster:
                logger.warning("No hay suficientes líneas válidas en el intervalo para validación coseno.")
                # Tomar como línea tabular el intervalo entre el encabezado y la línea bloqueada inmediata
                logger.info("Tomando intervalo entre encabezado y línea bloqueada inmediata como líneas tabulares.")
                interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
                return interval_line_ids

            mat = np.asarray(mat_rows, dtype=np.float64)
            mat_scaled = self._normalize_features_for_cosine_similarity(mat, feature_keys)

            X = np.array(mat_scaled)
            logger.info(f"{X}")

            # matriz todos contra todos (n x n)
            sims_mat: np.ndarray[Any, np.dtype[np.float64]] = cosine_similarity(mat_scaled, dense_output=True)
            # Para mejorar la legibilidad, separamos filas y columnas y mostramos la matriz línea por línea
            mean = np.array(sims_mat)
            mean_log = np.mean(mean)
            logger.info(f"Promedio matriz: {mean_log}")
            logger.debug("Matriz de similitud (cosine_similarity):")
            logger.debug("Filas/Columnas (en orden): %s", ", ".join(str(lid) for lid in candidate_line_ids))
            matriz_str = "\n".join(
                ["[" + "  ".join(f"{val:7.6f}" for val in row) + "]" for row in sims_mat]
            )
            logger.info("Matriz:\n%s", matriz_str)

            # para cada fila, calcular similitud media con las demás (excluir self)
            mean_sims: List[float] = []
            for i in range(n):
                if n == 1:
                    mean_sims.append(1.0)
                else:
                    mean_val = float((np.sum(sims_mat[i]) - 1.0) / (n - 1))
                    mean_sims.append(mean_val)

            matched_original_indices: List[int] = []
            consecutive_failures = 0
            for mean_sim, orig_idx, lid in zip(mean_sims, candidate_indices, candidate_line_ids):
                if mean_sim >= similarity_threshold:
                    matched_original_indices.append(int(orig_idx))
                    consecutive_failures +=1
                logger.info(f"Línea {lid} idx={orig_idx}: mean_sim={mean_sim:.4f}")

            # incluir header (aunque no participó en la similitud)
            final_indices = [header_idx] + matched_original_indices
            table_line_ids = [line_ids[i] for i in final_indices if i < len(line_ids)]
            return table_line_ids
        except Exception as e:
            logger.error(f"Error validando intervalo del scanner (all-vs-all): {e}", exc_info=True)
            return []

            
    def _normalize_features_for_cosine_similarity(self, mat: np.ndarray[Any, np.dtype[np.float64]], feature_keys: List[str]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Normalización específica para similitud coseno que agrupa features similares.
        """
        mat_normalized = mat.copy()
        
        # 1. FEATURES ESTADÍSTICOS - Normalizar por z-score para mantener distribución
        statistical_features = ['count', 'mean', 'std_dev', 'p25', 'p50', 'p75', 'iqr', 'skewness']
        for i, feature in enumerate(feature_keys):
            if feature in statistical_features:
                scaler = MinMaxScaler()
                mat_normalized[:, i] = scaler.fit_transform(mat[:, i:i+1]).flatten()
        
        # 2. FEATURES GEOMÉTRICOS - Normalizar por percentiles para reducir outliers
        geometric_features = ['line_area', 'bbox_width', 'height', 'perimeter', 'diagonal']
        for i, feature in enumerate(feature_keys):
            if feature in geometric_features:
                # Usar percentiles 5-95 para reducir impacto de outliers
                p5 = np.percentile(mat[:, i], 5)
                p95 = np.percentile(mat[:, i], 95)
                if p95 > p5:
                    mat_normalized[:, i] = np.clip((mat[:, i] - p5) / (p95 - p5), 0, 1)
                else:
                    mat_normalized[:, i] = 0.5  # valor neutro si no hay variación
        
        # 3. FEATURES DE RATIO - MinMax normalización
        ratio_features = ['ratio_area', 'aspect_ratio', 'compact']
        for i, feature in enumerate(feature_keys):
            if feature in ratio_features:
                scaler = MinMaxScaler()
                mat_normalized[:, i] = scaler.fit_transform(mat[:, i:i+1]).flatten()
        
        # 4. FEATURES DE ALINEACIÓN - Mantener valores originales (ya están en [-1,1])
        alignment_features = ['align_prev', 'align_next', 'prev_xmin_align', 'prev_xmax_align', 
                            'next_xmin_align', 'next_xmax_align']
        # No normalizar, ya están en rango apropiado
        
        # 5. FEATURES CONSTANTES - Eliminar o reemplazar por variación
        constant_features = ['numeric_count_norm', 'numeric_frec_rel', 'numeric_ratio_frec', 
                            'num_above', 'num_margin']
        for i, feature in enumerate(feature_keys):
            if feature in constant_features:
                # Si todos los valores son iguales, usar variación pequeña aleatoria
                if np.std(mat[:, i]) < 1e-6:
                    mat_normalized[:, i] = np.random.normal(0.5, 0.1, len(mat[:, i]))
                else:
                    scaler = MinMaxScaler()
                    mat_normalized[:, i] = scaler.fit_transform(mat[:, i:i+1]).flatten()
        
        # 6. FEATURES DE CARACTERES - Log normalización para reducir impacto de valores altos
        char_features = ['digit_char_count']
        for i, feature in enumerate(feature_keys):
            if feature in char_features:
                # Aplicar log(1 + x) para suavizar valores altos
                log_values = np.log1p(mat[:, i])
                scaler = MinMaxScaler()
                mat_normalized[:, i] = scaler.fit_transform(log_values.reshape(-1, 1)).flatten()
        
        return mat_normalized

    def _find_header_line_id(self, polygons: Dict[str, Polygons], all_lines: Dict[str, AllLines]) -> Optional[str]:
        """Localiza la line_id del encabezado basada en HeaderWords."""
        try:
            hdr_poly_ids: List[str] = [pid for pid, p in polygons.items() if getattr(p, "key_field", None) == "HeaderWords"]
            if not hdr_poly_ids: 
                return None
            
            hdr_set = set(hdr_poly_ids)
            counts = {lid: len(set(lobj.polygon_ids).intersection(hdr_set)) for lid, lobj in all_lines.items() if lobj.polygon_ids}
            
            if not counts: 
                return None
        
            header_line_id: Optional[str] = max(counts, key=counts.get)
        
            if not header_line_id:
                return None
            else:
                logger.info(f"Header_line_id={header_line_id} (via HeaderWords)")
                return header_line_id
        
        except Exception as e:
            logger.error(f"No hubo encabezado textual por similitud de encabezado: {e}", exc_info=True)
