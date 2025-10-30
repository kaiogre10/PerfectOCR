# PerfectOCR/core/workflow/vectorial_transformation/matricial_cosine.py
import numpy as np
import time
import logging
from typing import Dict, Any, List, Set, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import AllLines
from scipy.sparse import csr_matrix # type: ignore

logger = logging.getLogger(__name__)

class MatricialCusine(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('cos_sim', {})
        self.enabled_outputs = config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_lines", False)
                
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            analysis: Dict[str, Dict[str, float]] = context.get("all_features", {})
            if not analysis:
                logger.warning("No hay features disponibles para procesar por que ya se detectaron lineas tabulares")
                return True            
            table_line_ids: List[str] = self._compare_vectors(manager, analysis)
            if table_line_ids:
                succes = manager.save_tabular_lines(table_line_ids)
                if succes:
                    
                    logger.info("Tablas guaradas en el manager desde coseno")
                    if self.output:
                        from services.output_service import save_debug_json
                        return_objects: bool = True
                        tab_info: Dict[str, Any] = manager.get_tabular_lines(return_objects) # type: ignore
                        file_name: str = manager.workflow.metadata.image_name # type: ignore
                        worker_name = context.get("worker_name", {})
                        output_paths = context.get("output_paths", [])
                        save_debug_json(output_paths, worker_name, tab_info, file_name)
    
                    return True
        except Exception as e:
            logger.error(f"Error en matriz coseno: {e}", exc_info=True)
        return True

    def _compare_vectors(self, manager: DataFormatter, analysis: Dict[str, Dict[str, float]]) -> List[str]:
        
        try:
            start_time: float = time.time()
            logger.debug("Calculando matriz de similitud")            
            first_line_features = next(iter(analysis.values()))
            feature_keys: List[str] = list(first_line_features.keys())
            logger.debug(f"Features detectados: {feature_keys}")
            
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            line_ids: List[str] = list(all_lines.keys())
            header_line_id = [lid for lid, l in all_lines.items() if getattr(l, "header_line", not None)]
            header_line_id = header_line_id[0] if header_line_id else None

            if header_line_id is None:
                table_line_ids = self._emergency_fallback(analysis, line_ids, feature_keys, all_lines, manager)
                total_time = time.time() - start_time
                logger.debug(f"Validación coseno completada en {total_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                return table_line_ids
                
            else:
                return_objects: bool = False
                tabular_lines: List[str]= manager.get_tabular_lines(return_objects) #type: ignore
                logger.debug(f"TABULAR LINES: {tabular_lines}")
                header_idx = line_ids.index(header_line_id)
                                
                if tabular_lines:
                    logger.debug(f"Validando resultado del scanner con validación coseno all-vs-all ({len(tabular_lines)} líneas reportadas)")
                    table_line_ids = self._validate_scanner_interval_all_vs_all(analysis, tabular_lines, manager, header_line_id, line_ids, header_idx, feature_keys, all_lines)
                    if table_line_ids:
                        total_time = time.time() - start_time
                        logger.debug(f"Validación coseno completada en {total_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                        return table_line_ids
                else:
                    logger.warning("Ejecutando fallback: buscando líneas tabulares por similitud coseno con el encabezado")
                    table_line_ids = self._fallback_cosine(analysis, header_line_id, line_ids, header_idx, feature_keys)
                    if table_line_ids:
                        total_time = time.time() - start_time
                        logger.debug(f"Fallback coseno completado en {total_time:.6f}s. Líneas válidas: {len(table_line_ids)}: {table_line_ids}")
                        return table_line_ids
                    else:
                        table_line_ids = self._emergency_fallback(analysis, line_ids, feature_keys, all_lines, manager)
                        logger.warning("Método fallback falló, pasando al método de emergencia")
                        return table_line_ids
        except Exception as e:
            logger.error(f"Error en matriz de similitud coseno: {e}", exc_info=True)
        return []

    def _validate_scanner_interval_all_vs_all(self, analysis: Dict[str, Dict[str, float]], tabular_lines: List[str], manager: DataFormatter, header_line_id: str, line_ids: List[str], header_idx: int, feature_keys: List[str], all_lines: Dict[str, AllLines]) -> List[str]:
        """
        Validación all-vs-all por similitud coseno sobre el intervalo de líneas reportado por el scanner.
        No usa el header como referencia para el intervalo; el header sólo se añade si el intervalo es válido.
        """            
        similarity_threshold: float = self.worker_config.get("similarity_threshold")
        min_cluster = int(self.worker_config.get("min_cluster"))
        from core.utils.fun_cosine_similarity import cosine_similarity_global
        
        if line_ids.index(tabular_lines[0]) < line_ids.index(header_line_id):
            return []
            
        # Convertir tabular_lines (IDs de línea) a índices numéricos
        tabular_indices: List[int] = []
        for line_id in tabular_lines:
            if line_id in line_ids:
                tabular_indices.append(line_ids.index(line_id))
        
        if not tabular_indices:
            logger.error("Ninguna línea tabular encontrada en line_ids")
            return self._emergency_fallback(analysis, line_ids, feature_keys, all_lines, manager)
            
        last_scanner_idx = max(tabular_indices)

        start_idx = header_idx + 1
        end_idx = last_scanner_idx

        if start_idx > end_idx:
            logger.error("Intervalo para validar vacío (header al final o scanner produjo líneas anteriores al header).")
            return self._emergency_fallback(analysis, line_ids, feature_keys, all_lines, manager)

        # LOG: Mostrar el intervalo de líneas a validar
        interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
        logger.debug(f"Intervalo de validación: líneas {start_idx} a {end_idx} (total: {len(interval_line_ids)} líneas)")
        for i, line_id in enumerate(interval_line_ids):
            line_obj = all_lines.get(line_id)
            line_text = line_obj.text if line_obj else "SIN TEXTO"
            logger.debug(f"[{start_idx + i}] {line_id}: '{line_text}'")
        
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
                        logger.debug(f"Línea {line_id} bloqueada por contener polígonos con key_field")
            
        logger.debug(f"Líneas bloqueadas por key_field: {sorted(blocked_line_ids)}")
        
        # Filtrar el intervalo excluyendo líneas bloqueadas
        blocked_in_interval = [idx for idx in range(start_idx, end_idx + 1) if line_ids[idx] in blocked_line_ids]
        if blocked_in_interval:
            first_blocked = min(blocked_in_interval)
            new_end = first_blocked - 1
            logger.debug(f"Cortando intervalo en la primera línea bloqueada: bloqueada_idx={first_blocked}, nuevo end_idx={new_end}")
            end_idx = new_end

        # reconstruir intervalo y comprobar si quedó vacío tras el corte
        if start_idx > end_idx:
            logger.debug("Intervalo quedó vacío tras cortar por líneas bloqueadas.")
            return self._emergency_fallback(analysis, line_ids, feature_keys, all_lines, manager)

        filtered_interval_indices: List[int] = [i for i in range(start_idx, end_idx + 1)]
        logger.debug(f"Intervalo filtrado: {len(filtered_interval_indices)} líneas (de {end_idx - start_idx + 1} originales)")
                
        mat_rows: List[List[float]] = []
        candidate_indices: List[int] = []  
        candidate_line_ids: List[str] = []

        for idx in filtered_interval_indices:  
            line_id = line_ids[idx] 
            if not analysis:
                logger.error("Error extrayendo features")
            features = analysis.get(line_id, {}) 
            
            if not features:
                logger.warning(f"No se encontraron features para línea {line_id}")
                continue
                
            row: List[float] = [float(features.get(k, 0.0)) for k in feature_keys]
            mat_rows.append(row)
            candidate_indices.append(idx)
            candidate_line_ids.append(line_id)

        n = len(mat_rows)
        if n < min_cluster:
            logger.warning("No hay suficientes líneas válidas en el intervalo para validación coseno.")
            logger.debug("Tomando intervalo entre encabezado y línea bloqueada inmediata como líneas tabulares.")
            interval_line_ids = [line_ids[i] for i in range(start_idx, end_idx + 1)]
            return interval_line_ids
            
        try:
            X = csr_matrix(mat_rows, dtype=np.float32)

            timecos0 = time.perf_counter()

            sims_mat = cosine_similarity_global(X, dense_output=False).astype(np.float32)
            logger.debug(f"Coseno realizado en: {time.perf_counter()-timecos0:.10f}s")
            
        except Exception as e:
            logger.error(f"Error calculando matriz se similitud: {e}", exc_info=True)

        # Convertir la matriz dispersa a densa para mostrarla
        sims_mat_dense= sims_mat.toarray() #type: ignore
        mean_log = np.mean(sims_mat_dense) # type: ignore
        logger.info(f"Promedio matriz: {mean_log}")
        logger.debug("Filas/Columnas (en orden): %s", ", ".join(str(lid) for lid in candidate_line_ids))
        matriz_str = "\n".join(
            ["[" + "  ".join(f"{val:7.6f}" for val in row) + "]" for row in sims_mat_dense]
        )
        logger.info("Matriz:\n%s", matriz_str)

        # para cada fila, calcular similitud media con las demás (excluir self)
        mean_sims: List[float] = []
        for i in range(n):
            if n == 1:
                mean_sims.append(1.0)
            else:
                mean_val = float((np.sum(sims_mat[i]) - 1.0) / (n - 1)) # type: ignore
                mean_sims.append(mean_val)

        matched_original_indices: List[int] = []
        consecutive_failures = 0
        for mean_sim, orig_idx, lid in zip(mean_sims, candidate_indices, candidate_line_ids):
            if mean_sim > similarity_threshold:
                matched_original_indices.append(int(orig_idx))
                consecutive_failures += 1
            logger.info(f"Línea {lid} idx={orig_idx}: mean_sim={mean_sim:.6f}")

        # Si hay validaciones por coseno, devolver todo el intervalo hasta la última validada
        if matched_original_indices:
            last_valid_idx = max(matched_original_indices)
            final_end_idx = min(last_valid_idx, end_idx)  # respeta corte por key_field
            table_line_ids = [line_ids[i] for i in range(start_idx, final_end_idx + 1)]
            logger.debug(f"Intervalo asignado por coseno hasta último validado (idx={final_end_idx}): {len(table_line_ids)} líneas")
            return table_line_ids

        # Si ninguna línea superó el umbral, activar emergencia desde aquí
        logger.info("Ninguna línea validada por coseno en el intervalo; activando emergencia.")
        return self._emergency_fallback(analysis, line_ids, feature_keys, all_lines, manager)

    def _fallback_cosine(self, analysis: Dict[str, Dict[str, float]], header_line_id: str, line_ids: List[str], header_idx: int, feature_keys: List[str]) -> List[str]:
        """
        Fallback: Busca un bloque continuo de líneas tabulares después del encabezado.
        Compara cada línea con la línea de referencia (la primera después del encabezado).
        Tolera un número de fallos consecutivos ('interval') antes de cortar el bloque.
        """
        logger.warning("INICIANDO MÉTODO FALLBACK")
        similarity_threshold: float = float(self.worker_config.get("similarity_threshold"))
        interval_margin: int = int(self.worker_config.get("interval"))
        # La línea de referencia es la que está justo después del encabezado
        ref_line_idx = header_idx + 1
        if ref_line_idx >= len(line_ids):
            logger.warning("No hay líneas después del encabezado para usar como referencia.")
            return []

        # Buscar la primera línea después del encabezado que sí tenga features, si no la encuentra, usa el header igual
        ref_line_id = None
        ref_features = {}
        search_idx = ref_line_idx
        while search_idx < len(line_ids):
            candidate_id = line_ids[search_idx]
            candidate_features = analysis.get(candidate_id, {})
            if candidate_features:
                ref_line_id = candidate_id
                ref_features = candidate_features
                break
            else:
                logger.warning(f"No se encontraron features para la línea de referencia {candidate_id}, se prueba la siguiente.")
            search_idx += 1
        if ref_line_id is None:
            ref_line_id = header_line_id
            ref_features = analysis.get(header_line_id, {})
            logger.warning("Ninguna línea siguiente al encabezado tiene features. Usando el header como referencia.")
            return []

        ref_vec = np.array([float(ref_features.get(k, 0.0)) for k in feature_keys]).reshape(1, -1)

        # Preparar datos para cálculo de similitud en bloque
        candidate_rows: List[List[float]]= []
        candidate_line_ids: List[str] = []
        for idx in range(header_idx + 2, len(line_ids)):
            line_id = line_ids[idx]
            features = analysis.get(line_id, {})
            if not features:
                logger.warning(f"Sin features para línea {line_id}, se omitirá en fallback.")
                continue
            
            row: List[float] = [float(features.get(k, 0.0)) for k in feature_keys]
            candidate_rows.append(row)
            candidate_line_ids.append(line_id)

        if not candidate_rows:
            logger.warning("No hay líneas candidatas para fallback después de la línea de referencia.")
            return []

        # Calcular similitud y registrar la matriz
        from core.utils.fun_cosine_similarity import calculate_similarity_ref

        X = csr_matrix(candidate_rows, dtype=np.float32)
        sims = calculate_similarity_ref(X, ref_vec).astype(np.float32)
        # sims = cosine_similarity(ref_vec, X, dense_output=False)[0]

        logger.info(f"Promedio de similitud con línea de referencia '{ref_line_id}': {np.mean(sims):.6f}")
        logger.info("Candidatas (en orden): %s", ", ".join(str(lid) for lid in candidate_line_ids))
        sims_str = "[" + "  ".join(f"{val:7.6f}" for val in sims) + "]"
        logger.info("Similitudes:\n%s", sims_str)

        last_success_idx = ref_line_idx
        consecutive_failures = 0

        # Iterar sobre los resultados de similitud
        for i, sim in enumerate(sims):
            current_idx = line_ids.index(candidate_line_ids[i])
            logger.info(f"ref {ref_line_id}: línea {candidate_line_ids[i]}, sim={sim:.6f}")

            if sim > similarity_threshold:
                consecutive_failures = 0
                last_success_idx = current_idx
            else:
                consecutive_failures += 1

            if consecutive_failures > interval_margin:
                logger.info(f"Se superó el margen de error ({interval_margin} fallos). Cortando en índice {last_success_idx}.")
                break
        
        # Construir el resultado final como un bloque continuo hasta el último éxito
        final_tabular_lines = [line_ids[i] for i in range(ref_line_idx, last_success_idx + 1)]

        logger.debug(f"Fallback encontró {len(final_tabular_lines)} líneas tabulares continuas.")
        return final_tabular_lines

    def _emergency_fallback(self, analysis: Dict[str, Dict[str, float]], line_ids: List[str], feature_keys: List[str], all_lines: Dict[str, AllLines], manager: DataFormatter) -> List[str]:
        """
        Fallback de emergencia optimizado. Compara todas las líneas del documento contra vectores DUMMIE
        usando una similitud ponderada para encontrar el mejor cluster de líneas tabulares.
        """
        logger.info(f"INICIANDO MÉTODO DE EMERGENCA")
        similarity_threshold: float = self.worker_config.get("similarity_threshold")
        min_cluster: int = int(self.worker_config.get("min_cluster"))
        interval_margin: int = int(self.worker_config.get("interval"))
        dummie_weights: Tuple[float, float] = self.worker_config.get("dummie_weights", [])
        emergency_threshold = self.worker_config.get("emergency_threshold")
        mean_w, median_w = dummie_weights
        
        from core.utils.fun_cosine_similarity import calculate_similarity_ref

        all_lines_indices = all_lines.keys()

        # 1. Preparación de datos
        mat_rows: List[List[float]] = []
        all_line_ids: List[str] = []
        for line_id in all_lines_indices:
            features = analysis.get(line_id)
            if not features:
                continue
            
            row: List[float] = [float(features.get(k, 0.0)) for k in feature_keys]
            mat_rows.append(row)
            all_line_ids.append(line_id)

        n = len(mat_rows)
            
        X = csr_matrix(mat_rows, dtype=np.float32)
        amout = features = len(feature_keys)
        # 2. Construcción de vectores Dummie a partir de diccionarios
        median_dummie_dict = manager.get_median_dummie()
        median_dummie_list = [median_dummie_dict.get(k, 0.0) for k in feature_keys]
        amount_median_dummie = len(median_dummie_dict)
        
        if not amount_median_dummie == amout:
            logger.warning(f"Diferente numero de features para mediana: '{amount_median_dummie}'/{amout}")
            return []
        
        median_ref_vec = np.array(median_dummie_list, dtype=np.float32).reshape(1, -1)
        sims_median = calculate_similarity_ref(X, median_ref_vec, dense_output=False).flatten()
        logger.debug(f"Similitudes con Dummie MEDIAN: {sims_median}")

        mean_dummie_dict = manager.get_mean_dummie()
        mean_dummie_list = [mean_dummie_dict.get(k, 0.0) for k in feature_keys]
        amount_median_dummie = len(mean_dummie_dict)
        
        if not amount_median_dummie == amout:
            logger.warning(f"Diferente numero de features para media: '{amount_median_dummie}'/{amout}")
            return []
            
        mean_ref_vec = np.array(mean_dummie_list, dtype=np.float32).reshape(1, -1)
        sims_mean = calculate_similarity_ref(X, mean_ref_vec, dense_output=False).flatten()
        logger.debug(f"Similitudes con Dummie MEAN: {sims_mean}")

        # 3. Ponderación de resultados
        sims_final = (sims_median * median_w) + (sims_mean * mean_w)
        
        logger.info(f"Promedio de similitud final ponderada: {np.mean(sims_final):.6f}")
        logger.debug("Todas las líneas ordenadas: %s", ", ".join(str(lid) for lid in all_line_ids))
        sims_str = "[" + "  ".join(f"{val:7.6f}" for val in sims_final) + "]"
        logger.debug("Similitudes de emergencia finales:\n%s", sims_str)

        # Log detallado por línea
        for idx, (line_id, sim) in enumerate(zip(all_line_ids, sims_final)):
            logger.info(f"{line_id}: Sim: {sim:7.4f}")
        
        try:
            matched_indices = [idx for idx, sim in enumerate(sims_final) if sim > similarity_threshold]

            # Si no hay coincidencias, intentar con el umbral de emergencia más bajo
            if not matched_indices:
                logger.warning(f"Ninguna línea superó el umbral de {similarity_threshold}. Intentando con umbral de emergencia de {emergency_threshold}.")
                matched_indices = [idx for idx, sim in enumerate(sims_final) if sim > emergency_threshold]

            if not matched_indices:
                logger.warning("Ninguna línea superó el umbral de emergencia. No se encontraron clusters.")
                return []

            # Obtener las line_ids que pasaron el umbral
            candidate_line_ids = [all_line_ids[i] for i in matched_indices]
            
            # Ordenar por line_id (ascendente)
            sorted_candidates = sorted(candidate_line_ids, key=lambda x: line_ids.index(x))
            
            # Encontrar el cluster más grande que respete min_cluster e interval
            table_line_ids = self._find_best_cluster(sorted_candidates, min_cluster, interval_margin, all_lines)
            
            logger.info(f"Cluster '{len(table_line_ids)}' encontrado por fallback de emergencia: {table_line_ids}")
            return table_line_ids
        except Exception as e:
            logger.error(f"Error en falback de emergencia: {e}", exc_info=True)
            return []
        
    def _find_best_cluster(self, sorted_candidates: List[str], min_cluster: int, interval_margin: int, all_lines: Dict[str, AllLines]) -> List[str]:
        """Encuentra el mejor cluster respetando min_cluster e interval y devuelve todas las líneas del intervalo."""
        if len(sorted_candidates) < min_cluster:
            logger.warning(f"No hay suficientes candidatos '{len(sorted_candidates)}' para min_cluster: '{min_cluster}'")
            return []
        
        all_line_ids = list(all_lines.keys())
        candidate_indices = [all_line_ids.index(lid) for lid in sorted_candidates]
        
        best_start = None
        best_end = None
        best_size = 0

        for i in range(len(candidate_indices)):
            start_idx = candidate_indices[i]
            end_idx = start_idx
            current_size = 1

            for j in range(i + 1, len(candidate_indices)):
                if candidate_indices[j] - candidate_indices[j-1] <= interval_margin:
                    end_idx = candidate_indices[j]
                    current_size += 1
                else:
                    break

            if current_size > min_cluster and current_size > best_size:
                best_start = start_idx
                best_end = end_idx
                best_size = current_size

        if best_start is not None and best_end is not None:
            # Devuelve todas las líneas entre best_start y best_end (inclusive)
            return all_line_ids[best_start:best_end+1]
        else:
            return []