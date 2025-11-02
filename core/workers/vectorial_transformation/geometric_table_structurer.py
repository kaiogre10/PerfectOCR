# PerfectOCR/core/workers/vectorial_transformation/geometric_table_structurer.py
import logging
import time
from typing import List, Dict, Any, Tuple
import pandas as pd #type: ignore
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_models import Polygons, AllLines
from core.domain.data_formatter import DataFormatter
from core.utils.math_utils import alignment, euclidean_distance

logger = logging.getLogger(__name__)

class GeometricTableStructurer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('table_structurer', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_structured", False)

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Implementa el algoritmo geométrico de estructuración tabular en ℝ²
        basado en el modelo matemático riguroso de distancias horizontales y similitud coseno.
        """
        try:
            start_time = time.time()
            logger.debug("GeometricTableStructurer iniciado")
            
            if not manager.workflow:
                logger.warning("No hay workflow disponible")
                return False
                
            # Obtener datos usando data classes modernas
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            
            # Filtrar líneas tabulares usando propiedades de data class
            tabular_line_ids = [lid for lid, line_obj in all_lines.items() if line_obj.tabular_line]
            
            if not tabular_line_ids or not all_lines or not polygons:
                logger.error("Faltan datos necesarios para estructuración tabular")
                return False

            # 1. Detectar encabezado H* usando data classes
            try:
                # buscar líneas marcadas explícitamente como header_line == True
                header_line_id = [lid for lid, l in all_lines.items() if getattr(l, "header_line", False)]
                header_line_id = header_line_id[0] if header_line_id else None

                line_ids: List[str] = list(all_lines.keys())
                if header_line_id not in line_ids:
                    logger.warning("Header no encontrado en el manager")
                    return False
                
                # 2. Extraer centroides de referencia c_j del encabezado
                header_centroids = self._extract_header_centroids(header_line_id, all_lines, polygons)
                H = len(header_centroids)  # Número de columnas
                
                if H == 0:
                    logger.error("No se pudieron extraer centroides del encabezado")
                    return False
                    
                logger.debug(f"Encabezado detectado: {header_line_id}, H={H} columnas")

                # 3. Seleccionar filas S para procesamiento
                selected_lines = self._select_table_rows(header_line_id, tabular_line_ids, all_lines)
                
                # 4. Aplicar algoritmo geométrico de asignación a celdas
                table_matrix = self._apply_geometric_assignment(selected_lines, all_lines, polygons, header_centroids, H)

                # 5. Generar DataFrame estructurado
                df = self._create_structured_dataframe(table_matrix, H)
                
                # 6. LOG COMPLETO DE LA TABLA ESTRUCTURADA
                total_time = time.time() - start_time
                if not df.empty:
                    logger.info(f"Se encontraron {len(table_matrix)} filas.\n{df.to_string(index=False)}") # type: ignore
                    logger.debug(f"Estructuración de tabla completada en {total_time:.10f} s.")

                    context["table_copy"] = df.copy()

                    if self.output:
                        from services.output_service import save_debug_table
                        all_lines = manager.workflow.all_lines if manager.workflow else {}
                        polygons = manager.workflow.polygons if manager.workflow else {}

                        header_line_ids = [lid for lid, l in all_lines.items() if getattr(l, "header_line", False)]
                        header_line_id = header_line_ids[0] if header_line_ids else None

                        header_polygons = []
                        if header_line_id and header_line_id in all_lines:
                            line_obj = all_lines[header_line_id]
                            polygon_ids = getattr(line_obj, "polygon_ids", [])
                            header_polygons = [polygons[pid] for pid in polygon_ids if pid in polygons]

                        file_name: str = manager.workflow.metadata.image_name # type: ignore
                        worker_name = context.get("worker_name") or "geometrical_structurer"
                        output_paths = context.get("output_paths", [])
                        save_debug_table(df, file_name, output_paths, worker_name, header_polygons)

                    if manager.save_structured_table(df=df, columns=list(df.columns)):
                        logger.debug("Tabla guardada éxitosamente")

                        return True

                return False
            
            except Exception as e:
                logger.error(f"Error en línea de encabezado: {e}", exc_info=True)
                return False

        except Exception as e:
            logger.error(f"Error en estructuración geométrica: {e}", exc_info=True)
            return False

    def _extract_header_centroids(self, header_line_id: str, all_lines: Dict[str, AllLines], 
                                 polygons: Dict[str, Polygons]) -> List[List[float]]:
        """
        Extrae centroides de referencia c_j = (c_x,h_j, c_y,h_j) del encabezado H*
        usando acceso directo a data classes.
        """
        try:
            header_centroids: List[List[float]] = []
            header_line = all_lines[header_line_id]
            
            for poly_id in header_line.polygon_ids:
                poly_data = polygons.get(poly_id)
                if poly_data and poly_data.geometry:
                    # Acceso directo a centroide usando data class
                    centroid = poly_data.geometry.centroid.tolist()
                    header_centroids.append(centroid)

            logger.debug(f"HEADER CENTROIDS: {len(header_line.polygon_ids)}")
 
            return header_centroids
        
        except Exception as e:
            logger.error(f"Error extrayendo ecabezado: {e}", exc_info=True)
            return []

    def _select_table_rows(self, header_line_id: str, tabular_line_ids: List[str], 
                          all_lines: Dict[str, AllLines]) -> List[str]:
        """
        Selecciona filas S_k del conjunto P  H* para procesamiento tabular.
        """
        try:
            all_line_ids = list(all_lines.keys())
            line_order = {lid: idx for idx, lid in enumerate(all_line_ids)}
            
            if header_line_id in line_order and tabular_line_ids:
                header_idx = line_order[header_line_id]
                last_tabular_idx = max([line_order[lid] for lid in tabular_line_ids if lid in line_order])
                selected_lines = all_line_ids[header_idx + 1:last_tabular_idx + 1]
            else:
                selected_lines = tabular_line_ids
                
            return selected_lines
        
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _apply_geometric_assignment(self, selected_lines: List[str], all_lines: Dict[str, AllLines], polygons: Dict[str, Polygons], header_centroids: List[List[float]], H: int) -> List[List[Dict[str, Any]]]:
        """
        Implementa el algoritmo geométrico de asignación a celdas T[k][j]
        según los Casos A y B del modelo matemático.
        """
        try:
            table_matrix: List[List[Dict[str, Any]]] = []
            
            for line_id in selected_lines:
                line_obj = all_lines[line_id]
                
                # Extraer elementos P_i de la fila S_k usando data classes
                row_elements = self._extract_row_elements(line_obj, polygons)
                L_k = len(row_elements)  # Cardinalidad |S_k|
                
                # Inicializar fila de celdas vacías
                row_cells: List[Dict[str, Any]] = [{'words': [], 'cell_text': ''} for _ in range(H)]
                
                if L_k == 0:
                    table_matrix.append(row_cells)
                    continue
                
                # CASO A: L_k ≥ H (Más palabras que columnas)
                if L_k >= H:
                    # logger.info(f"Asignación A para {line_id}, elementos {L_k}")
                    row_cells = self._case_a_assignment(row_elements, H, L_k)
                
                # CASO B: L_k < H (Menos palabras que columnas)
                if L_k < H:
                    # logger.info(f"Asignación B para {line_id}, elementos: {L_k}")
                    row_cells = self._case_b_assignment(row_elements, H, L_k, header_centroids)
                
                # Generar texto de celda
                for cell_idx in range(H):
                    cell_elements = row_cells[cell_idx]['words']
                    if cell_elements:
                        row_cells[cell_idx]['cell_text'] = " ".join([elem.get('ocr_text', '') for elem in cell_elements]).strip()
                
                table_matrix.append(row_cells)
                
            return table_matrix
        
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _extract_row_elements(self, line_obj: AllLines, polygons: Dict[str, Polygons]) -> List[Dict[str, Any]]:
        """
        Extrae elementos P_i con atributos geométricos de una fila S_k.
        """
        try:
            row_elements: List[Dict[str, Any]] = []
            
            for poly_id in line_obj.polygon_ids:
                poly_data = polygons.get(poly_id)
                if poly_data and poly_data.geometry:
                    geom = poly_data.geometry
                    element: Dict[str, Any] = {
                        "xmin": geom.bounding_box[0],
                        "xmax": geom.bounding_box[2], 
                        "cx": geom.centroid[0],
                        "cy": geom.centroid[1],
                        "ocr_text": poly_data.ocr_text or "",
                        "semantic_clasification": poly_data.semantic_clasification,
                        "lineal_id": line_obj.lineal_id,
                        "polygon_ids": line_obj.polygon_ids, 
                    }
                    row_elements.append(element)
                    
            return row_elements
        
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _case_a_assignment(self, row_elements: List[Dict[str, Any]], H: int, L_k: int) -> List[Dict[str, Any]]:
        """
        CASO A: L_k ≥ H - Algoritmo de distancias horizontales Δ_i
        Calcula Δ_i = x_{i+1}^min - x_i^max y selecciona H-1 mayores espacios.
        """
        try:
            row_cells: List[Dict[str, Any]] = [{'words': [], 'cell_text': ''} for _ in range(H)]

            # 1. Calcular distancias horizontales Δ_i
            horizontal_distances: List[Tuple[float, float]] = []
            for i in range(L_k - 1):
                x_i_max = float(row_elements[i].get('xmax', 0))
                x_i1_min = float(row_elements[i + 1].get('xmin', 0))
                delta_i = max(0.001, x_i1_min - x_i_max) 
                horizontal_distances.append((delta_i, i))
            
            # 2. Seleccionar H-1 mayores Δ_i como puntos de corte J
            horizontal_distances.sort(key=lambda x: x[0], reverse=True)
            cut_indices = sorted([idx for _, idx in horizontal_distances[:H-1]])
            
            # 3. Asignación a intervalos T[k][j]
            start_idx = 0
            for col_idx in range(H):
                if col_idx < len(cut_indices):
                    end_idx = cut_indices[col_idx] + 1
                else:
                    end_idx = L_k
                    
                row_cells[col_idx]['words'] = row_elements[start_idx:end_idx]
                start_idx = end_idx
                
                if start_idx >= L_k:
                    break
                    
            return row_cells
        
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _case_b_assignment(self, row_elements: List[Dict[str, Any]], H: int, L_k: int, header_centroids: List[List[float]]) -> List[Dict[str, Any]]:
        """
        CASO B: L_k < H - Asignación por similitud coseno con restricciones semánticas
        Cada polígono se asigna individualmente UNA SOLA VEZ considerando disponibilidad semántica.
        """
        try:
            # Inicializar celdas vacías
            row_cells: List[Dict[str, Any]] = [{'words': [], 'cell_text': ''} for _ in range(H)]
            
            # Si no hay elementos en la fila, devolver celdas vacías
            if not row_elements:
                return row_cells
            
            # Si no hay centroides de encabezado, asignar todos a la primera columna
            if not header_centroids:
                for element in row_elements:
                    row_cells[0]['words'].append(element)
                logger.warning("No hay centroides de encabezado, asignando todos los elementos a la primera columna")
                return row_cells
            
            # Validar que tengamos suficientes centroides para las columnas
            if len(header_centroids) < H:
                logger.warning(f"Insuficientes centroides ({len(header_centroids)}) para {H} columnas. Usando los disponibles.")
                H = min(H, len(header_centroids)) # type: ignore
                row_cells = row_cells[:H]  # Ajustar tamaño de row_cells
            
            # Asignar cada elemento a una celda según restricciones semánticas
            for element in row_elements:
                element_centroid = [float(element.get('cx', 0)), float(element.get('cy', 0))]
                element_semantic: List[int] | int = element.get('semantic_clasification', 0)
                
                # 1. Filtrar celdas semánticamente disponibles
                available_columns: List[int] = []
                for col_idx in range(H):
                    cell_content = row_cells[col_idx]['words']
                    
                    # Verificar si la celda está semánticamente disponible
                    if self._is_semantically_available(cell_content, element_semantic):
                        available_columns.append(col_idx)
                
                # 2. Determinar la mejor columna basada en disponibilidad
                if len(available_columns) > 1:
                    # Múltiples opciones: usar distancia euclidiana
                    distances: List[Tuple[float, int]] = []
                    for col_idx in available_columns:
                        if col_idx < len(header_centroids):
                            header_centroid = header_centroids[col_idx]
                            distance: float = euclidean_distance(element_centroid, header_centroid)
                            distances.append((distance, col_idx))
                    
                    # Asignar a la columna con menor distancia si hay distancias calculadas
                    if distances:
                        best_col = min(distances, key=lambda x: x[0])[1]
                    else:
                        # Fallback si no hay distancias calculadas
                        best_col = available_columns[0]
                elif len(available_columns) == 1:
                    # Solo una opción disponible
                    best_col = available_columns[0]
                else:
                    # No hay columnas disponibles, usar algoritmo original como fallback
                    try:
                        sims: List[float] = []
                        for hc_idx, hc in enumerate(header_centroids):
                            if hc_idx < H:  # Limitar a H centroides
                                sim = alignment(hc, element_centroid)
                                sims.append(sim)
                        
                        if sims:
                            best_col = int(max(range(len(sims)), key=lambda j: sims[j]))
                        else:
                            # Si no hay similitudes, usar primera columna
                            best_col = 0
                    except Exception as e:
                        logger.warning(f"Error calculando similitud coseno: {e}")
                        best_col = 0  # Fallback a primera columna
                
                # Asegurar que best_col esté en rango
                best_col = int(max(0, min(best_col, H-1)))
                
                # Asignar elemento a la celda
                row_cells[best_col]['words'].append(element)
                
                logger.debug(f"elemento: {element.get('ocr_text', '')}, semantica: {element_semantic}, columnas_disponibles: {available_columns}, asignación: {best_col}")

            return row_cells
            
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return [{'words': [], 'cell_text': ''} for _ in range(H)]

    def _is_semantically_available(self, cell_content: List[Dict[str, Any]], element_semantic: List[int] | int) -> bool:
        """
        Verifica si una celda está semánticamente disponible para un elemento.
        Reglas:
        - Solo UN polígono numérico/cuantitativo por celda
        - Descriptivos y códigos no tienen restricciones
        """
        try:
            # Tipos que tienen restricciones (solo uno por celda)
            restricted_types = {1, 2}
            
            current_semantics = set(element_semantic if isinstance(element_semantic, list) else [element_semantic])

            # Si el elemento no es restrictivo, siempre puede ir
            if not (current_semantics & restricted_types):
                return True
            
            # Si el elemento ES restrictivo, verificar que no haya otros restrictivos en la celda
            for existing_element in cell_content:
                existing_semantic_val = existing_element.get('semantic_clasification', 0)
                existing_semantics = set(existing_semantic_val if isinstance(existing_semantic_val, list) else [existing_semantic_val])
                
                if existing_semantics & restricted_types:
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error verificando disponibilidad semántica: {e}", exc_info=True)
            return True  # En caso de error, permitir asignación
        
    def _create_structured_dataframe(self, table_matrix: List[List[Dict[str, Any]]], H: int) -> pd.DataFrame:
        """
        Genera DataFrame estructurado a partir de la matriz de celdas T[k][j].
        """
        try:
            columns = [f"col_{i}" for i in range(H)]
            if not table_matrix:
                return pd.DataFrame(columns=columns)

            df_data: List[List[str]] = []
            for row in table_matrix:
                row_data = [cell.get('cell_text', '') for cell in row[:H]]
                df_data.append(row_data)
                
            df = pd.DataFrame(df_data, columns=columns)
        
            logger.debug(f"Columnas: {len(columns)} y filas: {len(df_data)}")
            
            return df
        except Exception as e:
            logger.error(f"Error creando datadrame: {e}", exc_info=True)
            return pd.DataFrame()
