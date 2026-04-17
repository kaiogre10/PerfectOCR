# core/workers/vectorial_transformation/geometric_table_structurer.py
import logging
import time
from typing import List, Dict, Any, Tuple, cast
import pandas as pd #type: ignore
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_models import Polygons, AllLines
from core.domain.data_formatter import DataFormatter
from core.utils.math_utils import alignment, euclidean_distance
from core.utils.text_utils import format_cuant
# from services.output_service import save_debug_table

logger = logging.getLogger(__name__)

class GeometricTableStructurer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        # worker_config = config.get('table_structurer', {})
        self.output = config.get("table_structured", False)

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
            
            # for line_id, line_data in all_lines.items():
            #     if line_data.header_line is not None:
            #         logger.info(f"Texto de {line_id}: '{line_data.text}', polygons: {line_data.polygon_ids}")
            #     if line_data.polygon_ids:
            #         logger.info(f"{line_data.text}")
                    
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            tabular_line_ids = [lid for lid, line_obj in all_lines.items() if line_obj.tabular_line]
                
            if not tabular_line_ids or not all_lines or not polygons:
                logger.error("Faltan datos necesarios para estructuración tabular")
                return False
                
            try:
                H, header_line_id = self.get_headers(all_lines, polygons)
                if not header_line_id or H==0:
                    logger.error("No hay encabezados disponibles")
                    return False
                manager.workflow.H = H
                
                # logger.info(f"Encabezado detectado: {header_line_id}, H={H} columnas")

                # Pasar target_columns a la función de extracción
                header_centroids = self._extract_header_centroids(header_line_id, all_lines, polygons, H)

                # 3. Seleccionar filas tabulares para procesamiento
                selected_lines = self._select_table_rows(tabular_line_ids, all_lines)
                
                # 4. Aplicar algoritmo geométrico de asignación a celdas
                table_matrix = self._apply_geometric_assignment(selected_lines, all_lines, polygons, header_centroids, H)
                # logger.info(f"Tabla matrix generada con {table_matrix} filas")

                # Publicar estructura rica en contexto para workers posteriores (ej. Math Max)
                context["table_matrix"] = table_matrix

                # 5. Validar y loggear estructura generada
                total_time = time.time() - start_time
                if table_matrix:
                    logger.debug(f"Estructuración de tabla completada en {total_time:.10f}s")

                    # if self.output:
                    # all_lines = manager.workflow.all_lines if manager.workflow else {}
                    # polygons = manager.workflow.polygons if manager.workflow else {}
                    # df = self._create_structured_dataframe(table_matrix, H)
                    # logger.info("Tabla geometrical:\n" + df.to_string(index=False))


                        # header_line_ids = [lid for lid, l in all_lines.items() if getattr(l, "header_line", False)]
                        # header_line_id = header_line_ids[0] if header_line_ids else None

                        # header_polygons = []
                        # if header_line_id and header_line_id in all_lines:
                        #     line_obj = all_lines[header_line_id]
                        #     polygon_ids = getattr(line_obj, "polygon_ids", [])
                        #     header_polygons = [polygons[pid] for pid in polygon_ids if pid in polygons]

                        # file_name: str = manager.workflow.metadata.image_name # type: ignore
                        # worker_name = context.get("worker_name") or "geometrical_structurer"
                        # output_paths = context["output_paths"]
                        # save_debug_table(df, file_name, output_paths, worker_name, header_polygons)

                    logger.debug("Table matrix publicada en contexto éxitosamente")
                    return True

                return False
            
            except Exception as e:
                logger.error(f"Error en línea de encabezado: {e}", exc_info=True)
                return False

        except Exception as e:
            logger.error(f"Error en estructuración geométrica: {e}", exc_info=True)
            return False

    def _extract_header_centroids(self, header_line_id: str, all_lines: Dict[str, AllLines], polygons: Dict[str, Polygons], target_columns: int) -> List[List[float]]:
        """
        Extrae centroides de referencia c_j del encabezado H*.
        Si hay menos polígonos que columnas necesarias, subdivide los polígonos.
        """
        try:
            header_centroids: List[List[float]] = []
            header_line = all_lines[header_line_id]
            
            # Obtener polígonos del encabezado ordenados por x (izquierda a derecha)
            header_polys: List[Tuple[str, Polygons]] = []
            for poly_id in header_line.polygon_ids:
                poly_data = polygons.get(poly_id)
                if poly_data and poly_data.geometry:
                    header_polys.append((poly_id, poly_data))
            
            # Ordenar por posición x (centroide en eje X)
            header_polys.sort(key=lambda x: x[1].geometry.centroid[0])
            
            num_polys = len(header_polys)
            
            if num_polys == 0:
                return []
            
            # Si tenemos suficientes polígonos, usar directamente sus centroides
            if num_polys >= target_columns:
                for _, poly_data in header_polys[:target_columns]:
                    centroid = poly_data.geometry.centroid.tolist()
                    header_centroids.append(centroid)
            else:
                # Necesitamos subdivider los polígonos
                subdivisions_per_poly = target_columns // num_polys
                remainder = target_columns % num_polys
                
                for idx, (poly_id, poly_data) in enumerate(header_polys):
                    geom = poly_data.geometry
                    bbox = geom.bounding_box
                    
                    # Determinar cuántas subdivisiones para este polígono
                    num_subdivisions = subdivisions_per_poly
                    if idx < remainder:
                        num_subdivisions += 1
                    
                    # Calcular centroides de las subdivisiones
                    x_min, x_max = float(bbox[0]), float(bbox[2])
                    y_centroid = float(geom.centroid[1])
                    
                    # Dividir el ancho en partes iguales
                    segment_width = (x_max - x_min) / num_subdivisions
                    
                    for sub_idx in range(num_subdivisions):
                        sub_x = x_min + (sub_idx + 0.5) * segment_width
                        header_centroids.append([sub_x, y_centroid])
                 
            return header_centroids
        
        except Exception as e:
            logger.error(f"Error extrayendo encabezado: {e}", exc_info=True)
            return []

    def _select_table_rows(self, tabular_line_ids: List[str], all_lines: Dict[str, AllLines]) -> List[str]:
        """
        Selecciona solo filas marcadas como tabulares, preservando el orden de lectura.
        """
        try:
            all_line_ids = list(all_lines.keys())
            tabular_set = set(tabular_line_ids)
            selected_lines = [line_id for line_id in all_line_ids if line_id in tabular_set]

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
                semantic_blocks = self._build_semantic_blocks(row_elements)
                B_k = len(semantic_blocks)
                
                # Inicializar fila de celdas vacías
                row_cells: List[Dict[str, Any]] = [{'words': []} for _ in range(H)]
                
                if L_k == 0:
                    table_matrix.append(self._finalize_row_cells(row_cells, H))
                    continue

                # CASO 0 por bloques: B_k == H (mismo número de bloques que columnas)
                if B_k == H:
                    row_cells = self._case_exact_assignment_by_blocks(semantic_blocks, H)
                    table_matrix.append(self._finalize_row_cells(row_cells, H))
                    continue

                # CASO A por bloques: B_k > H (más bloques que columnas)
                if B_k > H:
                    row_cells = self._case_a_assignment_by_blocks(semantic_blocks, H, B_k)

                # CASO 0 por polígonos: L_k == H (fallback)
                elif L_k == H:
                    row_cells = self._case_exact_assignment(row_elements, H)

                # CASO A por polígonos: L_k > H (fallback)
                elif L_k > H:
                    row_cells = self._case_a_assignment(row_elements, H, L_k)

                # CASO B: L_k < H (menos palabras que columnas)
                else:
                    # logger.info(f"Asignación B para {line_id}, elementos: {L_k}")
                    row_cells = self._case_b_assignment(row_elements, H, L_k, header_centroids)
                
                table_matrix.append(self._finalize_row_cells(row_cells, H))
                
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
                    sc = poly_data.semantic_clasification
                    ocr_text = format_cuant(poly_data.ocr_text or "") if (4 in sc) else poly_data.ocr_text
                    element: Dict[str, Any] = {
                        "polygon_id": poly_id,
                        "xmin": geom.bounding_box[0],
                        "xmax": geom.bounding_box[2], 
                        "cx": geom.centroid[0],
                        "cy": geom.centroid[1],
                        "ocr_text": ocr_text,
                        "semantic_clasification": sc,
                        "lineal_id": line_obj.lineal_id,
                        "polygon_ids": line_obj.polygon_ids, 
                    }
                    row_elements.append(element)

                    # Orden estable izquierda->derecha para soportar asignación secuencial.
                    row_elements.sort(key=lambda element: float(element.get("cx", 0.0)))
                    
            return row_elements
        
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _case_exact_assignment(self, row_elements: List[Dict[str, Any]], H: int) -> List[Dict[str, Any]]:
        """
        CASO 0: L_k == H - Asignación secuencial 1 a 1 por orden horizontal.
        """
        try:
            row_cells: List[Dict[str, Any]] = [{'words': []} for _ in range(H)]
            for col_idx, element in enumerate(row_elements[:H]):
                row_cells[col_idx]['words'] = [element]

            return row_cells

        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _case_exact_assignment_by_blocks(self, semantic_blocks: List[List[Dict[str, Any]]], H: int) -> List[Dict[str, Any]]:
        """
        CASO 0 por bloques: B_k == H - Asignación secuencial 1 bloque por columna.
        """
        try:
            row_cells: List[Dict[str, Any]] = [{'words': []} for _ in range(H)]
            for col_idx, block in enumerate(semantic_blocks[:H]):
                row_cells[col_idx]['words'] = block
            return row_cells
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _case_a_assignment(self, row_elements: List[Dict[str, Any]], H: int, L_k: int) -> List[Dict[str, Any]]:
        """
        CASO A: L_k ≥ H - Algoritmo de distancias horizontales Δ_i
        Calcula Δ_i = x_{i+1}^min - x_i^max y selecciona H-1 mayores espacios.
        """
        try:
            row_cells: List[Dict[str, Any]] = [{'words': []} for _ in range(H)]

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

    def _case_a_assignment_by_blocks(self, semantic_blocks: List[List[Dict[str, Any]]], H: int, B_k: int) -> List[Dict[str, Any]]:
        """
        CASO A por bloques: B_k > H.
        Usa distancias horizontales entre bloques contiguos para elegir cortes.
        """
        try:
            row_cells: List[Dict[str, Any]] = [{'words': []} for _ in range(H)]
            if not semantic_blocks:
                return row_cells

            horizontal_distances: List[Tuple[float, int]] = []
            for i in range(B_k - 1):
                left_block = semantic_blocks[i]
                right_block = semantic_blocks[i + 1]
                left_xmax = max(float(elem.get('xmax', 0.0)) for elem in left_block)
                right_xmin = min(float(elem.get('xmin', 0.0)) for elem in right_block)
                delta_i = max(0.001, right_xmin - left_xmax)
                horizontal_distances.append((delta_i, i))

            horizontal_distances.sort(key=lambda x: x[0], reverse=True)
            cut_indices = sorted([idx for _, idx in horizontal_distances[:H - 1]])

            start_idx = 0
            for col_idx in range(H):
                if col_idx < len(cut_indices):
                    end_idx = cut_indices[col_idx] + 1
                else:
                    end_idx = B_k

                merged_words: List[Dict[str, Any]] = []
                for block in semantic_blocks[start_idx:end_idx]:
                    merged_words.extend(block)
                row_cells[col_idx]['words'] = merged_words
                start_idx = end_idx

                if start_idx >= B_k:
                    break

            return row_cells
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

    def _build_semantic_blocks(self, row_elements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Construye bloques semánticos contiguos sobre la fila (izquierda->derecha).
        Regla:
        - SC=4 (cuantitativo) siempre va aislado en un bloque propio.
        - El resto se agrupa si la clase dominante es la misma y no es 4.
        """
        try:
            if not row_elements:
                return []

            blocks: List[List[Dict[str, Any]]] = []
            current_block: List[Dict[str, Any]] = []
            current_class: int | None = None

            for element in row_elements:
                element_class = self._get_primary_semantic_class(element)

                if element_class == 4:
                    if current_block:
                        blocks.append(current_block)
                        current_block = []
                        current_class = None
                    blocks.append([element])
                    continue

                if not current_block:
                    current_block = [element]
                    current_class = element_class
                    continue

                if current_class == element_class:
                    current_block.append(element)
                else:
                    blocks.append(current_block)
                    current_block = [element]
                    current_class = element_class

            if current_block:
                blocks.append(current_block)

            return blocks
        except Exception as e:
            logger.error(f"Error construyendo bloques semánticos: {e}", exc_info=True)
            return [[element] for element in row_elements]

    def _get_primary_semantic_class(self, element: Dict[str, Any]) -> int:
        """
        Obtiene la clase semántica dominante del elemento.
        Prioriza SC=4 si existe en la lista.
        """
        semantic_value = element['semantic_clasification']
        if isinstance(semantic_value, list):
            semantic_list = [int(v) for v in semantic_value if isinstance(v, (int, float, str))]
            if 4 in semantic_list:
                return 4
            if semantic_list:
                return semantic_list[0]
            return 1
        try:
            return int(semantic_value)
        except Exception:
            return 1

    def _case_b_assignment(self, row_elements: List[Dict[str, Any]], H: int, L_k: int, header_centroids: List[List[float]]) -> List[Dict[str, Any]]:
        """
        CASO B: L_k < H - Asignación por similitud coseno con restricciones semánticas
        Cada polígono se asigna individualmente UNA SOLA VEZ considerando disponibilidad semántica.
        """
        try:
            # Inicializar celdas vacías
            row_cells: List[Dict[str, Any]] = [{'words': []} for _ in range(H)]
            
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
                element_semantic: List[int]= element['semantic_clasification']
                
                # 1. Filtrar celdas semánticamente disponibles
                available_columns: List[int] = []
                for col_idx in range(H):
                    cell_content = row_cells[col_idx]['words']
                    
                    # Verificar si la celda está semánticamente disponible
                    if self._is_semantically_available(cell_content, element_semantic):
                        available_columns.append(col_idx)
                
                # 2. Determinar la mejor columna basada en disponibilidad
                if len(available_columns) > 1:
                    distances: List[Tuple[float, int]] = []
                    for col_idx in available_columns:
                        if col_idx < len(header_centroids):
                            header_centroid = header_centroids[col_idx]
                            distance = euclidean_distance(element_centroid, header_centroid) # type: ignore
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
            return [{'words': []} for _ in range(H)]

    def _finalize_row_cells(self, row_cells: List[Dict[str, Any]], H: int) -> List[Dict[str, Any]]:
        """
        Reduce la representación interna de cada celda a solo texto, IDs de polígonos
        y clasificación semántica.
        """
        final_row: List[Dict[str, Any]] = []

        for cell in row_cells[:H]:
            cell_words = cell.get('words', [])
            text = " ".join(
                [str(elem.get('ocr_text', '')).strip() for elem in cell_words if elem.get('ocr_text')]
            ).strip()

            polygon_ids: List[str] = []
            semantic_values: List[int] = []

            for elem in cell_words:
                polygon_id = elem.get('polygon_id')
                if polygon_id and polygon_id not in polygon_ids:
                    polygon_ids.append(polygon_id)

                semantic_value = elem['semantic_clasification']
                if isinstance(semantic_value, list):
                    semantic_list = cast(List[Any], semantic_value)
                    for value in semantic_list:
                        if value is None:
                            continue
                        if not isinstance(value, (int, float, str)):
                            continue
                        try:
                            semantic_values.append(int(value))
                        except Exception:
                            continue
                elif semantic_value is not None:
                    semantic_values.append(int(semantic_value))

            final_row.append({
                'text': text,
                'polygon_ids': polygon_ids,
                'semantic_clasification': semantic_values,
            })

        while len(final_row) < H:
            final_row.append({
                'text': '',
                'polygon_ids': [],
                'semantic_clasification': [],
            })
            
        text = [row['text'] for row in final_row]
        return final_row

    def _is_semantically_available(self, cell_content: List[Dict[str, Any]], element_semantic: List[int] | int) -> bool:
        """
        Verifica si una celda está semánticamente disponible para un elemento.
        Reglas:
        - SC=4 (cuantitativo) siempre va aislado.
        - Nunca conviven cuantitativos con otros tipos.
        - Como máximo 1 cuantitativo por celda.
        """
        try:
            # Tipos que tienen restricciones (solo uno por celda)
            restricted_types = {4}
            current_semantics = set(element_semantic if isinstance(element_semantic, list) else [element_semantic])
            current_is_restricted = bool(current_semantics & restricted_types)

            # Analizar contenido semántico actual de la celda
            cell_has_restricted = False
            cell_has_non_restricted = False
            for existing_element in cell_content:
                existing_semantic_val = existing_element['semantic_clasification']
                existing_semantics = set(existing_semantic_val if isinstance(existing_semantic_val, list) else [existing_semantic_val]) # type: ignore
                if existing_semantics & restricted_types:
                    cell_has_restricted = True
                else:
                    cell_has_non_restricted = True

            # Si la celda ya tiene un cuantitativo, no puede entrar ningún otro elemento.
            if cell_has_restricted:
                return False

            # Si el elemento actual es cuantitativo, no puede entrar a celda no vacía.
            if current_is_restricted and (cell_has_non_restricted or len(cell_content) > 0):
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
                row_data = [cell.get('text', '') for cell in row[:H]]
                df_data.append(row_data)
                
            df = pd.DataFrame(df_data, columns=columns)
        
            logger.debug(f"Columnas: {len(columns)} y filas: {len(df_data)}")
            
            return df
        except Exception as e:
            logger.error(f"Error creando datadrame: {e}", exc_info=True)
            return pd.DataFrame()

    def get_headers(self, all_lines: Dict[str, AllLines], polygons: Dict[str, Polygons]) -> Tuple[int, str]:
        """Asignar key_field = 6 a todos los polígonos de la línea de encabezadoy calcular la cantidad de columnas (H) basado en los key_fields"""
        for line_id, line_data in all_lines.items():
            if line_data.header_line is not None:
                # line_text = line_data.text
                header_line_id = line_id
                h = 0
                header_line_text: List[str] = []
                
                for poly_id in line_data.polygon_ids:
                    poly = polygons.get(poly_id)
                    if poly:
                        poly_text = poly.ocr_text or ""
                        if poly.key_field is None:
                            poly.key_field = [6]
                            h += 1
                            header_line_text.append(poly_text)
                        elif isinstance(poly.key_field, list):
                            h += len(poly.key_field)
                            header_line_text.append(poly_text)
                            if 6 not in poly.key_field:
                                poly.key_field.append(6)
                                header_line_text.append(poly_text)
                        else:
                            h += 1
                            header_line_text.append(poly_text)
                            if poly.key_field != 6:
                                poly.key_field = [poly.key_field, 6]
                            
                # logger.info(f"H: {h} LINEA DE ENCABEZADO POLY:'{" ".join(header_line_text)}'\n"f"Linea: '{line_text}'")
                return h, header_line_id
        return (0, "")