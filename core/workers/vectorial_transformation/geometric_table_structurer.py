# PerfectOCR/core/workers/vectorial_transformation/geometric_table_structurer.py
import logging
import time
from typing import List, Dict, Any
import pandas as pd # type: ignore
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_models import Polygons, AllLines
from core.domain.data_formatter import DataFormatter
from core.utils.cosine_similarity import alignment

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
            all_lines: Dict[str, AllLines] = manager.workflow.all_lines
            polygons: Dict[str, Polygons] = manager.workflow.polygons
            
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
                    logger.info(f"Se encontraron {len(table_matrix)} filas.\n{df.to_string(index=False)}")
                else:
                    logger.warning("No se pudo generar la tabla estructurada, el DataFrame está vacío.")
                logger.debug(f"Estructuración de tabla completada en {total_time:.10f} s.")
                
                # 7. Guardar usando DataFormatter moderno
                if df.empty:
                    logger.warning("No se guardará la tabla estructurada porque está vacía.")
                    return False
                success = manager.save_structured_table(df=df, columns=list(df.columns))
                
                return success
            
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
            min_cosine_similarity = self.worker_config.get("min_cosine_similarity")
            
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
                    row_cells = self._case_a_assignment(row_elements, H, L_k)
                
                # CASO B: L_k < H (Menos palabras que columnas)  
                else:
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
                        "semantic_type": poly_data.semantic_type,
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
            
            if H == 1:
                row_cells[0]['words'] = row_elements
                return row_cells
            
            # 1. Calcular distancias horizontales Δ_i
            horizontal_distances: List[tuple[float, int]] = []
            for i in range(L_k - 1):
                x_i_max = float(row_elements[i].get('xmax', 0))
                x_i1_min = float(row_elements[i + 1].get('xmin', 0))
                delta_i = max(0.001, x_i1_min - x_i_max)  # ε = 0.001 para solapamientos
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

    def _case_b_assignment(self, row_elements: List[Dict[str, Any]], H: int, L_k: int,
                          header_centroids: List[List[float]]) -> List[Dict[str, Any]]:
        """
        CASO B: L_k < H - Asignación por similitud coseno o secuencial
        """
        try:
            row_cells: List[Dict[str, Any]] = [{'words': [], 'cell_text': ''} for _ in range(H)]
            
            # Subcaso B.1: L_k = 1 - Similitud coseno con centroides de encabezado
            if L_k == 1:
                element = row_elements[0]
                element_centroid = [float(element.get('cx', 0)), float(element.get('cy', 0))]
                
                # Calcular similitudes con todos los centroides y escoger el argmax.
                # IMPORTANTE: siempre asignar al argmax aunque la similitud sea baja.
                if not header_centroids:
                    # sin encabezado, asignar a la primera columna
                    row_cells[0]['words'] = [element]
                else:
                    sims = [alignment(hc, element_centroid) for hc in header_centroids]
                    # escoger índice del máximo (si hay empates, elige el primero)
                    best_col = int(max(range(len(sims)), key=lambda j: sims[j]))
                    best_similarity = sims[best_col]

                    # Si por alguna razón best_similarity es NaN o inválido, fallback por proximidad X
                    try:
                        if not (isinstance(best_similarity, float) and best_similarity == best_similarity):
                            raise ValueError("similarity inválida")
                    except Exception:
                        elem_x = element_centroid[0]
                        distances = [abs(float(hc[0]) - elem_x) for hc in header_centroids]
                        best_col = int(min(range(len(distances)), key=lambda j: distances[j]))

                    row_cells[best_col]['words'] = [element]
            
            # Subcaso B.2: 1 < L_k < H - Asignación secuencial
            else:
                for i in range(min(L_k, H)):
                    row_cells[i]['words'] = [row_elements[i]]
                    
            return row_cells
        
        except Exception as e:
            logger.error(f"Error en geometric: {e}", exc_info=True)
            return []

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