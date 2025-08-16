# PerfectOCR/core/table_structure/geometric_table_structurer.py
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math
import time
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
import pandas

logger = logging.getLogger(__name__)

class GeometricTableStructurer(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('table_structurer', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("table_structured", False)        

        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> Optional[pandas.DataFrame]:
        try:
            start_time = time.time()
            
            # Obtener todo el diccionario de datos para tener acceso a todas las claves
            workflow_data = manager.get_dict_data()
            all_lines = workflow_data.get("all_lines", {})
            polygons = workflow_data.get("polygons", {})
            tabular_line_ids = list(workflow_data.get("tabular_lines", {}).keys())

            if not tabular_line_ids or not all_lines or not polygons:
                logger.warning("GeometricTableStructurer: Faltan datos necesarios (líneas tabulares, all_lines o polígonos) para la estructuración.")
                return None

            # 1. Asumir que la primera línea tabular es el encabezado y determinar H
            header_line_id = tabular_line_ids[0]
            header_poly_ids = all_lines.get(header_line_id, {}).get("polygon_ids", [])
            H = len(header_poly_ids)
            if H == 0:
                logger.warning("La línea de encabezado no tiene polígonos. No se puede determinar el número de columnas.")
                return None

            # 2. Construir 'main_header_line_elements' (H*) con la geometría requerida
            main_header_line_elements = []
            for poly_id in header_poly_ids:
                poly_data = polygons.get(poly_id)
                if poly_data and "geometry" in poly_data:
                    geom = poly_data["geometry"]
                    main_header_line_elements.append({
                        "text_raw": poly_data.get("ocr_text", ""),
                        "cx": geom.get("centroid", [0,0])[0],
                        "cy": geom.get("centroid", [0,0])[1]
                    })
            
            # 3. Construir 'lines_table_only' (S) para todas las líneas tabulares
            lines_table_only = []
            for line_id in tabular_line_ids:
                line_data = all_lines.get(line_id, {})
                poly_ids_for_line = line_data.get("polygon_ids", [])
                
                constituent_elements = []
                for poly_id in poly_ids_for_line:
                    poly_data = polygons.get(poly_id)
                    if poly_data and "geometry" in poly_data:
                        geom = poly_data["geometry"]
                        bbox = geom.get("bounding_box", [0,0,0,0])
                        constituent_elements.append({
                            "text_raw": poly_data.get("ocr_text", ""),
                            "xmin": bbox[0],
                            "ymin": bbox[1],
                            "xmax": bbox[2],
                            "ymax": bbox[3],
                            "cx": geom.get("centroid", [0,0])[0],
                            "cy": geom.get("centroid", [0,0])[1]
                        })
                
                lines_table_only.append({
                    "line_id": line_id,
                    "constituent_elements_ocr_data": constituent_elements
                })

            logger.info(f"Iniciando estructuración de tabla con H={H} columnas.")
            table_matrix = self.structure_table(lines_table_only, main_header_line_elements)

            # Convertir la matriz a DataFrame
            df = pandas.DataFrame([
                [cell.get('cell_text', '') for cell in row]
                for row in table_matrix
            ])
            # Usar la primera fila como encabezado si lo deseas
            if not df.empty:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)

            logger.info(f"Estructuración de tabla completada en {time.time() - start_time:.4f} segundos. Se encontraron {len(table_matrix)} filas.")
            logger.info(f"\n{df.to_string(index=False)}")
            return df

        except Exception as e:
            logger.error(f"Error fatal en GeometricTableStructurer.vectorize: {e}", exc_info=True)
            return None

    

    def structure_table(self,
                        lines_table_only: List[Dict[str, Any]],
                        main_header_line_elements: List[Dict[str, Any]]
                        ) -> List[List[Dict[str, Any]]]:
        """
        Transforms a list of text lines from a table area into a 2D cell structure.

        Args:
            lines_table_only: List of merged text lines for the table data area.
                              Each line dict has 'constituent_elements_ocr_data' (list of word dicts).
                              Each word dict requires 'text_raw', 'xmin', 'xmax', 'cx', 'cy'.
            main_header_line_elements: List of word/segment dicts for the main header line.
                                       Each element requires 'text_raw', 'cx', 'cy'.

        Returns:
            A 2D list (list of rows, where each row is a list of cell dicts).
            Each cell dict: {'words': List[Dict], 'cell_text': str}.
        """
        if not main_header_line_elements:
            logger.warning("GeometricTableStructurer: No header elements provided, cannot determine column count (H).")
            return []
        
        H = len(main_header_line_elements) # Number of columns
        if H == 0:
            logger.warning("GeometricTableStructurer: Number of columns (H) is 0 based on header elements.")
            return []

        logger.info(f"GeometricTableStructurer: Structuring table with H={H} columns.")
        
        header_centroids = []
        for header_elem in main_header_line_elements:
            if 'cx' in header_elem and 'cy' in header_elem:
                header_centroids.append((float(header_elem['cx']), float(header_elem['cy'])))
            else:
                logger.warning(f"Header element missing cx/cy: {header_elem.get('text_raw', 'N/A')}. Cannot use for B.1 centroid matching.")
                # Add a placeholder or handle appropriately if this case is critical
                header_centroids.append(None)


        table_matrix_T: List[List[Dict[str, Any]]] = []

        for k, line_sk in enumerate(lines_table_only):
            # Initialize row with H empty cells
            current_row_cells: List[Dict[str, Any]] = [{'words': [], 'cell_text': ''} for _ in range(H)]
            
            words_pk = line_sk.get('constituent_elements_ocr_data', [])
            # Ensure words are sorted by xmin, which should be the case from LineReconstructor
            words_pk.sort(key=lambda w: float(w.get('xmin', float('inf'))))

            lk = len(words_pk)

            if lk == 0:
                table_matrix_T.append(current_row_cells) # Add empty row
                continue

            # Case A: Lk >= H (Sufficient words for column cuts)
            if lk >= H:
                if H == 1:
                    current_row_cells[0]['words'] = words_pk
                else: # H > 1
                    horizontal_distances: List[Tuple[float, int]] = [] # (distance, index_of_first_word_in_pair)
                    for i in range(lk - 1):
                        word1_xmax = float(words_pk[i].get('xmax', 0))
                        word2_xmin = float(words_pk[i+1].get('xmin', 0))
                        dist = word2_xmin - word1_xmax
                        if dist < 0: # Overlapping words, treat as small distance
                            dist = 0.001 
                        horizontal_distances.append((dist, i))
                    
                    # Sort distances in descending order to find largest gaps
                    horizontal_distances.sort(key=lambda x: x[0], reverse=True)
                    
                    # Select H-1 largest distances as cut points (indices of the word *before* the cut)
                    # The cut occurs *after* words_pk[cut_indices[j]]
                    cut_indices_sorted = sorted([dist_info[1] for dist_info in horizontal_distances[:H-1]])
                    
                    start_idx = 0
                    for col_idx in range(H):
                        if col_idx < len(cut_indices_sorted):
                            end_idx = cut_indices_sorted[col_idx] + 1 # Words up to and including this index
                        else: # Last column
                            end_idx = lk
                        
                        current_row_cells[col_idx]['words'] = words_pk[start_idx:end_idx]
                        start_idx = end_idx
                        if start_idx >= lk and col_idx < H -1 : # Ran out of words before filling all H-1 cuts
                            logger.debug(f"Line {k}: Ran out of words ({lk}) while trying to fill {H} columns based on {H-1} cuts. Remaining columns will be empty.")
                            break


            # Case B: Lk < H (Insufficient words)
            else: 
                # Subcase B.1: Lk == 1 (Single word on the line)
                if lk == 1:
                    single_word = words_pk[0]
                    word_centroid_x = float(single_word.get('cx', 0))
                    word_centroid_y = float(single_word.get('cy', 0))
                    word_centroid = (word_centroid_x, word_centroid_y)
                    
                    best_col_j_star = -1
                    max_cosine_similarity = -2.0 # Cosine similarity ranges from -1 to 1

                    if all(hc is not None for hc in header_centroids):
                        for j, header_cent in enumerate(header_centroids):
                            if header_cent: # Check if header_cent is not None
                                similarity = self._calculate_centroid_cosine_similarity(word_centroid, header_cent)
                                if similarity > max_cosine_similarity:
                                    max_cosine_similarity = similarity
                                    best_col_j_star = j
                    
                    if best_col_j_star != -1:
                        current_row_cells[best_col_j_star]['words'] = [single_word]
                    else:
                        # Fallback: if no valid header centroids or other issue, place in first column
                        current_row_cells[0]['words'] = [single_word]
                        logger.warning(f"Line {k}, Word '{single_word.get('text_raw','N/A')}': Could not determine best column via centroid similarity. Placing in first column.")

                # Subcase B.2: 1 < Lk < H
                else: # (lk > 1 and lk < H)
                    for i in range(lk):
                        current_row_cells[i]['words'] = [words_pk[i]]
            
            # Populate cell_text for the current row
            for cell_idx in range(H):
                cell_words = current_row_cells[cell_idx]['words']
                if cell_words:
                    current_row_cells[cell_idx]['cell_text'] = " ".join([w.get('text_raw', '') for w in cell_words]).strip()
            
            table_matrix_T.append(current_row_cells)

        logger.info(f"GeometricTableStructurer: Successfully structured {len(table_matrix_T)} lines into {H} columns.")
        return table_matrix_T
        
        
    def _calculate_centroid_cosine_similarity(self, centroid1: Tuple[float, float], centroid2: Tuple[float, float]) -> float:
        """Calculates cosine similarity between two centroid vectors relative to origin (0,0)."""
        c1_x, c1_y = centroid1
        c2_x, c2_y = centroid2

        dot_product = c1_x * c2_x + c1_y * c2_y
        magnitude1 = math.sqrt(c1_x**2 + c1_y**2)
        magnitude2 = math.sqrt(c2_x**2 + c2_y**2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0  # Avoid division by zero; no similarity if one vector is zero
        
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return cosine_similarity