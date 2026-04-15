# PerfectOCR/core/workers/vectorial_transformation/linal_reconstructor.py
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from services.output_service import save_raw_json

logger = logging.getLogger(__name__)

class LinealReconstructor(VectorizationAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('lineal', {})
        self.overlap_threshold = worker_config.get('overlap_threshold')
        self.get_vectors = worker_config.get('get_vectors')
        self.output = config.get("reconstructed_lines", False)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time = time.perf_counter()
            logger.debug(f"Lineal: Estado de get_vectors: {self.get_vectors}")
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False
            
            reconsturctued_lines = self._reconstruct_lines(polygons)
            if reconsturctued_lines is None:
                logger.error("LinealReconstructor: Error al guardar lineas de texto en el workflowdict")
                return False
            
            lines_info, table_range = reconsturctued_lines
            logger.debug(f"'{len(lines_info)}' líneas amadas en {time.perf_counter() - start_time:.10f}")

            if manager.create_text_lines(lines_info):
                logger.debug(f"Lineas guardads correctamente en el manager")
            
                head, foot = table_range
                if head < 1 and foot < 1:
                # No hay tabla detectada → siempre vectorizar
                    context["vectorice"] = True
                    context["table_range"] = []
                else:
                    # Hay tabla detectada → vectorizar solo si get_vectors está activo
                    table_lines = list(range(head + 1, foot))
                    # logger.info(f"Table range: {table_range}")
                    context["vectorice"] = self.get_vectors
                    context["table_range"] = table_lines

                if self.output:
                    file_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    worker_name = context.get("worker_name") or "lineal"
                    output_paths = context["output_paths"]                    
                    save_raw_json(output_paths, worker_name, lines_info, file_name)

                return True
                                            
        except Exception as e:
            logger.error(f"error {e}", exc_info=True)
        return False
        
    def _reconstruct_lines(self, polygons: Dict[str, Polygons]) -> Optional[Tuple[Dict[str, Any], Tuple[int, int]]]:
        """
        Reconstruye líneas agrupando polígonos y devuelve un dict con la debug completa de cada línea,
        incluyendo los textos OCR concatenados.
        """
        prepared_sorted = sorted(polygons.values(), key=lambda p: p.geometry.centroid[1])
        lines_info: Dict[str, Any] = {}
        current_line_polys: List[Polygons] = []
        current_line_bbox: Optional[List[float]] = None
        line_counter = 0
        boundaries = self.find_tabular_lines(polygons)
        headers = set(boundaries[0])
        footers = set(boundaries[1]) if boundaries[0] else None
        bboxes: List[np.ndarray[Any, Any]] = []
        lines_bbox: List[Any] = []
        header_idx: int = 0
        footer_idx: int = 0
        
        for poly in prepared_sorted:
            bbox = poly.geometry.bounding_box
            if bbox.size == 0:
                continue

            bboxes.append(bbox)

            if not current_line_polys or current_line_bbox is None:
                current_line_polys = [poly]
                current_line_bbox = list(bbox)
            else:
                y1_min, y1_max = current_line_bbox[1], current_line_bbox[3]
                y2_min, y2_max = bbox[1], bbox[3]
                overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                min_h = min(y1_max - y1_min, y2_max - y2_min)
                overlap = overlap_abs / min_h if min_h > 1e-5 else 0.0

                if overlap > self.overlap_threshold:
                    current_line_polys.append(poly)
                    all_bboxes = [p.geometry.bounding_box for p in current_line_polys]
                    if all_bboxes:
                        all_xs = [b[0] for b in all_bboxes] + [b[2] for b in all_bboxes]
                        all_y_mins = [b[1] for b in all_bboxes]
                        all_y_maxs = [b[3] for b in all_bboxes]
                        avg_y_min = sum(all_y_mins) / len(all_y_mins)
                        avg_y_max = sum(all_y_maxs) / len(all_y_maxs)
                        current_line_bbox = [min(all_xs), avg_y_min, max(all_xs), avg_y_max]
                        
                        # Antes de cerrar la línea, ordena los polígonos actuales de la línea por el eje X (centroide[0])
                        current_line_polys.sort(key=lambda p: p.geometry.centroid[0])
                else:
                    # Finaliza la línea actual y guarda la debug
                    polygon_ids = [p.polygon_id for p in current_line_polys]
                    polygons_index = [p.poly_index for p in current_line_polys]
                    texts = [p.ocr_text or "" for p in current_line_polys]

                    header_line = line_counter if (headers and headers.intersection(set(polygons_index)) and header_idx == 0) else None
                    footer_line = line_counter if (footers and footers.intersection(set(polygons_index)) and footer_idx == 0) else None
                     
                    tabular_line: bool = False

                    if header_line is not None:
                        header_idx = header_line  # Asignación directa, no suma
                        tabular_line = False

                    elif footer_line is not None:
                        # Validar que el footer aparezca DESPUÉS del header
                        if header_idx > 0 and line_counter > header_idx:
                            footer_idx = footer_line
                        else:
                            footer_line = None # Invalida si aparece antes del header
                        tabular_line = False
                    
                    elif header_idx > 0 and footer_idx == 0:
                        # Si ya pasamos el header y no hay footer, es tabla
                        tabular_line = True
                        
                    else:
                        tabular_line = False

                    joined_text = " ".join(texts).strip()

                    # Validar el texto antes de crear la entrada
                    if not joined_text:
                        # Si no es válido, iniciar una nueva línea sin incrementar el contador
                        current_line_polys = [poly]
                        current_line_bbox = list(bbox)
                        continue
                    
                    line_t_cuant = sum((p.cuant_chars or 0) for p in current_line_polys) if poly.key_field is not None or not 0 in poly.semantic_clasification else 0
                    
                    lines_bbox.append(current_line_bbox)  # Agregar aquí: bbox de la línea completada
                    
                    # El centroide de la línea se calcula como el centroide del bounding box de la línea
                    line_centroid = [           
                            (current_line_bbox[0] + current_line_bbox[2]) / 2,
                            (current_line_bbox[1] + current_line_bbox[3]) / 2
                    ] if current_line_bbox else [0, 0]
                    
                    line_id = f"line_{line_counter:04d}"
                    lines_info[line_id] = {
                        "text": joined_text,
                        "line_index": line_counter,
                        "line_bbox": current_line_bbox,
                        "line_centroid": line_centroid,
                        "polygon_ids": polygon_ids,
                        "polygons_index": polygons_index,
                        "header_line": header_line,
                        "footer_line": footer_line,
                        "tabular_line": tabular_line,
                        "t_cuant": line_t_cuant
                    }
                            
                    line_counter += 1
                    current_line_polys = [poly]
                    current_line_bbox = list(bbox)
                
                    # if header_line is not None:
                    #     logger.info(f"{line_id}: ENCABEZADO '{joined_text}' | {polygon_ids} ")
                        
                    # if footer_line is not None:
                    #     logger.info(f"{line_id}: FOOTER '{joined_text}' | {polygon_ids}")

                    # if tabular_line:
                    #     logger.info(f"{line_id}: '{joined_text}' TABULAR")

        # Finaliza la última línea
        if current_line_polys:
            polygon_ids = [p.polygon_id for p in current_line_polys]
            polygons_index = [p.poly_index for p in current_line_polys]
            texts = [p.ocr_text or "" for p in current_line_polys]
            joined_text = " ".join(texts).strip()
            
            footer_line = line_counter if (footers and footers.intersection(set(polygons_index)) and footer_idx == 0) else None
            
            if footer_line is not None:
                if header_idx > 0 and line_counter > header_idx:
                    footer_idx = footer_line
                else:
                    footer_line = None
                tabular_line = False
            elif header_idx > 0 and footer_idx == 0:
                tabular_line = True
            else:
                tabular_line = False
            
            # Validar también el texto de la última línea
            if joined_text:
                current_line_polys.sort(key=lambda p: p.geometry.centroid[0])
                line_t_cuant = sum((p.cuant_chars or 0) for p in current_line_polys) if poly.key_field is not None or not 0 in poly.semantic_clasification else 0
                lines_bbox.append(current_line_bbox)
                
                line_centroid = [
                    (current_line_bbox[0] + current_line_bbox[2]) / 2,
                    (current_line_bbox[1] + current_line_bbox[3]) / 2
                ] if current_line_bbox else [0, 0]
                
                line_id = f"line_{line_counter:04d}"
                lines_info[line_id] = {
                    "text": joined_text,
                    "line_index": line_counter,
                    "line_bbox": current_line_bbox,
                    "line_centroid": line_centroid,
                    "polygon_ids": polygon_ids,
                    "polygons_index": polygons_index,
                    "header_line": None,
                    "footer_line": footer_line,
                    "tabular_line": tabular_line,
                    "t_cuant": line_t_cuant
                }

                # if footer_line is not None:
                #     logger.info(f"{line_id}: FOOTER '{joined_text}' | {polygon_ids}")
                
                # if tabular_line:
                #     logger.info(f"{line_id}: '{joined_text}' TABULAR")

        return lines_info, (header_idx if header_idx > 0 else 0, footer_idx if footer_idx > 0 else 0)

    def find_tabular_lines(self, polygons: Dict[str, Polygons]) -> Tuple[List[int], List[int]]:
        """
        Método placeholder para encontrar líneas tabulares.
        """
        try:
            headers: List[int] = []
            footer: List[int] = []
            for _, poly in polygons.items():
                key_field = poly.key_field
                if key_field is None:
                    continue

                keys = key_field if isinstance(key_field, list) else [key_field]
                polygon_index = poly.poly_index

                if 6 in keys:
                    # logger.info(f"Encabezado encontrado en: {poly_id}, idx: {polygon_index}")
                    headers.append(polygon_index)
                    continue

                elif 1 in keys:
                    footer.append(polygon_index)
                    # logger.info(f"Pie de tabla TOTAL MONETARIO encontrado en: {poly_id}, idx: {polygon_index}, key_field: {key_field}")
                    continue
                
                elif 2 in keys:
                    footer.append(polygon_index)
                    # logger.debug(f"Pie de tabla TOTALES CANTIDAD encontrado en: {poly_id}, idx: {polygon_index}, key_field: {key_field}")
                    continue

                else:
                    continue

            table_boundaries: Tuple[List[int], List[int]] = headers, footer
            # logger.info(f"Límites de la tabla: {table_boundaries}")

            return table_boundaries
        except Exception as e:
            logger.warning(f"Error buscando límites: {e}", exc_info=True)
        return [], []
