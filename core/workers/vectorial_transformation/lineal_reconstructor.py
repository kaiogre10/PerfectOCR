# PerfectOCR/core/workers/vectorial_transformation/linal_reconstructor.py
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from core.contracts.abstract_worker import VectorizationAbstractWorker
from domain.data_formatter import DataFormatter
from services.output_service import save_text_debug
from domain.class_models import KeyField
from utils.compiled_utils import space_removal
from core.assets.assets import SMALL_NUM

_small_num = SMALL_NUM

logger = logging.getLogger(__name__)

class LinealReconstructor(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get('lineal', {})
        self.overlap_threshold = worker_config.get('overlap_threshold')
        self.get_vectors = worker_config.get('get_vectors')
        self.output = config.get("reconstructed_lines", False)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time = time.perf_counter()
            logger.debug(f"Lineal: Estado de get_vectors: {self.get_vectors}")
            polygons = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.error("Sin poligonos")
                return False
                
            boundaries = self.find_tabular_lines(polygons)
            reconsturctued_lines = self.reconstruct_lines(polygons, boundaries)
            if reconsturctued_lines is None:
                logger.error("LinealReconstructor: Error al guardar lineas de texto en el workflowdict")
                return False
            
            lines_info, table_range = reconsturctued_lines
            logger.debug(f"'{len(lines_info)}' líneas amadas en {time.perf_counter() - start_time:.10f}")
            
            head, foot = table_range
            if foot is None or head is None:
                for line_data in lines_info.values():
                    line_data["tabular_line"] = False

            if manager.create_text_lines(lines_info):
            
                if head is not None and foot is not None:
                    # Hay tabla detectada → vectorizar solo si get_vectors está activo
                    table_lines = list(range(head + 1, foot))
                    logger.debug(f"Table range: {table_range}")
                    context["vectorice"] = self.get_vectors
                    context["table_range"] = table_lines
                else:
                    # No hay tabla detectada → siempre vectorizar
                    context["vectorice"] = True
                    context["table_range"] = []
                    
                if self.output:
                    file_name = manager.workflow.image_name if manager.workflow else "" # type: ignore
                    save_text_debug(lines_info, file_name)

                return True
                                            
        except Exception as e:
            logger.error(f"ERROR RECONSTRUYENDO LÍNEAS {e}", exc_info=True)
        return False
        
    def reconstruct_lines(self, polygons: Dict[str, Any], boundaries: Tuple[Set[int], Set[int]]) -> Optional[Tuple[Dict[str, Any], Tuple[Optional[int], Optional[int]]]]:
        """
        Reconstruye líneas agrupando polígonos y devuelve un dict con la debug completa de cada línea,
        incluyendo los textos OCR concatenados.
        """
        prepared_sorted = sorted(polygons.values(), key=lambda p: p.centroid[1])
        lines_info: Dict[str, Any] = {}
        current_line_polys: List[Any] = []
        current_line_bbox: Optional[List[float]] = None
        line_counter = 0
        headers = boundaries[0]
        footers = boundaries[1] if boundaries[0] else None
        bboxes: List[Any] = []
        lines_bbox: List[List[float]] = []
        header_idx: int = 0
        footer_idx: int = 0
        total_polys = len(prepared_sorted)
        cum_poly = 0

        for _, poly in enumerate(prepared_sorted):
            bbox = poly.bounding_box
            if len(bbox) == 0:
                total_polys -= 1
                continue

            bboxes.append(bbox)
            close_line = False

            if not current_line_polys or current_line_bbox is None:
                current_line_polys = [poly]
                current_line_bbox = bbox
            else:
                y1_min, y1_max = current_line_bbox[1], current_line_bbox[3]
                y2_min, y2_max = bbox[1], bbox[3]
                overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                min_h = min(y1_max - y1_min, y2_max - y2_min)
                overlap = overlap_abs / min_h if min_h > _small_num else 0.0

                if overlap > self.overlap_threshold:
                    current_line_polys.append(poly)
                    all_bboxes = [p.bounding_box for p in current_line_polys]
                    if all_bboxes:
                        all_xs = [b[0] for b in all_bboxes] + [b[2] for b in all_bboxes]
                        all_y_mins = [b[1] for b in all_bboxes]
                        all_y_maxs = [b[3] for b in all_bboxes]
                        avg_y_min = sum(all_y_mins) / len(all_y_mins)
                        avg_y_max = sum(all_y_maxs) / len(all_y_maxs)
                        current_line_bbox = [min(all_xs), avg_y_min, max(all_xs), avg_y_max]
                        
                        # Antes de cerrar la línea, ordena los polígonos actuales de la línea por el eje X (centroide[0])
                        current_line_polys.sort(key=lambda p: p.centroid[0])
                else:
                    close_line = True

            total_line_polys = len(current_line_polys)
            if cum_poly + total_line_polys == total_polys:
                close_line = True

            if not close_line:
                continue

            # Finaliza la línea actual y guarda la debug
            cum_poly += total_line_polys
            polygons_index = [p.poly_index for p in current_line_polys]

            header_line = line_counter if (headers and headers.intersection(polygons_index) and header_idx == 0) else None
            footer_line = line_counter if (footers and footers.intersection(polygons_index) and footer_idx == 0) else None

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

            joined_text = space_removal(" ".join(p.ocr_text or "" for p in current_line_polys))
             
            # Validar el texto antes de crear la entrada
            if not joined_text:
                if cum_poly == total_polys:
                    break
                # Si no es válido, iniciar una nueva línea sin incrementar el contador
                current_line_polys = [poly]
                current_line_bbox = bbox
                continue

            if not current_line_bbox:
                continue

            lines_bbox.append(current_line_bbox)  # Agregar aquí: bbox de la línea completada
            
            # El centroide de la línea se calcula como el centroide del bounding box de la línea
            line_centroid = [(current_line_bbox[0] + current_line_bbox[2]) / 2, (current_line_bbox[1] + current_line_bbox[3]) / 2] if current_line_bbox else [0, 0]
            polygon_ids = [p.polygon_id for p in current_line_polys]
            
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
                "tabular_line": tabular_line
            }
            
            if cum_poly == total_polys:
                break
                    
            line_counter += 1
            current_line_polys = [poly]
            current_line_bbox = bbox

        return (lines_info, (header_idx if headers else None, footer_idx if footer_idx > header_idx and footers else None))

    def find_tabular_lines(self, polygons: Dict[str, Any]) -> Tuple[Set[int], Set[int]]:
        """Método placeholder para encontrar líneas tabulares"""
        try:
            headers: Set[int] = set()
            footer: Set[int] = set()
            for poly_id, poly in polygons.items():
                key_field = poly.key_field or None
                if key_field is None:
                    continue

                polygon_index = poly.poly_index

                if KeyField.header.value in key_field:
                    logger.debug(f"Encabezado encontrado en: {poly_id}, idx: {polygon_index}")
                    headers.add(polygon_index)
                    continue

                elif any(k in (KeyField.total_doc.value, KeyField.total_art.value) for k in key_field):
                    footer.add(polygon_index)
                    logger.debug(f"Pie de tabla TOTAL MONETARIO encontrado en: {poly_id}, idx: {polygon_index}, key_field: {key_field}")
                    continue
                else:
                    continue

            return headers, footer
        except Exception as e:
            logger.warning(f"Error buscando límites: {e}")
            return set(), set()