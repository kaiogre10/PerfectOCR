# PerfectOCr/core/workers/ocr/fragmenter.py
import dataclasses
import logging
import re
from typing import Dict, Any, List
import numpy as np
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from dataclasses import asdict

logger = logging.getLogger(__name__)

class Fragmenter(OCRAbstractWorker):
    """
    Fragmenta polígonos basados en señales textuales (espacios) y visuales (blobs).
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config = config
        self.worker_config = self.config.get('fragmenter', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get('separated_text', False)

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("Fragmenter: No hay polígonos para procesar.")
            return True

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons
        blob_metrics: Dict[str, Any] = context.get('blob_metrics', {})
        
        final_polygons: List[Polygons] = []
        fragmented_count = 0

        sorted_poly_ids = sorted(polygons_in.keys())

        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            
            text_needs_frag = (
                self.worker_config and
                polygon.semantic_type != "numeric" and
                " " in (polygon.ocr_text or "").strip()
            )

            punctuation_needs_frag = (
                self.worker_config and
                polygon.semantic_type != "numeric" and
                not text_needs_frag and  # Solo si no hay espacios
                any(punct in (polygon.ocr_text or "") for punct in [";", ":", "!", "?"])
            )

            poly_blob_metrics = blob_metrics.get(poly_id, {})
            visual_needs_frag = poly_blob_metrics.get('needs_fragmentation', False)

            if text_needs_frag or visual_needs_frag or punctuation_needs_frag:
                if visual_needs_frag and text_needs_frag:
                    reason = "visual y texto"
                elif visual_needs_frag:
                    reason = "visual"
                elif text_needs_frag:
                    reason = "texto"
                else:
                    reason = "puntuación"
                logger.debug(f"Fragmentando poly_id={poly_id} (motivo: {reason}). Texto original: '{polygon.ocr_text}'")
                
                fragments = self._fragment_polygon(polygon, poly_blob_metrics, punctuation_needs_frag)
                final_polygons.extend(fragments)
                if len(fragments) > 1:
                    fragmented_count += 1
            else:
                final_polygons.append(polygon)

        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(final_polygons):
            new_id = f"poly_{idx:04d}"
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id)
            final_polygons_dict[new_id] = final_poly_obj
        
            manager.workflow.polygons = final_polygons_dict
        
        if fragmented_count > 0:
            if self.output:
                self._save_ocr_raw(context, manager)

            logger.debug(f"Fragmenter: Se fragmentaron {fragmented_count} polígonos, resultando en {len(final_polygons_dict)} polígonos totales.")
            return True
        else:
            logger.debug(f"No se fragmentaron polígonos")
            return True

    def _fragment_polygon(self, polygon: Polygons, poly_blob_metrics: Dict[str, Any], punctuation_needs_frag: bool = False) -> List[Polygons]:
        # Prioridad 1: Si hay información visual fiable, se usa para los cortes
        if poly_blob_metrics.get('needs_fragmentation', False):
            return self._fragment_by_blobs(polygon, poly_blob_metrics)
        
        # Prioridad 2: Si hay espacios, se usa la información textual
        if " " in (polygon.ocr_text or "").strip():
            return self._fragment_by_text(polygon)
        
        # Prioridad 3: Si hay puntuación, se usa como delimitador
        if punctuation_needs_frag:
            return self._fragment_by_punctuation(polygon)
        
        return [polygon]

    def _fragment_by_blobs(self, polygon: Polygons, poly_blob_metrics: Dict[str, Any]) -> List[Polygons]:
        new_polys: List[Polygons] = []
        blobs_norm_boxes = poly_blob_metrics.get('blobs_norm_boxes', [])
        
        if not blobs_norm_boxes or len(blobs_norm_boxes) <= 1:
            return [polygon]

        pad_xmin, pad_ymin, _, _ = polygon.cropedd_geometry.padding_coords
        poly_width = polygon.cropedd_geometry.poly_dims['poly_width']
        poly_height = polygon.cropedd_geometry.poly_dims['poly_height']
        
        text_parts = (polygon.ocr_text or "").strip().split()

        for i, box_norm in enumerate(blobs_norm_boxes):
            xn1, yn1, xn2, yn2 = box_norm
            
            xmin_abs = pad_xmin + (xn1 * poly_width)
            xmax_abs = pad_xmin + (xn2 * poly_width)
            ymin_abs = pad_ymin + (yn1 * poly_height)
            ymax_abs = pad_ymin + (yn2 * poly_height)

            new_bbox = np.array([xmin_abs, ymin_abs, xmax_abs, ymax_abs])
            new_centroid = np.array([(xmin_abs + xmax_abs) / 2, (ymin_abs + ymax_abs) / 2])
            
            new_geom = dataclasses.replace(
                polygon.geometry,
                bounding_box=new_bbox,
                centroid=new_centroid,
                polygon_coords=np.array([
                    [new_bbox[0], new_bbox[1]],
                    [new_bbox[2], new_bbox[1]],
                    [new_bbox[2], new_bbox[3]],
                    [new_bbox[0], new_bbox[3]],
                ])
            )
            
            frag_text = text_parts[i] if i < len(text_parts) else ""
            
            logger.debug(f"-> Fragmento visual: texto='{frag_text}', bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=frag_text,
                was_fragmented=True
            )
            new_polys.append(new_poly)

        return new_polys

    def _fragment_by_text(self, polygon: Polygons) -> List[Polygons]:
        text: str = (polygon.ocr_text or "").strip()
        parts = [p for p in text.split(' ') if p]
        
        if len(parts) <= 1:
            return [polygon]

        char_lengths = [len(p) for p in parts]
        total_chars = sum(char_lengths)
        if total_chars == 0:
            return [polygon]

        xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
        width = xmax - xmin
        
        new_polys: List[Polygons] = []
        current_x = xmin

        for i, part in enumerate(parts):
            part_ratio = char_lengths[i] / total_chars
            part_width = part_ratio * width
            
            new_xmax = current_x + part_width
            
            new_bbox = np.array([current_x, ymin, new_xmax, ymax])
            new_centroid = np.array([(current_x + new_xmax) / 2, (ymin + ymax) / 2])

            new_geom = dataclasses.replace(
                polygon.geometry,
                bounding_box=new_bbox,
                centroid=new_centroid,
                polygon_coords=np.array([
                    [new_bbox[0], new_bbox[1]],
                    [new_bbox[2], new_bbox[1]],
                    [new_bbox[2], new_bbox[3]],
                    [new_bbox[0], new_bbox[3]],
                ])
            )
            
            logger.debug(f"Fragmento por texto: texto='{part}', bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_fragmented=True
            )
            new_polys.append(new_poly)
            current_x = new_xmax

        return new_polys

    def _fragment_by_punctuation(self, polygon: Polygons) -> List[Polygons]:
        """
        Fragmenta un polígono dividiendo por puntuación (.,;:!?) y creando polígonos separados.
        """
        text = (polygon.ocr_text or "").strip()
        if not text:
            return [polygon]

        # Dividir por puntuación pero mantener la puntuación en cada parte
        parts = re.split(r'([.,;:!?])', text)
        
        # Filtrar partes vacías y reconstruir con puntuación
        filtered_parts: List[str] = []
        for i, part in enumerate(parts):
            if part.strip():
                # Si la siguiente parte es puntuación, incluirla
                if i + 1 < len(parts) and parts[i + 1] in [";", ":", "!", "?"]:
                    filtered_parts.append(part + parts[i + 1])
                elif part not in [";", ":", "!", "?"]:
                    filtered_parts.append(part)
        
        if len(filtered_parts) <= 1:
            return [polygon]

        # Longitud de caracteres visibles para el cálculo proporcional
        char_lengths = [len(p) for p in filtered_parts]
        total_chars = sum(char_lengths)
        if total_chars == 0:
            return [polygon]

        xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
        width = xmax - xmin
        
        new_polys: List[Polygons] = []
        current_x = xmin

        for i, part in enumerate(filtered_parts):
            part_ratio = char_lengths[i] / total_chars
            part_width = part_ratio * width
            
            new_xmax = current_x + part_width
            
            new_bbox = np.array([current_x, ymin, new_xmax, ymax])
            new_centroid = np.array([(current_x + new_xmax) / 2, (ymin + ymax) / 2])

            new_geom = dataclasses.replace(
                polygon.geometry,
                bounding_box=new_bbox,
                centroid=new_centroid,
                polygon_coords=np.array([
                    [new_bbox[0], new_bbox[1]],
                    [new_bbox[2], new_bbox[1]],
                    [new_bbox[2], new_bbox[3]],
                    [new_bbox[0], new_bbox[3]],
                ])
            )
            
            logger.debug(f"Fragmento por puntuación: texto='{part}', bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_fragmented=True
            )
            new_polys.append(new_poly)
            current_x = new_xmax

        return new_polys

    def _save_ocr_raw(self, context: Dict[str, Any], manager: DataFormatter):
        logger.debug("OUTPUT PARA FRAGMENTADOR INICIADO")
        from services.output_service import save_json
        import os
        try:
            project_root = self.project_root
            file_name: str = manager.workflow.metadata.image_name
            output_paths = context.get("output_paths", [])
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            polygons_list = [asdict(poly) for poly in polygons.values()]
            logger.debug(f"{polygons_list}")
        except Exception as e:
            logger.debug(f"Error generando output: {e}", exc_info=True)
            for path in output_paths:
                output_dir: str = os.path.join(path, "separated_text")
                json_file_name = f"{os.path.splitext(file_name)[0]}.json"
                save_json(polygons_list, output_dir, json_file_name, project_root)
                if output_paths:
                    logger.debug(f"Texto Fragmentado para '{file_name}' guardado en {len(output_paths)} ubicaciones.")