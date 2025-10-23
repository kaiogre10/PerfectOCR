# PerfectOCr/core/utils/word_divider.py
import logging
import re
import numpy as np
from core.domain.data_models import Polygons
from typing import Dict, Any, List, Tuple
from core.utils.pattern_finder import find_quantitative_runs

logger = logging.getLogger(__name__)
        
def fragment(worker_config: Dict[str, Any], polygon: Polygons, blob_metrics: Dict[str, Any], sc: Tuple[str, bool]) -> List[Dict[str, Any]]:
    """
    Fragmenta un polígono en múltiples fragmentos (como diccionarios).
    NO modifica dataclasses. El Refiner reconstruye y reindexar.
    
    Returns:
        List[Dict[str, Any]]: Lista de diccionarios con datos de fragmentos.
                             Si no se fragmenta, retorna [datos del polígono original].
    """
    try:
        poly_id = polygon.polygon_id
        
        numeric = getattr(sc[0] , "numeric", False) or None
        quantitative = getattr(sc[0] , "quantitative", False) or None
        umd = getattr(sc[0] , "umd", False) or None
        text_needs_frag = (
            worker_config and
            not (numeric or quantitative or umd) and
            " " in (polygon.ocr_text or "").strip()
        )

        punctuation_needs_frag = (
            worker_config and
            not (numeric or quantitative or umd) and
            not text_needs_frag and 
            (any(punct in (polygon.ocr_text or "") for punct in [";", ":", "!", "?"]) or 
            (polygon.ocr_text or "").count('.') == 1) 
        )
        
        quant_runs = []
        if quantitative:
            quant_runs = find_quantitative_runs(polygon.ocr_text or "")
            
        quant_needs_frag = len(quant_runs) > 1
        visual_needs_frag: bool = blob_metrics.get('needs_fragmentation', False)

        if visual_needs_frag or text_needs_frag or punctuation_needs_frag or quant_needs_frag:
            if visual_needs_frag:
                reason = "visual"
            elif quant_needs_frag:
                reason = "quantitativo"
            elif text_needs_frag:
                reason = "texto"
            else:
                reason = "puntuación"

            # active_fields = [field for field in ['quantitative', 'umd', 'numeric', 'descriptive', 'code'] if getattr(sc[1], False)]
            logger.info(f"{poly_id}: '{polygon.ocr_text}' MOTIVO: {reason} ")
            
            if visual_needs_frag:
                fragments = fragment_by_blobs(polygon, blob_metrics)
            elif quant_needs_frag:
                fragments = fragment_by_quantitative(polygon, quant_runs)
            elif text_needs_frag:
                fragments = fragment_by_text(polygon)
            elif punctuation_needs_frag:
                fragments = fragment_by_punctuation(polygon, sc)
            else:
                # Retornar el polígono original como diccionario
                fragments = [_polygon_to_dict(polygon)]

            return fragments
        else:
            # No se fragmenta, retornar el polígono original como diccionario
            return [_polygon_to_dict(polygon)]
            
    except Exception as e:
        logger.warning(f"Error fragmentando: {e}", exc_info=True)
        return [_polygon_to_dict(polygon)]


def _polygon_to_dict(polygon: Polygons) -> Dict[str, Any]:
    """Convierte un polígono a diccionario simple para que el Refiner lo reconstruya."""
    return {
        'polygon': polygon,  # Referencia al polígono original
        'geometry_updates': None,  # Sin cambios de geometría
        'text_updates': None,  # Sin cambios de texto
        'was_refined': False
    }

def fragment_by_blobs(polygon: Polygons, blob_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fragmenta por blobs visuales. Retorna lista de diccionarios."""
    fragments: List[Dict[str, Any]] = []
    blobs_norm_boxes = blob_metrics.get('blobs_norm_boxes', [])
    
    if not blobs_norm_boxes or len(blob_metrics) < 2:
        return [_polygon_to_dict(polygon)]

    pad_xmin, pad_ymin, _, _ = polygon.cropedd_geometry.padding_coords
    poly_width = int(polygon.cropedd_geometry.croppy_dims.get('poly_width', 0))
    poly_height = int(polygon.cropedd_geometry.croppy_dims.get('poly_height', 0))
    
    if poly_width <= 0 or poly_height <= 0:
        logger.warning("Dimensiones de recorte inválidas para fragmentación por blobs.")
        return [_polygon_to_dict(polygon)]
    
    text_parts = (polygon.ocr_text or "").strip().split()
    num_blobs = blob_metrics.get('num_blobs')
    
    if len(text_parts) != num_blobs:
        logger.debug(f"No se fragmenta: blobs/palabras {num_blobs} / {len(text_parts)}")
        return [_polygon_to_dict(polygon)]

    for i, box_norm in enumerate(blobs_norm_boxes):
        xn1, yn1, xn2, yn2 = box_norm
        
        xmin_abs = pad_xmin + (xn1 * poly_width)
        xmax_abs = pad_xmin + (xn2 * poly_width)
        ymin_abs = pad_ymin + (yn1 * poly_height)
        ymax_abs = pad_ymin + (yn2 * poly_height)

        new_bbox = np.array([xmin_abs, ymin_abs, xmax_abs, ymax_abs])
        new_centroid = np.array([(xmin_abs + xmax_abs) / 2, (ymin_abs + ymax_abs) / 2])
        new_coords = np.array([
            [new_bbox[0], new_bbox[1]],
            [new_bbox[2], new_bbox[1]],
            [new_bbox[2], new_bbox[3]],
            [new_bbox[0], new_bbox[3]],
        ])
        
        frag_text = text_parts[i] if i < len(text_parts) else ""
        logger.info(f"Fragmento visual: texto='{frag_text}', bbox={new_bbox.tolist()}")

        fragments.append({
            'polygon': polygon,
            'geometry_updates': {
                'bounding_box': new_bbox,
                'centroid': new_centroid,
                'polygon_coords': new_coords
            },
            'text_updates': {'ocr_text': frag_text},
            'was_refined': True
        })

    return fragments


def fragment_by_text(polygon: Polygons) -> List[Dict[str, Any]]:
    """Fragmenta por espacios. Retorna lista de diccionarios."""
    text: str = (polygon.ocr_text or "").strip()
    parts = [p for p in text.split(' ') if p]
    
    if len(parts) <= 1:
        return [_polygon_to_dict(polygon)]

    char_lengths = [len(p) for p in parts]
    total_chars = sum(char_lengths)
    if total_chars == 0:
        return [_polygon_to_dict(polygon)]

    xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
    width = xmax - xmin
    
    fragments: List[Dict[str, Any]] = []
    current_x = xmin

    for i, part in enumerate(parts):
        part_ratio = char_lengths[i] / total_chars
        part_width = part_ratio * width
        new_xmax = current_x + part_width
        
        new_bbox = np.array([current_x, ymin, new_xmax, ymax])
        new_centroid = np.array([(current_x + new_xmax) / 2, (ymin + ymax) / 2])
        new_coords = np.array([
            [new_bbox[0], new_bbox[1]],
            [new_bbox[2], new_bbox[1]],
            [new_bbox[2], new_bbox[3]],
            [new_bbox[0], new_bbox[3]],
        ])
        
        logger.info(f"Fragmentos: {part}")

        fragments.append({
            'polygon': polygon,
            'geometry_updates': {
                'bounding_box': new_bbox,
                'centroid': new_centroid,
                'polygon_coords': new_coords
            },
            'text_updates': {'ocr_text': part},
            'was_refined': True
        })
        current_x = new_xmax

    return fragments


def fragment_by_punctuation(polygon: Polygons, sc: Tuple[str, bool]) -> List[Dict[str, Any]]:
    """Fragmenta por puntuación. Retorna lista de diccionarios."""
    numeric = getattr(sc[0] , "numeric", False) or None
    quantitative = getattr(sc[0] , "quantitative", False) or None
    umd = getattr(sc[0] , "umd", False) or None
    text = (polygon.ocr_text or "").strip()
    if not text:
        return [_polygon_to_dict(polygon)]
    
    point_count = text.count('.')
    if point_count == 1:
        if numeric or quantitative or umd:
            return [_polygon_to_dict(polygon)]

        dot_index = text.find('.')
        has_digits_before = dot_index > 0 and text[dot_index - 1].isdigit()
        has_digits_after = dot_index < len(text) - 1 and text[dot_index + 1].isdigit()

        if has_digits_before and has_digits_after:
            return [_polygon_to_dict(polygon)]

        parts = text.split('.')
        if len(parts) == 2 and all(p.strip() for p in parts):
            filtered_parts = [parts[0], parts[1]]
            
            char_lengths = [len(p) for p in filtered_parts]
            total_chars = sum(char_lengths)
            if total_chars == 0:
                return [_polygon_to_dict(polygon)]

            xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
            width = xmax - xmin
            
            fragments = []
            current_x = xmin

            for i, part in enumerate(filtered_parts):
                part_ratio = char_lengths[i] / total_chars
                part_width = part_ratio * width
                new_xmax = current_x + part_width
                
                new_bbox = np.array([current_x, ymin, new_xmax, ymax])
                new_centroid = np.array([(current_x + new_xmax) / 2, (ymin + ymax) / 2])
                new_coords = np.array([
                    [new_bbox[0], new_bbox[1]],
                    [new_bbox[2], new_bbox[1]],
                    [new_bbox[2], new_bbox[3]],
                    [new_bbox[0], new_bbox[3]],
                ])
                
                logger.debug(f"Fragmento por punto único: texto='{part}', bbox={new_bbox.tolist()}")

                fragments.append({
                    'polygon': polygon,
                    'geometry_updates': {
                        'bounding_box': new_bbox,
                        'centroid': new_centroid,
                        'polygon_coords': new_coords
                    },
                    'text_updates': {'ocr_text': part},
                    'was_refined': True
                })
                current_x = new_xmax

            return fragments

    parts = re.split(r'([.,;:!?])', text)
    
    filtered_parts: List[str] = []
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        
        if i + 1 < len(parts) and parts[i + 1] in [";", ":", "!", "?"]:
            filtered_parts.append(part + parts[i + 1])
        elif part not in [".", ";", ":", "!", "?"]:
            filtered_parts.append(part)
    
    if len(filtered_parts) <= 1:
        return [_polygon_to_dict(polygon)]

    char_lengths = [len(p) for p in filtered_parts]
    total_chars = sum(char_lengths)
    if total_chars == 0:
        return [_polygon_to_dict(polygon)]

    xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
    width = xmax - xmin
    
    fragments: List[Dict[str, Any]] = []
    current_x = xmin

    for i, part in enumerate(parts):
        if not part.strip():
            continue
        
        part_text = part
        if i + 1 < len(parts) and parts[i + 1] in [";", ":", "!", "?"]:
            part_text += parts[i+1]
        
        if part_text in [".", ";", ":", "!", "?"]:
            continue

        part_ratio = len(part_text) / total_chars
        part_width = part_ratio * width
        new_xmax = current_x + part_width
        
        new_bbox = np.array([current_x, ymin, new_xmax, ymax])
        new_centroid = np.array([(current_x + new_xmax) / 2, (ymin + ymax) / 2])
        new_coords = np.array([
            [new_bbox[0], new_bbox[1]],
            [new_bbox[2], new_bbox[1]],
            [new_bbox[2], new_bbox[3]],
            [new_bbox[0], new_bbox[3]],
        ])
        
        logger.info(f"Fragmento por puntuación: texto='{part}', bbox={new_bbox.tolist()}")

        fragments.append({
            'polygon': polygon,
            'geometry_updates': {
                'bounding_box': new_bbox,
                'centroid': new_centroid,
                'polygon_coords': new_coords
            },
            'text_updates': {'ocr_text': part},
            'was_refined': True
        })
        current_x = new_xmax

    return fragments

    
def fragment_by_quantitative(polygon: Polygons, quant_runs: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
    """Fragmenta por tokens cuantitativos. Retorna lista de diccionarios."""
    text: str = (polygon.ocr_text or "").strip()
    
    if " " in text:
        parts = [p for p in text.split(' ') if p]
    else:
        parts = [tok for _, _, tok in quant_runs]
    
    if len(parts) <= 1:
        return [_polygon_to_dict(polygon)]

    char_lengths = [len(p) for p in parts]
    total_chars = sum(char_lengths) or 1

    xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
    width = xmax - xmin

    fragments: List[Dict[str, Any]] = []
    current_x = xmin

    for i, part in enumerate(parts):
        part_ratio = char_lengths[i] / total_chars
        part_width = part_ratio * width
        new_xmax = current_x + part_width

        new_bbox = np.array([current_x, ymin, new_xmax, ymax])
        new_centroid = np.array([(current_x + new_xmax) / 2, (ymin + ymax) / 2])
        new_coords = np.array([
            [new_bbox[0], new_bbox[1]],
            [new_bbox[2], new_bbox[1]],
            [new_bbox[2], new_bbox[3]],
            [new_bbox[0], new_bbox[3]],
        ])

        logger.debug(f"Fragmento por cuantitativo: texto='{part}', bbox={new_bbox.tolist()}")

        fragments.append({
            'polygon': polygon,
            'geometry_updates': {
                'bounding_box': new_bbox,
                'centroid': new_centroid,
                'polygon_coords': new_coords
            },
            'text_updates': {'ocr_text': part},
            'was_refined': True
        })
        current_x = new_xmax

    return fragments