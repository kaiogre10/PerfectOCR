# PerfectOCr/core/workers/ocr/fragmenter.py
import dataclasses
import logging
import re
import numpy as np
from typing import Dict, Any, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker

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

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("Fragmenter: No hay polígonos para procesar.")
            return False

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons
        blob_metrics: Dict[str, Any] = context.get('blob_metrics', {})
        
        final_polygons: List[Polygons] = []
        fragmented_count = 0

        sorted_poly_ids = sorted(polygons_in.keys())

        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            
            text_needs_frag = (
                self.worker_config and
                polygon.semantic_clasification not in ("numeric", "quantitative", "rfc", "umd") and
                " " in (polygon.ocr_text or "").strip()
            )

            punctuation_needs_frag = (
                self.worker_config and
                polygon.semantic_clasification not in ("numeric", "quantitative", "rfc", "umd") and
                not text_needs_frag and 
                (any(punct in (polygon.ocr_text or "") for punct in [";", ":", "!", "?"]) or 
                (polygon.ocr_text or "").count('.') == 1) 
            )

            poly_blob_metrics = blob_metrics.get(poly_id, {})
            visual_needs_frag = poly_blob_metrics.get('needs_fragmentation', False)
            quant_runs = []
            if polygon.semantic_clasification == "quantitative":
                quant_runs = self._quantitative_runs(polygon.ocr_text or "")
            quant_needs_frag = len(quant_runs) >= 2

            if text_needs_frag or visual_needs_frag or punctuation_needs_frag or quant_needs_frag:
                if visual_needs_frag and text_needs_frag:
                    reason = "visual y texto"
                elif visual_needs_frag:
                    reason = "visual"
                elif quant_needs_frag:
                    reason = "quantitativo"
                elif text_needs_frag:
                    reason = "texto"
                else:
                    reason = "puntuación"

                logger.debug(f"{poly_id}: MOTIVO: {reason}, TIPO: {polygon.semantic_clasification} = {polygon.ocr_text}")
                
                if visual_needs_frag:
                    fragments = self._fragment_by_blobs(polygon, poly_blob_metrics)
                elif quant_needs_frag:
                    fragments = self._fragment_by_quantitative(polygon, quant_runs)
                elif text_needs_frag:
                    fragments = self._fragment_by_text(polygon)
                else:
                    fragments = self._fragment_by_punctuation(polygon)

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
        poly_width = polygon.cropedd_geometry.croppy_dims.get('poly_width', {})
        poly_height = polygon.cropedd_geometry.croppy_dims.get('poly_height', {})
        
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
            
            logger.debug(f"Fragmento visual: texto='{frag_text}', bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=frag_text,
                was_refined=True
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
            
            logger.debug(f"Fragmentos: {part}")#, bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_refined=True
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
        
        point_count = text.count('.')
        if point_count == 1:
            # Si el tipo semántico ya es numérico o cuantitativo, no fragmentar.
            if polygon.semantic_clasification in ("numeric", "quantitative", "rfc", "umd"):
                return [polygon]

            dot_index = text.find('.')
            has_digits_before = dot_index > 0 and text[dot_index - 1].isdigit()
            has_digits_after = dot_index < len(text) - 1 and text[dot_index + 1].isdigit()

            # Si hay dígitos antes y después del punto, es probablemente un número
            if has_digits_before and has_digits_after:
                logger.debug(f"No fragmentando por punto: detectado potencial número '{text}'")
                return [polygon]

            parts = text.split('.')
            if len(parts) == 2 and all(p.strip() for p in parts):

                # Reconstruir sin incluir el punto en ninguna parte
                filtered_parts = [parts[0], parts[1]]
                
                # Crear fragmentos usando la misma lógica que ya existe
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
                    
                    logger.debug(f"Fragmento por punto único: texto='{part}', bbox={new_bbox.tolist()}")

                    new_poly = dataclasses.replace(
                        polygon,
                        geometry=new_geom,
                        ocr_text=part,
                        was_refined=True
                    )
                    new_polys.append(new_poly)
                    current_x = new_xmax

                return new_polys

        parts = re.split(r'([.,;:!?])', text)
        
        # Filtrar partes vacías y reconstruir con puntuación (excepto el punto)
        filtered_parts: List[str] = []
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            # Si la siguiente parte es puntuación (NO un punto), incluirla
            if i + 1 < len(parts) and parts[i + 1] in [";", ":", "!", "?"]:
                filtered_parts.append(part + parts[i + 1])
            # Si la parte actual NO es un signo de puntuación, añadirla sola
            elif part not in [".", ";", ":", "!", "?"]:
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

        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            part_text = part
            # Si la siguiente parte es puntuación (NO un punto), incluirla para el cálculo de ratio
            if i + 1 < len(parts) and parts[i + 1] in [";", ":", "!", "?"]:
                part_text += parts[i+1]
            
            # Si la parte actual es un signo de puntuación, saltarla
            if part_text in [".", ";", ":", "!", "?"]:
                continue

            part_ratio = len(part_text) / total_chars
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
                ocr_text=part, # Usar solo la parte de texto, sin la puntuación adjunta
                was_refined=True
            )
            new_polys.append(new_poly)
            current_x = new_xmax

        return new_polys
        
    def _fragment_by_quantitative(self, polygon: Polygons, quant_runs: List[Tuple[int, int, str]]) -> List[Polygons]:
        """
        Fragmenta un polígono que contiene múltiples tokens cuantitativos.
        - Si hay espacios: fragmenta por espacios (preserva todo el texto)
        - Si NO hay espacios: fragmenta por patrones cuantitativos (extrae solo números/monedas)
        """
        text: str = (polygon.ocr_text or "").strip()
        
        # ESTRATEGIA 1: Si hay espacios, fragmentar por espacios (como _fragment_by_text)
        if " " in text:
            parts = [p for p in text.split(' ') if p]
        else:
            # ESTRATEGIA 2: Si no hay espacios, fragmentar por patrones cuantitativos
            parts = [tok for _, _, tok in quant_runs]
        
        if len(parts) <= 1:
            return [polygon]

        char_lengths = [len(p) for p in parts]
        total_chars = sum(char_lengths) or 1

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

            logger.debug(f"Fragmento por cuantitativo: texto='{part}', bbox={new_bbox.tolist()}")

            new_polys.append(dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_refined=True
            ))
            current_x = new_xmax

        return new_polys
            
    def _quantitative_runs(self, s: str) -> List[tuple[int, int, str]]:
        s = (s or "").strip()
        if not s:
            return []
        currency = r"[$€£¥¢]"
        amount_body = r"(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?"
        # cuantitativo: decimales o con símbolo (sin %)
        quant_token = rf"{currency}\s*{amount_body}|{amount_body}\s*{currency}|{amount_body}"
        runs: List[tuple[int, int, str]] = []
        # Buscar todos los tokens cuantitativos
        for m in re.finditer(quant_token, s):
            tok = s[m.start():m.end()]
            if "%" in tok:
                continue
            is_decimal = bool(re.match(r"^\d+[.,]\d+$", tok) or re.match(r"^\d{1,3}(?:[.,]\d{3})+[.,]\d+$", tok))
            has_currency = bool(re.search(currency, tok))
            if is_decimal or has_currency:
                runs.append((m.start(), m.end(), tok))
        # Si hay más de un símbolo de divisa, dividir los tokens
        tokens = [tok for _, _, tok in runs]
        currency_count = sum(1 for t in tokens if re.search(currency, t))
        if currency_count > 1:
            # Dividir por cada símbolo de divisa encontrado
            split_tokens = re.findall(rf"{currency}\s*\d+(?:[.,]\d+)?", s)
            runs = []
            for match in split_tokens:
                start = s.find(match)
                end = start + len(match)
                runs.append((start, end, match))
        return runs
    