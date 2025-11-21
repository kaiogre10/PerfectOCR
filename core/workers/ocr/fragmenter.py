# PerfectOCr/core/workers/ocr/fragmenter.py
import dataclasses
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.pattern_finder import is_acronym, find_quantitative_runs, separate_punt
from core.utils.text_validator import validate_text, punc_chars

logger = logging.getLogger(__name__)

class Fragmenter(OCRAbstractWorker):
    """Fragmenta polígonos basados en señales textuales (espacios) y visuales (blobs)"""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('fragmenter', {})
        self.min_contours_for_frag = self.worker_config.get("min_contours_for_frag")
        self.output = config.get("fragmented_polys", False)
        self.punc_chars: List[str] = punc_chars()

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool: 
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Fragmentador no tiene polígonos para procesar")
                return False
            
            polygons_in: Dict[str, Polygons] = manager.workflow.polygons
            sorted_poly_ids = sorted(polygons_in.keys())
            blob_metrics = context.get("blob_metrics", {})
            logger.info(f"Cantidad de polígonos recibidos:{len(sorted_poly_ids)}")
            fragmented_count = 0
            final_polygons: List[Polygons] = []
            
            for poly_id in sorted_poly_ids:
                polygon = polygons_in[poly_id]
                poly_blob_metrics = blob_metrics.get(poly_id, {})

                if not poly_blob_metrics:
                    logger.debug(f"Sin Metricas para: {poly_id}")
                    final_polygons.append(polygon)
                    continue
                    
                # logger.info(f"{poly_id}: cantidad de blob ='{blob_metrics.get("num_blobs")}'")
                
                sc: List[int] | int = polygon.semantic_clasification or 0
                ocr_text: str = polygon.ocr_text or ""
                
                if not validate_text(ocr_text): 
                    logger.debug(f"Polygono sin texto: {poly_id}")
                    continue

                # Si el texto corresponde a una sigla (p.e. 'P.U.C.D', 'I.V.A.') se conserva intacto
                if is_acronym(ocr_text):
                    logger.warning(f"{poly_id} no fragmentando sigla detectada: '{ocr_text}'")
                    final_polygons.append(polygon)
                    continue

                text_needs_frag = (
                    not (all(cls in (1, 2, -2) for cls in sc) if isinstance(sc, list) else sc in (1, 2, -2)) and
                    " " in (ocr_text or "").strip()
                )

                punctuation_needs_frag = (
                    not (all(cls in (1, 2, -2) for cls in sc) if isinstance(sc, list) else sc in (1, 2, -2)) and
                    not text_needs_frag and
                    (any(punct in (ocr_text or "") for punct in [";", ":", "!", "?"]) or
                    (ocr_text or "").count('.') == 1)
                )

                quant_runs = []
                if (isinstance(sc, list) and any(cls == 2 for cls in sc)) or (isinstance(sc, int) and sc == 2):
                    quant_runs = find_quantitative_runs(ocr_text)

                quant_needs_frag = len(quant_runs) > 1

                if not poly_blob_metrics:
                    visual_needs_frag = False
                    
                visual_needs_frag: bool = poly_blob_metrics.get('needs_fragmentation', False)

                if visual_needs_frag or text_needs_frag or punctuation_needs_frag or quant_needs_frag:

                    if visual_needs_frag:
                        reason = "visual"
                    elif quant_needs_frag:
                        reason = "quantitativo"
                    elif text_needs_frag:
                        reason = "texto"
                    else:
                        reason = "puntuación"

                    semantic_type_name = self.get_semantic_type_name(sc, manager)
                    logger.debug(f"{poly_id}: MOTIVO: {reason}= {ocr_text} | Tipo: {semantic_type_name}")

                    if visual_needs_frag:
                        fragments = self.fragment_by_blobs(polygon, poly_blob_metrics)
                    elif quant_needs_frag:
                        fragments = self.fragment_by_quantitative(polygon, quant_runs, poly_blob_metrics)
                    elif text_needs_frag:
                        fragments = self.fragment_by_text(polygon)
                    elif punctuation_needs_frag:
                        fragments = self.fragment_by_punctuation(polygon)
                    else:
                        fragments = [polygon]
                     
                    final_polygons.extend(fragments)
                    if len(fragments) > 1:
                        fragmented_count += 1

                        if self.output:
                            from services.output_service import save_croped_image
                            for frag_poly in fragments:
                                worker_name = "fragmenter"
                                image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                                output_paths = context["output_paths"]
                                cropped_image = frag_poly.cropped_img.cropped_img # type: ignore
                                poly_id = frag_poly.polygon_id or ""
                                save_croped_image(image_name, poly_id, cropped_image, output_paths, worker_name, method=None) # type: ignore
                else:
                    final_polygons.append(polygon)

            final_polygons_dict: Dict[str, Polygons] = {}
            for idx, poly_obj in enumerate(final_polygons):
                new_id = f"poly_{idx:04d}"
                final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id)
                final_polygons_dict[new_id] = final_poly_obj
                manager.workflow.polygons = final_polygons_dict
            
            if fragmented_count > 0:
                logger.info(f"Fragmenter: Se fragmentaron {fragmented_count} resultando en {len(final_polygons_dict)} polígonos totales.")
                return True
                
        except Exception as e:
            logger.warning(f"Error fragmentando: {e}", exc_info=True)
        return False

    def fragment_by_blobs(self, polygon: Polygons, blob_metrics: Dict[str, Any]) -> List[Polygons]:
        """
        Fragmenta un polígono 1-a-1 con cada blob detectado,
        garantizando geometría precisa (sin solapes ni desfases).
        """
        new_polys: List[Polygons] = []
        blobs_norm_boxes = blob_metrics["blobs_norm_boxes"]
        num_blobs = blob_metrics.get("num_blobs", 0)
        text = polygon.ocr_text or ""
        text_parts = (text or "").strip().split()
        
        if not validate_text(text):
            return [polygon]

        # Debe existir correspondencia exacta blobs ↔ palabras
        if not blobs_norm_boxes or num_blobs < self.min_contours_for_frag or len(text_parts) != num_blobs:
            logger.debug(f"No se fragmenta: blobs/palabras {num_blobs}/{len(text_parts)}")
            return [polygon]

        # Padding coords = referencia absoluta del recorte
        pad_xmin, pad_ymin, _, _ = polygon.cropedd_geometry.padding_coords
        poly_width  = polygon.cropedd_geometry.croppy_dims.get("poly_width", 0)
        poly_height = polygon.cropedd_geometry.croppy_dims.get("poly_height", 0)

        if poly_width < 1 or poly_height < 1:
            logger.warning("Fragmenter: Dimensiones de recorte inválidas.")
            return [polygon]

        for i, (xn1, yn1, xn2, yn2) in enumerate(blobs_norm_boxes):
            # Clamp y conversión segura
            xn1 = float(np.clip(xn1, 0.0, 1.0))
            yn1 = float(np.clip(yn1, 0.0, 1.0))
            xn2 = float(np.clip(xn2, 0.0, 1.0))
            yn2 = float(np.clip(yn2, 0.0, 1.0))

            xmin_abs = round(pad_xmin + xn1 * poly_width)
            xmax_abs = round(pad_xmin + xn2 * poly_width)
            ymin_abs = round(pad_ymin + yn1 * poly_height)
            ymax_abs = round(pad_ymin + yn2 * poly_height)

            new_bbox = np.array([xmin_abs, ymin_abs, xmax_abs, ymax_abs], dtype=np.float32)
            new_centroid = np.array([(xmin_abs + xmax_abs) / 2, (ymin_abs + ymax_abs) / 2], dtype=np.float32)

            new_geom = dataclasses.replace(
                polygon.geometry,
                bounding_box=new_bbox,
                centroid=new_centroid,
                polygon_coords=np.array([
                    [new_bbox[0], new_bbox[1]],
                    [new_bbox[2], new_bbox[1]],
                    [new_bbox[2], new_bbox[3]],
                    [new_bbox[0], new_bbox[3]],
                ], dtype=np.float32)
            )

            frag_text = text_parts[i]  # 1-a-1 con blobs
            
            logger.info(f"Fragmento visual: texto='{frag_text}'") #, bbox={new_bbox.tolist()}")
            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=frag_text,
                was_refined=True
            )
            new_polys.append(new_poly)

        return new_polys

    def fragment_by_text(self, polygon: Polygons) -> List[Polygons]:
        text: str = (polygon.ocr_text or "").strip()
        if not validate_text(text):
            return [polygon]
            
        parts = [p for p in text.split(' ') if p]
        
        if len(parts) < self.min_contours_for_frag:
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
            
            logger.debug(f"Fragmentos textuales: '{part}'")#, bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_refined=True
            )
            new_polys.append(new_poly)
            current_x = new_xmax

        logger.debug(f"'{len(parts)}' Fragmentos totales para {polygon.polygon_id}")
        return new_polys

    def fragment_by_punctuation(self, polygon: Polygons) -> List[Polygons]:
        """
        Fragmenta un polígono dividiendo por puntuación (.,;:!?) y creando polígonos separados.
        """
        text = (polygon.ocr_text or "").strip()
        if not validate_text(text):
            return [polygon]

        sc: List[int] | int = polygon.semantic_clasification
        if (isinstance(sc, list) and all(cls in (1, 2, -2, -1) for cls in sc)) or \
           (isinstance(sc, int) and sc in (1, 2, -2, -1)):
            return [polygon]
        
        point_count = text.count('.')
        if point_count == 1:
            dot_index = text.find('.')
            has_digits_before = dot_index > 0 and text[dot_index - 1].isdigit()
            has_digits_after = dot_index < len(text) - 1 and text[dot_index + 1].isdigit()

            # Si hay dígitos antes y después del punto, es probablemente un número
            if has_digits_before and has_digits_after:
                logger.info(f"No fragmentando por punto: detectado potencial número '{text}'")
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
                
                new_polys = []
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
                    
                    logger.debug(f"División por puntuación: texto='{part}'")#, bbox={new_bbox.tolist()}")

                    new_poly = dataclasses.replace(
                        polygon,
                        geometry=new_geom,
                        ocr_text=part,
                        was_refined=True
                    )
                    new_polys.append(new_poly)
                    current_x = new_xmax

                return new_polys

        parts = separate_punt(text)
        
        # Filtrar partes vacías y reconstruir con puntuación (excepto el punto)
        filtered_parts: List[str] = []
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            # Si la siguiente parte es puntuación (NO un punto), incluirla
            if i + 1 < len(parts) and parts[i + 1] in [";", ":", "!", "?"]:
                filtered_parts.append(part + parts[i + 1])
            # Si la parte actual NO es un signo de puntuación, añadirla sola
            elif part not in self.punc_chars:
                filtered_parts.append(part)
        
        if len(filtered_parts) < 2:
            logger.debug(f"No se fragmenta por puntuación '{text}', no hay suficientes partes válidas.")
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
            if part_text in self.punc_chars:
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
            
            logger.info(f"Fragmento por puntuación: texto='{part}'")#, bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_refined=True
            )
            new_polys.append(new_poly)
            current_x = new_xmax

        return new_polys
        
    def fragment_by_quantitative(self, polygon: Polygons, quant_runs: List[Tuple[int, int, str]], blob_metrics: Dict[str, Any] | None = None) -> List[Polygons]:
        """
        Fragmenta un polígono que contiene múltiples tokens cuantitativos Y/O descriptivos,
        preservando TODOS los fragmentos de texto.
        Utiliza los índices de 'quant_runs' para segmentar el texto original.
        """
        text: str = (polygon.ocr_text or "").strip()
        if not validate_text(text) or not quant_runs:
            return [polygon]
        
        parts: List[str] = []
        last_index = 0

        for start, end, token in quant_runs:
            # 1. Añadir el fragmento DESCRIPTIVO (el "gap") antes del token actual
            if start > last_index:
                descriptive_part = text[last_index:start].strip()
                if descriptive_part:
                    parts.append(descriptive_part)
            
            # 2. Añadir el fragmento CUANTITATIVO
            quantitative_part = token.strip() # Usamos el token de quant_runs
            if quantitative_part:
                parts.append(quantitative_part)
            
            last_index = end

        # 3. Añadir cualquier fragmento DESCRIPTIVO restante al final del texto
        if last_index < len(text):
            final_part = text[last_index:].strip()
            if final_part:
                parts.append(final_part)
        
        # Filtrar partes vacías que hayan podido quedar
        parts = [p for p in parts if p]

        # Si el resultado es menos de 2 fragmentos, no hacer nada.
        if len(parts) < self.min_contours_for_frag:
             logger.info(f"Fragmentación cuantitativa cancelada para '{text}'. Partes resultantes: {parts}")
             return [polygon]

        # Intentar usar blobs si coinciden en número (Lógica visual preferida)
        if blob_metrics:
            boxes = blob_metrics.get("blobs_norm_boxes")
            if boxes and len(boxes) >= len(parts):
                boxes_sorted = sorted(boxes, key=lambda b: b[0])
                selected = boxes_sorted[: len(parts)]
                logger.debug(f"Fragmentando '{text}' en {parts} usando {len(selected)} blobs visuales.")
                return self._fragment_using_boxes(polygon, parts, selected)
        
        logger.debug(f"Fragmentando '{text}' en {parts} usando geometría proporcional.")
        
        char_lengths = [len(p) for p in parts]
        total_chars = sum(char_lengths)
        if total_chars == 0:
            return [polygon] # Evitar división por cero

        xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
        width = xmax - xmin

        new_polys: List[Polygons] = []
        current_x = xmin

        for i, part in enumerate(parts):
            part_ratio = char_lengths[i] / total_chars
            part_width = part_ratio * width
            new_xmax = current_x + part_width

            # Asegurar que new_xmax no exceda el límite original
            new_xmax = min(new_xmax, xmax)

            new_bbox = np.array([current_x, ymin, new_xmax, ymax], dtype=np.float32)
            new_centroid = np.array([(current_x + new_xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)

            new_geom = dataclasses.replace(
                polygon.geometry,
                bounding_box=new_bbox,
                centroid=new_centroid,
                polygon_coords=np.array([
                    [new_bbox[0], new_bbox[1]],
                    [new_bbox[2], new_bbox[1]],
                    [new_bbox[2], new_bbox[3]],
                    [new_bbox[0], new_bbox[3]],
                ], dtype=np.float32)
            )

            logger.info(f"Fragmento cuantitativo/mixto: texto nuevo='{part}'")

            new_polys.append(dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_refined=True
            ))
            current_x = new_xmax

        return new_polys
    
    def _fragment_using_boxes(self, polygon: Polygons, parts: List[str], boxes_norm: List[List[float]]) -> List[Polygons]:
        """Crea fragmentos usando cajas normalizadas (exactas)"""
        new_polys: List[Polygons] = []
        pad_xmin, pad_ymin, _, _ = polygon.cropedd_geometry.padding_coords
        poly_width = polygon.cropedd_geometry.croppy_dims.get("poly_width", 0)
        poly_height = polygon.cropedd_geometry.croppy_dims.get("poly_height", 0)

        if poly_width < 1 or poly_height < 1:
            return [polygon]

        for i, (xn1, yn1, xn2, yn2) in enumerate(boxes_norm):
            xn1 = float(np.clip(xn1, 0, 1)); yn1 = float(np.clip(yn1, 0, 1))
            xn2 = float(np.clip(xn2, 0, 1)); yn2 = float(np.clip(yn2, 0, 1))

            xmin_abs = round(pad_xmin + xn1 * poly_width)
            xmax_abs = round(pad_xmin + xn2 * poly_width)
            ymin_abs = round(pad_ymin + yn1 * poly_height)
            ymax_abs = round(pad_ymin + yn2 * poly_height)

            new_bbox = np.array([xmin_abs, ymin_abs, xmax_abs, ymax_abs], dtype=np.float32)
            new_centroid = np.array([(xmin_abs + xmax_abs)/2, (ymin_abs + ymax_abs)/2], dtype=np.float32)

            new_geom = dataclasses.replace(
                polygon.geometry,
                bounding_box=new_bbox,
                centroid=new_centroid,
                polygon_coords=np.array([
                    [new_bbox[0], new_bbox[1]],
                    [new_bbox[2], new_bbox[1]],
                    [new_bbox[2], new_bbox[3]],
                    [new_bbox[0], new_bbox[3]],
                ], dtype=np.float32)
            )

            frag_text = parts[i] if i < len(parts) else ""
            new_polys.append(dataclasses.replace(polygon, geometry=new_geom, ocr_text=frag_text, was_refined=True))

        return new_polys

    def get_semantic_type_name(self, semantic_clasification: List[int] | int, manager: DataFormatter) -> str:
        """Convierte el tipo semántico numérico a nombre legible usando el mapeo del formatter"""
        semantic_map = manager.get_semmantic_types()
        for name, value in semantic_map.items():
            if value == semantic_clasification:
                return name
        return "descriptive"
