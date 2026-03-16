# PerfectOCr/core/workers/ocr/fragmenter.py
import dataclasses
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import validate_text, is_acronym, separate_punt
from core.utils.data_utils import PUNC_CHARS

logger = logging.getLogger(__name__)

class Fragmenter(OCRAbstractWorker):
    """Fragmenta polígonos basados en señales textuales (espacios) y visuales (blobs)"""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get("text_refiner", {})
        self.min_contours_for_frag = worker_config.get("min_cc_for_frag")
        self.output = config.get("fragmented_polys", False)

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool: 
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Fragmentador no tiene polígonos para procesar")
                return False
            
            polygons_in: Dict[str, Polygons] = manager.workflow.polygons
            sorted_poly_ids = sorted(polygons_in.keys())
            logger.debug(f"Cantidad de polígonos recibidos:{len(sorted_poly_ids)}")
            
            fragmented_count = 0
            final_polygons: List[Polygons] = []
            
            for poly_id in sorted_poly_ids:
                polygon = polygons_in[poly_id]
                
                sc: List[int] | int = polygon.semantic_clasification or 0
                ocr_text: str = polygon.ocr_text or ""
                
                if not validate_text(ocr_text):
                    logger.debug(f"Polygono sin texto: {poly_id}")
                    continue

                # Si el texto corresponde a una sigla (p.e. 'P.U.C.D', 'I.V.A.') se conserva intacto
                if is_acronym(ocr_text):
                    logger.debug(f"{poly_id} no fragmentando sigla detectada: '{ocr_text}'")
                    final_polygons.append(polygon)
                    continue

                punctuation_needs_frag = (
                    not (all(cls in (1, 2, -2) for cls in sc) if isinstance(sc, list) else sc in (1, 2, -2)) and
                    (any(punct in (ocr_text or "") for punct in [";", ":", "!", "?"]) or
                    (ocr_text or "").count('.') == 1)
                )
                    
                # logger.info(f"{poly_id}: {visual_needs_frag}")

                semantic_frag = isinstance(sc, list) and any(c != 0 for c in sc)

                if semantic_frag or punctuation_needs_frag:
                    if semantic_frag:
                        reason = "semantic"
                    else:
                        reason = "puntuación"

                    logger.debug(f"{poly_id}: FRAG por {reason}: '{ocr_text}' | SC: {sc}")
                    
                    if semantic_frag:
                        fragments = self.fragment_by_semantic_classification(polygon)

                    elif punctuation_needs_frag:
                        fragments = self.fragment_by_punctuation(polygon)
                    else:
                        fragments = [polygon]
                     
                    final_polygons.extend(fragments)
                    if len(fragments) > 1:
                        fragmented_count += 1

                else:
                    final_polygons.append(polygon)

            final_polygons_dict: Dict[str, Polygons] = {}
            for idx, poly_obj in enumerate(final_polygons):
                new_id = f"poly_{idx:04d}"
                new_index = idx
                final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, poly_index=new_index)
                final_polygons_dict[new_id] = final_poly_obj
                manager.workflow.polygons = final_polygons_dict

            # if self.output and blob_metrics:
            #     logger.info(f"Enaled output activdo")
            #     from services.output_service import save_croped_image
            #     output_paths = context["output_paths"]
            #     polygons = manager.workflow.polygons if manager.workflow else {}
            #     image_name = manager.workflow.metadata.image_name if manager.workflow else ""

            #     for poly_id, polygon in polygons.items():
            #         cropped_img = polygon.cropped_img.cropped_img if polygon.was_fragmented else None
            #         # if polygon.was_fragmented:
            #         logger.debug(f"{poly_id}: {polygon.was_fragmented}")
            #         save_croped_image(image_name, poly_id, cropped_img, output_paths, "fragmenter")
            if fragmented_count > 0:
                logger.debug(f"Fragmenter: Se fragmentaron {fragmented_count} resultando en {len(final_polygons_dict)} polígonos totales.")
                return True
                
        except Exception as e:
            logger.warning(f"Error fragmentando: {e}", exc_info=True)
        return False

    # def fragment_by_blobs(self, polygon: Polygons, blob_metrics: Dict[str, Any]) -> List[Polygons]:
    #     """
    #     Fragmenta un polígono 1-a-1 con cada blob detectado,
    #     garantizando geometría precisa (sin solapes ni desfases).
    #     """
    #     new_polys: List[Polygons] = []
    #     blobs_norm_boxes = blob_metrics["blobs_norm_boxes"]
    #     num_blobs = blob_metrics.get("num_blobs", 0)
    #     text = polygon.ocr_text or ""
    #     text_parts = (text or "").strip().split()
        
    #     if not validate_text(text):
    #         return [polygon]

    #     # Debe existir correspondencia exacta blobs ↔ palabras
    #     if not blobs_norm_boxes or num_blobs < self.min_contours_for_frag or len(text_parts) != num_blobs:
    #         # logger.info(f"No se fragmenta {polygon.polygon_id}: blobs/palabras {num_blobs}/{len(text_parts)}")
    #         return [polygon]

    #     # Padding coords = referencia absoluta del recorte
    #     pad_xmin, pad_ymin, _, _ = polygon.cropedd_geometry.padding_coords
    #     poly_width  = polygon.cropedd_geometry.croppy_dims.get("poly_width", 0)
    #     poly_height = polygon.cropedd_geometry.croppy_dims.get("poly_height", 0)

    #     if poly_width < 1 or poly_height < 1:
    #         logger.warning("Fragmenter: Dimensiones de recorte inválidas.")
    #         return [polygon]

    #     for i, (xn1, yn1, xn2, yn2) in enumerate(blobs_norm_boxes):
    #         # Clamp y conversión segura
    #         xn1 = float(np.clip(xn1, 0.0, 1.0))
    #         yn1 = float(np.clip(yn1, 0.0, 1.0))
    #         xn2 = float(np.clip(xn2, 0.0, 1.0))
    #         yn2 = float(np.clip(yn2, 0.0, 1.0))

    #         xmin_abs = round(pad_xmin + xn1 * poly_width)
    #         xmax_abs = round(pad_xmin + xn2 * poly_width)
    #         ymin_abs = round(pad_ymin + yn1 * poly_height)
    #         ymax_abs = round(pad_ymin + yn2 * poly_height)

    #         new_bbox = np.array([xmin_abs, ymin_abs, xmax_abs, ymax_abs], dtype=np.float32)
    #         new_centroid = np.array([(xmin_abs + xmax_abs) / 2, (ymin_abs + ymax_abs) / 2], dtype=np.float32)

    #         new_geom = dataclasses.replace(
    #             polygon.geometry,
    #             bounding_box=new_bbox,
    #             centroid=new_centroid,
    #             polygon_coords=np.array([
    #                 [new_bbox[0], new_bbox[1]],
    #                 [new_bbox[2], new_bbox[1]],
    #                 [new_bbox[2], new_bbox[3]],
    #                 [new_bbox[0], new_bbox[3]],
    #             ], dtype=np.float32)
    #         )

    #         frag_text = text_parts[i]  # 1-a-1 con blobs
            
    #         logger.info(f"Fragmento visual: texto='{frag_text}'") #, bbox={new_bbox.tolist()}")
    #         new_poly = dataclasses.replace(
    #             polygon,
    #             geometry=new_geom,
    #             ocr_text=frag_text,
    #             was_fragmented=True
    #         )
    #         new_polys.append(new_poly)

    #     return new_polys

    def fragment_by_punctuation(self, polygon: Polygons) -> List[Polygons]:
        """
        Fragmenta un polígono dividiendo por puntuación (.,;:!?) y creando polígonos separados.
        """
        text = (polygon.ocr_text or "").strip()

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
                    )
                    new_polys.append(new_poly)
                    current_x = new_xmax - 1

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
            elif part not in PUNC_CHARS:
                filtered_parts.append(part)
        
        if len(filtered_parts) < self.min_contours_for_frag:
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
            if part_text in PUNC_CHARS:
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
            
            logger.debug(f"Fragmento por puntuación: texto='{part}'")#, bbox={new_bbox.tolist()}")

            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=part,
                was_fragmented=True
            )
            new_polys.append(new_poly)
            current_x = new_xmax - 1

        return new_polys

    def fragment_by_semantic_classification(self, polygon: Polygons) -> List[Polygons]:
        """
        Fragmenta un polígono según su clasificación semántica.
        Regla: solo se permite más de una clasificación por polígono si todas son 0.
        Cualquier valor distinto de 0 (1, 2, -1, -2) debe estar solo en su polígono.
        """
        text: str = (polygon.ocr_text or "").strip()
        if not text:
            return [polygon]

        sc: List[int] = polygon.semantic_clasification if isinstance(polygon.semantic_clasification, list) else [polygon.semantic_clasification]
        
        parts = [p for p in text.split(' ') if p]
        
        # Verificar alineación
        if len(parts) != len(sc):
            logger.warning(f"Desalineación en {polygon.polygon_id}: {len(parts)} tokens vs {len(sc)} clasificaciones")
            return [polygon]
        
        # Construir fragmentos según la regla:
        # - 0s consecutivos → un solo fragmento
        # - Cada valor no-0 → un fragmento individual
        fragments: List[Tuple[List[str], List[int]]] = []
        current_tokens: List[str] = []
        current_scs: List[int] = []
        
        for _, (token, cls) in enumerate(zip(parts, sc)):
            if cls == 0:
                # Acumular 0s consecutivos
                current_tokens.append(token)
                current_scs.append(cls)
            else:
                # Primero, cerrar cualquier fragmento de 0s pendiente
                if current_tokens:
                    fragments.append((current_tokens, current_scs))
                    current_tokens = []
                    current_scs = []
                # Cada no-0 va en su propio fragmento
                fragments.append(([token], [cls]))
        
        # Cerrar último fragmento de 0s si quedó pendiente
        if current_tokens:
            fragments.append((current_tokens, current_scs))
        
        # Si solo hay un fragmento, no hace falta dividir
        if len(fragments) <= 1:
            return [polygon]
        
        # Calcular geometría proporcional
        xmin, ymin, xmax, ymax = polygon.geometry.bounding_box
        width = xmax - xmin
        
        # Longitud en caracteres de cada fragmento (para proporción)
        frag_char_lengths = [sum(len(t) for t in frag_tokens) for frag_tokens, _ in fragments]
        total_chars = sum(frag_char_lengths)
        
        if total_chars == 0:
            return [polygon]
        
        new_polys: List[Polygons] = []
        current_x = xmin
        
        for (frag_tokens, _), frag_len in zip(fragments, frag_char_lengths):
            frag_ratio = frag_len / total_chars
            frag_width = frag_ratio * width
            new_xmax = current_x + frag_width
            
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
            
            # Texto: unir tokens con espacio
            frag_text = ' '.join(frag_tokens)
            
            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=frag_text,
                was_fragmented=True
            )
            new_polys.append(new_poly)
            current_x = new_xmax - 1
        
        # logger.info(f"Fragmentación semántica de {polygon.polygon_id}: Original: '{text}' -> '{fragments}'")
        return new_polys