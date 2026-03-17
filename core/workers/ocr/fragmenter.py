# PerfectOCr/core/workers/ocr/fragmenter.py
import dataclasses
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, Geometry
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import is_acronym
from core.utils.math_utils import fragment_geometry_horizontal

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
                
                sc: List[int]= polygon.semantic_clasification
                ocr_text: str = polygon.ocr_text or ""
                
                if not ocr_text:
                    logger.debug(f"Polygono sin texto: {poly_id}")
                    continue

                # Si el texto corresponde a una sigla (p.e. 'P.U.C.D', 'I.V.A.') se conserva intacto
                if is_acronym(ocr_text):
                    # logger.info(f"{poly_id} no fragmentando sigla detectada: '{ocr_text}'")
                    final_polygons.append(polygon)
                    continue
                
                semantic_frag = len(sc) > 1 and any(c != 0 for c in sc)
                                    
                if semantic_frag:
                    # logger.info(f"Poligono {poly_id}: '{ocr_text} se fragmentará")
                    fragments = self.fragment_by_semantic_classification(polygon)
                else:
                    fragments = [polygon]
                
                # Checar si verdaderamente hubo fragmentación para el conteo
                if len(fragments) > 1:
                    fragmented_count += 1

                # Extender la lista final una sola vez con los fragmentos (o el polígono original si era 1 solo)
                final_polygons.extend(fragments)

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

    def fragment_by_semantic_classification(self, polygon: Polygons) -> List[Polygons]:
        """
        Fragmenta un polígono según su clasificación semántica.
        Regla: solo se permite más de una clasificación por polígono si todas son 0.
        Cualquier valor distinto de 0 (1, 2, -1, -2) debe estar solo en su polígono.
        """
        text: str = (polygon.ocr_text or "").strip()
        if not text:
            return [polygon]

        sc: List[int] = polygon.semantic_clasification
        
        # Usar split() sin argumentos ayuda a lidiar con cualquier formato de espacios en blanco
        parts = [p for p in text.split(' ') if p]
        
        # Verificar alineación
        if len(parts) != len(sc):
            logger.warning(f"Desalineación en {polygon.polygon_id}: {len(parts)} tokens vs {len(sc)} clasificaciones. Texto: {parts}, Clas: {sc}")
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
        # Longitud en caracteres de cada fragmento (para proporción)
        frag_char_lengths = [sum(len(t) for t in frag_tokens) for frag_tokens, _ in fragments]
        total_chars = sum(frag_char_lengths)
        
        if total_chars == 0:
            return [polygon]
        
        new_polys: List[Polygons] = []
        proportions = [float(frag_len) / float(total_chars) for frag_len in frag_char_lengths]
        geom_parts = fragment_geometry_horizontal(polygon.geometry, num_fragments=len(fragments), proportions=proportions)
        if not geom_parts:
            return [polygon]
            
        for (frag_tokens, _), geom_part in zip(fragments, geom_parts):
            new_geom = Geometry(
                polygon_coords=geom_part["polygon_coords"],
                bounding_box=geom_part["bounding_box"],
                centroid=geom_part["centroid"],
            )
            frag_text = ' '.join(frag_tokens)
            new_poly = dataclasses.replace(
                polygon,
                geometry=new_geom,
                ocr_text=frag_text,
            )
            new_polys.append(new_poly)
        
        # logger.info(f"Fragmentación semántica de {polygon.polygon_id}: Original: '{text}' -> '{fragments}'")
        return new_polys