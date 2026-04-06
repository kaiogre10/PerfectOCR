# PerfectOCr/core/workers/ocr/fragmenter.py
import dataclasses
import logging
from typing import Dict, Any, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, Geometry
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import is_acronym
from core.utils.math_utils import fragment_geometry_horizontal
# from services.output_service import save_croped_image

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
                    logger.debug(f"Poligono {poly_id}: '{ocr_text} se fragmentará")
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
            if fragmented_count > 0:
                logger.debug(f"Fragmenter: Se fragmentaron {fragmented_count} resultando en {len(final_polygons_dict)} polígonos totales.")
                return True
                
        except Exception as e:
            logger.warning(f"Error fragmentando: {e}", exc_info=True)
        return False

    def fragment_by_semantic_classification(self, polygon: Polygons) -> List[Polygons]:
        """
        Fragmenta un polígono según su clasificación semántica.
        Regla: solo se permite más de una clasificación por polígono si todas son 0.
        Cualquier valor distinto de 0 (1, 2, -1, -2) debe estar solo en su polígono.
        """
        text: str = polygon.ocr_text or ""
        if not text:
            return [polygon]

        sc: List[int] = polygon.semantic_clasification
        
        # Usar split() sin argumentos ayuda a lidiar con cualquier formato de espacios en blanco
        parts = [p for p in text.split(' ') if p]
        logger.info(f"TEXTO: '{text}' | PARTS: '{parts}'")
        
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