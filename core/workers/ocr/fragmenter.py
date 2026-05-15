# PerfectOCr/core/workers/ocr/fragmenter.py
import dataclasses
import logging
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
        self.output = config.get("fragmented_polys", False)

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool: 
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Fragmentador no tiene polígonos para procesar")
                return False
            
            polygons_in: Dict[str, Polygons] = manager.workflow.polygons            
            
            fragmented_count = 0
            final_polygons: List[Polygons] = []
            for poly_id, polygon in polygons_in.items():
                kf = polygon.key_field or None
                sc: List[int]= polygon.semantic_clasification
                if kf is not None or 0 in sc:
                    # logger.info(f"'{poly_id}' con KEYFIELD '{kf}' no se fragmenta: '{polygon.ocr_text}'")
                    final_polygons.append(polygon)
                    continue
                
                ocr_text: str = polygon.ocr_text or ""
                ocr_text = ocr_text.strip()
                
                if not ocr_text:
                    logger.debug(f"Polygono sin texto: {poly_id}")
                    continue

                # Si el texto corresponde a una sigla (p.e. 'P.U.C.D', 'I.V.A.') se conserva intacto
                if is_acronym(ocr_text):
                    logger.debug(f"{poly_id} no fragmentando sigla detectada: '{ocr_text}'")
                    final_polygons.append(polygon)
                    continue
                
                semantic_frag = len(sc) > 1 and any(c > 0 for c in sc)
                                    
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
            
            poly_debug: Dict[str, Polygons] = manager.workflow.polygons
            for pid, pd in poly_debug.items():
                text = pd.ocr_text or ""
                sc = pd.semantic_clasification
                kf = pd.key_field or None
                congruency = len(text.split(" ")) == len(sc)
                if not congruency:
                    if 0 in sc and kf is not None:
                        continue
                    logger.warning(f"{pid} INCONGRUENTE CON SC: TEXTO: '{text}' -> SC {sc}")
                    
            return True
        except Exception as e:
            logger.warning(f"Error fragmentando: {e}", exc_info=True)
        return False

    def fragment_by_semantic_classification(self, polygon: Polygons) -> List[Polygons]:
        """
        Fragmenta un polígono según su clasificación semántica.
        Regla: agrupa tokens consecutivos del mismo tipo de clasificación.
        - Tokens con cls=1 se agrupan entre sí
        - Tokens con cls=2 se agrupan entre sí
        - Otros valores (3, 4, 5) van cada uno en su propio fragmento
        """
        text: str = polygon.ocr_text or ""
        if not text:
            return [polygon]

        sc: List[int] = polygon.semantic_clasification
        
        # Usar split() sin argumentos ayuda a lidiar con cualquier formato de espacios en blanco
        parts = [p for p in text.split(' ') if p]
        # logger.debug(f"TEXTO: '{text}' | PARTS: '{parts}'")
        
        # Verificar alineación
        total_tokens = len(parts)
        total_sc = len(sc)
        if total_tokens != total_sc:
            logger.warning(f"Desalineación en '{text}': {total_tokens} tokens vs {total_sc} clasificaciones. Texto: {parts}, Clas: {sc}")
            return [polygon]
        
        fragments: List[Tuple[List[str], List[int]]] = []
        current_tokens: List[str] = []
        current_scs: List[int] = []
        current_cls: int | None = None
        for _, (token, cls) in enumerate(zip(parts, sc)):
            if cls in (1, 2):
                # Si la clase cambia o es la primera, cerrar fragmento anterior
                if current_cls is not None and current_cls != cls:
                    if current_tokens:
                        fragments.append((current_tokens, current_scs))
                    current_tokens = []
                    current_scs = []
                current_cls = cls
                current_tokens.append(token)
                current_scs.append(cls)
            else:
                # Cerrar fragmento anterior si existe
                if current_tokens:
                    fragments.append((current_tokens, current_scs))
                    current_tokens = []
                    current_scs = []
                    current_cls = None
                # Cada otro valor va en su propio fragmento
                fragments.append(([token], [cls]))
        
        # Cerrar último fragmento si quedó pendiente
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
            
        for (frag_tokens, frag_scs), geom_part in zip(fragments, geom_parts):
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
                semantic_clasification=frag_scs
            )
            new_polys.append(new_poly)
        
        # logger.info(f"Fragmentación semántica de '{polygon.polygon_id}' = Original: '{text}' -> '{fragments}'")
        return new_polys