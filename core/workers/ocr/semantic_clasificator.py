# core/workers/ocr/semantic_clasificator.py
import logging
import numpy as np
from typing import Dict, Any, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_encoder import encode_text, get_morphological_map
from core.utils.pattern_finder import find_umd, find_quantitative, contains_quantitative

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("semantic_clasificator", {})
        self.char_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"]
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("semantic_field", False)
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter, filter_modified: bool = False) -> bool:
        """Clasifica polígonos semánticamente"""
        logger.debug(f"Clasificador iniciado (filter_modified={filter_modified})")
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para procesar")
                return False
                
            all_polygons: Dict[str, Polygons] = manager.workflow.polygons
            
            # FILTRO SELECTIVO: Solo clasificar polígonos modificados si filter_modified=True
            if filter_modified:
                polygons_to_classify = {
                    pid: p for pid, p in all_polygons.items() 
                    if p.was_refined
                }
                logger.debug(f"Modo selectivo: {len(polygons_to_classify)}/{len(all_polygons)} polígonos modificados para reclasificar")
            else:
                polygons_to_classify = all_polygons
                logger.info(f"Modo completo: clasificando todos los {len(all_polygons)} polígonos")
            
            if not polygons_to_classify:
                logger.debug("No hay polígonos que clasificar")
                return True

            # Clasificar solo los polígonos seleccionados
            encoder: Dict[str, float] = manager.get_density_encoder()
            inv_encoder: Dict[str, float] = manager.get_inverse_frecuency_encoder()
            final_results: Dict[str, int] = self._clasify_words(polygons_to_classify, encoder, inv_encoder)
            
            classified_count = len(final_results)
            logger.debug(f"Total clasificados: {classified_count}")

            # Actualizar semantic_type Y resetear was_refined si es modo filtrado
            manager.update_semantic_clasification(final_results, reset_refined=filter_modified)

            if self.output:
                from services.output_service import save_debug_ocr
                file_name: str = manager.workflow.metadata.image_name  # type: ignore
                worker_name = "semantic_clasificator"
                output_paths = context["output_paths"]
                polygons = manager.workflow.polygons if manager.workflow else {}
                results: Dict[str, Any] = {}
                for poly_id, polygon in polygons.items():
                    text = getattr(polygon, "ocr_text", None)
                    sc = getattr(polygon, "semantic_clasification", None)
                    results[poly_id] = {
                        "text": text,
                        "semantic_clasification": sc
                    }

                save_debug_ocr( output_paths, worker_name, results, file_name)
            
            return True

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
            
    # Cambiar el método _clasify_words para devolver Dict[str, int] en lugar de Dict[str, SemanticClassification]
    def _clasify_words(self, polygons: Dict[str, Polygons], encoder: Dict[str, float], inv_encoder: Dict[str, float]) -> Dict[str, int]:
        semantic_range: Tuple[float, float] = self.worker_config.get("semantic_range", [])
        encode_mean: Tuple[float, float] = self.worker_config.get("encode_mean", [])
        morph_mean: Tuple[float, float] = self.worker_config.get("morph_mean", [])

        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, int] = {}

        for pid, s in texts.items():
            chars = [ch for ch in s if not ch.isspace()]
            total = len(chars)
            pct = (sum(1 for ch in chars if ch in self.char_num) / total) * 100.0 if total else 0.0

            encoded_poly = encode_text(s, encoder)
            poly_mean = np.mean(encoded_poly)

            inv_encoded_poly = encode_text(s, inv_encoder)
            inv_poly_mean = np.mean(inv_encoded_poly)

            morph_text = get_morphological_map(s)
            poly_morph_mean = np.mean(morph_text) if morph_text else - 1.0

            # Lógica de clasificación simplificada a enteros
            semantic_type = 0  # descriptive por defecto
            
            if contains_quantitative(s):
                semantic_type = 2  # quantitative
            elif find_umd(s):
                semantic_type = -2  # umd
            elif  morph_mean[1] < poly_morph_mean and poly_mean < encode_mean[0] and encode_mean[1] < inv_poly_mean and semantic_range[1] < pct :
                has_quantitative = find_quantitative(s)
                if has_quantitative:
                    semantic_type = 2  # quantitative
                else:
                    semantic_type = 1  # numeric
            elif semantic_range[0] < pct < semantic_range[1] and morph_mean[0] < poly_morph_mean < morph_mean[1]:
                semantic_type = -1  # code
            else:
                # pct < semantic_range[0] and poly_morph_mean < morph_mean[0]
                semantic_type = 0  # Descriptive

            logger.debug(f"{pid}: '{s}'| mean: {poly_mean:.4f}, inv_mean: {inv_poly_mean}, morph: {poly_morph_mean}, {pct}% | sc: {semantic_type}")

            final_results[pid] = semantic_type

        return final_results
