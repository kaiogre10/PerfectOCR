# core/workers/ocr/semantic_clasificator.py
import logging
from typing import Dict, Any, Tuple, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_encoder import encode_text, get_morphological_encode, get_char_num
from core.utils.pattern_finder import find_umd, find_quantitative, contains_quantitative
from core.utils.math_utils import vectorice_values
from core.utils.text_validator import validate_text

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("semantic_clasificator", {})
        self.semantic_range: Tuple[float, float] = self.worker_config["semantic_range"]
        self.encode_mean: Tuple[float, float] = self.worker_config["encode_mean"]
        self.morph_mean: Tuple[float, float] = self.worker_config["morph_mean"]
        self.enabled_outputs = self.config.get("image_load_outputs", {})
        self.output = self.enabled_outputs.get("semantic_field", False)
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter, final_pass: str = "") -> bool:
        """Clasifica polígonos semánticamente"""
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para procesar")
                return False
                
            polygons_to_classify: Dict[str, Polygons] = manager.workflow.polygons
            
            if not polygons_to_classify:
                logger.warning("No hay polígonos que clasificar")
                return True

            # Clasificar solo los polígonos seleccionados
            encoder: Dict[str, float] = manager.get_density_encoder()
            inv_encoder: Dict[str, float] = manager.get_inverse_frecuency_encoder()
            final_results: Dict[str, int | List[int]] = self._clasify_words(polygons_to_classify, encoder, inv_encoder)
            
            classified_count = len(final_results)
            logger.debug(f"Total clasificados: {classified_count}")

            # Actualizar semantic_type Y resetear was_refined si es modo filtrado
            manager.update_semantic_clasification(final_results)

            for poly_id, polygon in polygons_to_classify.items():
                text = polygon.ocr_text
                sc = polygon.semantic_clasification
                logger.debug(f"Clasificación {poly_id}: '{text}', '{sc}'")

            if self.output and validate_text(final_pass):
                from services.output_service import save_raw_json
                file_name: str = manager.workflow.metadata.image_name  # type: ignore
                name = "semantic_clasificator"
                worker_name = f"{name}_{final_pass}"
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

                save_raw_json( output_paths, worker_name, results, file_name)
            
            return True

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
            
    def _clasify_words(self, polygons: Dict[str, Polygons], encoder: Dict[str, float], inv_encoder: Dict[str, float]) -> Dict[str, int | List[int]]:
        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, int | list[int]] = {}

        def classify_token(tok: str) -> int:
            s = tok.strip(' ')
            chars = [ch for ch in s if not ch.isspace()]
            total = len(chars)
            char_num = get_char_num()
            pct = (sum(1 for ch in chars if ch in char_num) / total) * 100.0 if total else 0.0

            encoded_poly = encode_text(s, encoder)
            poly_mean: float = vectorice_values(encoded_poly, value="mean") # type: ignore
            inv_encoded_poly = encode_text(s, inv_encoder)
            inv_poly_mean: float = vectorice_values(inv_encoded_poly, value="mean") # type: ignore
            morph_text = get_morphological_encode(s)
            poly_morph_mean: float = vectorice_values(morph_text, value="mean") # type: ignore

            semantic_type = 0  # descriptive
            if contains_quantitative(s):
                semantic_type = 2  # quantitative
            elif find_umd(s):
                semantic_type = -2  # umd
            elif self.morph_mean[1] < poly_morph_mean and poly_mean < self.encode_mean[0] and self.encode_mean[1] < inv_poly_mean and self.semantic_range[1] < pct :
                has_quantitative = find_quantitative(s)
                if has_quantitative:
                    semantic_type = 2
                else:
                    semantic_type = 1  # numeric
                    
            elif self.semantic_range[0] < pct < self.semantic_range[1] and self.morph_mean[0] < poly_morph_mean < self.morph_mean[1]:
                semantic_type = -1  # code
            else:
                semantic_type = 0  # descriptive
            return int(semantic_type)

        for pid, word in texts.items():
            s = word.strip(' ')
            tokens = [t for t in s.split(' ') if t != '']

            if len(tokens) <= 1:
                sc_val = classify_token(s)
                final_results[pid] = sc_val
            else:
                sc_list = [classify_token(t) for t in tokens]
                final_results[pid] = sc_list
                
        return final_results
