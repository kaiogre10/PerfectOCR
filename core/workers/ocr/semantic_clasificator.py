# core/workers/ocr/semantic_clasificator.py
import logging
import time
from typing import Dict, Any, List, Tuple, Set
import numpy as np
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_encoder import text_encode
from core.utils.pattern_finder import find_umd, find_quantitative, contains_quantitative
from core.utils.text_validator import validate_text, get_char_num

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get("semantic_clasificator", {})
        self.semantic_range: Tuple[float, float] = worker_config["semantic_range"]
        self.encode_mean: Tuple[float, float] = worker_config["encode_mean"]
        self.morph_mean: Tuple[float, float] = worker_config["morph_mean"]
        self.char_num: Set[str] = get_char_num()
        self.output = config.get("semantic_field", False)
            
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
            final_results: Dict[str, int | List[int]] = self._clasify_words(polygons_to_classify)
            
            classified_count = len(final_results)
            logger.debug(f"Total clasificados: {classified_count}")

            # Actualizar semantic_type Y resetear was_refined si es modo filtrado
            manager.update_semantic_clasification(final_results)

            for poly_id, polygon in polygons_to_classify.items():
                text = polygon.ocr_text
                sc = polygon.semantic_clasification
                # logger.info(f"Clasificación {poly_id}: | '{text}', | '{sc}' |")

            if self.output and validate_text(final_pass):
                from services.output_service import save_raw_json
                file_name: str = manager.workflow.metadata.image_name  # type: ignore
                name = "semantic_clasification"
                worker_name = f"{name}"
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
            
    def _clasify_words(self, polygons: Dict[str, Polygons]) -> Dict[str, int | List[int]]:
        t0 = time.perf_counter()
        morph_mean_high = self.morph_mean[1]
        morph_mean_low = self.morph_mean[0]
        encode_mean_low = self.encode_mean[0]
        encode_mean_high = self.encode_mean[1]
        semantic_range_high = self.semantic_range[1]
        semantic_range_low = self.semantic_range[0]
        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, int | list[int]] = {}

        def classify_token(tok: str) -> int:
            s = tok.strip(' ')
            
            total = len(s)
            pct = (sum(1 for ch in s if ch in self.char_num) / total) * 100.0 if total else 0.0
            encoded_text = text_encode(s, ["all"])
            means = np.mean(encoded_text, axis=1).astype(np.float32)
            # logger.info(f"Means de ´{s}'" 
            #             "\n"f"{means}")
            poly_mean = means[0]
            inv_poly_mean = means[1]
            poly_morph_mean = means[2]

            semantic_type = 0  # descriptive
            if contains_quantitative(s):
                semantic_type = 2  # quantitative
            elif find_umd(s):
                semantic_type = -2  # umd
            elif morph_mean_high < poly_morph_mean and poly_mean < encode_mean_low and encode_mean_high < inv_poly_mean and self.semantic_range[1] < pct :
                has_quantitative = find_quantitative(s)
                if has_quantitative:
                    semantic_type = 2
                else:
                    semantic_type = 1  # numeric
                    
            elif semantic_range_low < pct < semantic_range_high and morph_mean_low < poly_morph_mean < morph_mean_high:
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
        logger.info(f"Clasificación semantica completa en: {time.perf_counter() - t0:.6f}'s")
        return final_results
