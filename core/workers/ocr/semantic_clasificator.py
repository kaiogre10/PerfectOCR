# core/workers/ocr/semantic_clasificator.py
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.math_utils import text_encode
from core.utils.text_utils import find_umd, contains_quantitative
from core.utils.data_utils import CHAR_NUM

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get("semantic_clasificator", {})
        self.semantic_range: Tuple[float, float] = worker_config["semantic_range"]
        self.encode_mean: Tuple[float, float] = worker_config["encode_mean"]
        self.morph_mean: Tuple[float, float] = worker_config["morph_mean"]
        self.output = config.get("semantic_field", False)
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter, final_pass: bool = False) -> bool:
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

            if self.output and final_pass:
                from services.output_service import save_raw_json
                file_name: str = manager.workflow.metadata.image_name
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
        # t0 = time.perf_counter()
        final_results: Dict[str, int | list[int]] = {}

        def classify_token(s: str) -> int:
            total = len(s)
            pct = (sum(1 for ch in s if ch in CHAR_NUM) / total) * 100.0 if total else 0.0
            
            encoded_text = text_encode(s, ["all"])
            means = np.mean(encoded_text, axis=1).astype(np.float32)
            
            poly_mean = means[0]
            inv_poly_mean = means[1]
            poly_morph_mean = means[2]

            if contains_quantitative(s):
                return 2

            elif find_umd(s):
                return -2
            
            elif self.morph_mean[1] < poly_morph_mean and poly_mean < self.encode_mean[0] and self.encode_mean[1] < inv_poly_mean and self.semantic_range[1] < pct:
                return 1  # numeric
            elif self.semantic_range[0] < pct < self.semantic_range[1] and self.morph_mean[0] < poly_morph_mean < self.morph_mean[1]:
                return -1  # code
            
            return 0  # descriptive

        for pid, polygon in polygons.items():
            s = polygon.ocr_text or ""
            if not s:
                continue

            # Fast Path 1: Alfabético puro (mismo resultado para todos los tokens)
            if s.replace(' ', '').isalpha():
                tokens = s.split(' ')
                final_results[pid] = [0] * len(tokens) if len(tokens) > 1 else 0
                continue

            # Fast Path 2: Numérico puro (mismo resultado para todos los tokens)
            elif s.replace(' ', '').isdecimal():
                tokens = s.split(' ')
                final_results[pid] = [1] * len(tokens) if len(tokens) > 1 else 1
                continue

            # Fast Path 3: UMD (Solo si es palabra única)
            elif ' ' not in s and find_umd(s):
                final_results[pid] = -2
                continue

            elif ' ' not in s and contains_quantitative(s):
                final_results[pid] = 2
                continue

            # Procesamiento normal si no entró en los Fast Paths
            elif ' ' in s:
                tokens = s.split(' ')
                final_results[pid] = [classify_token(t) for t in tokens]
            else:
                final_results[pid] = classify_token(s)
                
        # logger.info(f"Clasificación semantica completa en: {time.perf_counter() - t0:.6f}'s")
        return final_results
