# core/workers/ocr/semantic_clasificator.py
import logging
import dataclasses
import numpy as np
from typing import Dict, Any, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, SemanticClassification
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_encoder import encode_text
from core.utils.pattern_finder import find_umd, find_quantitative

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("semantic_clasificator", {})
        self.char_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"]
            
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
                logger.debug(f"Modo completo: clasificando todos los {len(all_polygons)} polígonos")
            
            if not polygons_to_classify:
                logger.debug("No hay polígonos que clasificar")
                return True

            # Clasificar solo los polígonos seleccionados
            encoder: Dict[str, float] = manager.get_density_encoder()
            final_results: Dict[str, SemanticClassification] = self._clasify_words(polygons_to_classify, encoder)
            
            classified_count = len(final_results)
            logger.debug(f"Total clasificados: {classified_count}")
            
            # for poly_id, semantic_obj in final_results.items():
            #     if poly_id in all_polygons:
            #         polygon = all_polygons[poly_id]
            #         active_fields = [field for field in ['quantitative', 'umd', 'rfc', 'numeric', 'descriptive', 'code'] if getattr(semantic_obj, field)]
            #         logger.debug(f"{poly_id}: {active_fields} | texto: '{polygon.ocr_text or ''}'")
            
            # Actualizar semantic_type Y resetear was_refined si es modo filtrado
            manager.update_semantic_clasification(final_results, reset_refined=filter_modified)
            
            return True

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
            
    def _clasify_words(self, polygons: Dict[str, Polygons], encoder: Dict[str, float]) -> Dict[str, SemanticClassification]:
        semantic_range: Tuple[float, float] = self.worker_config.get("semantic_range", [])
        encode_mean: Tuple[float, float] = self.worker_config.get("encode_mean", [])

        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, SemanticClassification] = {}

        for pid, s in texts.items():
            chars = [ch for ch in s if not ch.isspace()]
            total = len(chars)
            pct = (sum(1 for ch in chars if ch in self.char_num) / total) * 100.0 if total else 0.0

            encoded_poly = encode_text(s, encoder)
            poly_mean = np.mean(encoded_poly)
            poly_std = np.std(encoded_poly)
            poly_var = np.var(encoded_poly)
            poly_median = np.median(encoded_poly)

            # Verificar si existe patrón cuantitativo PRIMERO
            tokens = [t for t in (s or "").split() if t]
            if not tokens:
                tokens = [s]

            # Crear objeto SemanticClassification con todos los campos en False
            semantic_clasification = SemanticClassification(
                quantitative=False,
                umd=False,
                numeric=False,
                descriptive=False,
                code=False
            )

            # RFC y UMD antes (patrones fuertes/ortogonales)
            if find_umd(s):
                logger.debug(
                    f"{pid}: '{s}'| mean: {poly_mean:.4f}, median: {poly_median:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, {pct}% | UMD")
                semantic_clasification = dataclasses.replace(semantic_clasification, umd=True)
            # Numérico primero; cuantitativo solo si es numérico
            elif semantic_range[1] < pct and poly_mean < encode_mean[0]:
                # Verificación cuantitativa SOLO dentro de los numéricos
                has_quantitative = find_quantitative(s)
                if has_quantitative:
                    logger.debug(
                        f"{pid}: '{s}'| mean: {poly_mean:.4f}, median: {poly_median:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, {pct}% | QUANTITATIVE")
                    semantic_clasification = dataclasses.replace(semantic_clasification, quantitative=True)
                else:
                    logger.debug(
                        f"{pid}: '{s}'| mean: {poly_mean:.4f}, median: {poly_median:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, {pct}% | NUMERIC")
                    semantic_clasification = dataclasses.replace(semantic_clasification, numeric=True)
            elif pct < semantic_range[0]:
                logger.debug(
                    f"{pid}: '{s}'| mean: {poly_mean:.4f}, median: {poly_median:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, {pct}% | DESCRIPTIVE")
                semantic_clasification = dataclasses.replace(semantic_clasification, descriptive=True)
            else:
                logger.debug(
                    f"{pid}: '{s}' | mean: {poly_mean:.4f}, median: {poly_median:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, {pct}% | CODE")
                semantic_clasification = dataclasses.replace(semantic_clasification, code=True)

            final_results[pid] = semantic_clasification

        return final_results
            
    