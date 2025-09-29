# semantic_clasificator
import logging
import re
from typing import Dict, Any
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("semantic_clasificator", {})
        self.enabled_outputs = config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("semantic_words", False)  
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        
        logger.debug("Clasificador inciado")
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para preocesar")
                
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            
            final_results: Dict[str, str] = self._clasify_words(polygons)
            
            classified_count = 0

            for poly_id, semantic_type in final_results.items():
                if poly_id in polygons:
                    polygon = polygons[poly_id]
                    classified_count += 1

            logger.info(f"Total clasificados: {classified_count}")
            for poly_id, semantic_type in final_results.items():
                if poly_id in polygons:
                    polygon = polygons[poly_id]
                    logger.info(f"{poly_id}, semantic={semantic_type}, text='{polygon.ocr_text or ''}'")
                    
            manager.update_semantic_type(final_results)

            file_name: str = manager.workflow.metadata.image_name
            
            if self.output:
                self._save_ocr_raw(context, final_results, file_name)
            return True
            
        except Exception as e:
            logger.debug(f"Error en el clasiicador{e}", exc_info=True)
            
    def _clasify_words(self, polygons: Dict[str, Polygons]) -> Dict[str, str]:
        numeric_range = self.worker_config.get("numeric", [70.0, 100.0])
        code_range = self.worker_config.get("code", [31.0, 69.9])

        def norm(r): return (min(r[0], r[1]), max(r[0], r[1]))
        n_min, n_max = norm(numeric_range)
        c_min, c_max = norm(code_range)

        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, str] = {}

        for pid, s in texts.items():
            chars = [ch for ch in s if not ch.isspace()]
            total = len(chars)
            pct = (sum(1 for ch in chars if ch.isdigit()) / total) * 100.0 if total else 0.0

            if n_min <= pct <= n_max:
                semantic = "numeric"
                tokens = [t for t in (s or "").split() if t]
                if any(self._is_quantitative(t) for t in tokens):
                    semantic = "quantitative"
            elif c_min <= pct <= c_max:
                semantic = "code"
            else:
                semantic = "descriptive"

            final_results[pid] = semantic

        return final_results
            
    def _is_decimal_number(self, s: str) -> bool:
        s = (s or "").strip()
        if not s or "%" in s:
            return False
        # casos: 2.98, 39,90, 1,000.00, 1.000,00
        patterns = [
            r"^\d+[.,]\d+$",                                # 2.98 / 39,90
            r"^\d{1,3}(?:[.,]\d{3})+[.,]\d+$",             # 1,000.00 / 1.000,00
        ]
        return any(re.match(p, s) for p in patterns)

    def _is_currency_amount(self, s: str) -> bool:
        s = (s or "").strip()
        if not s or "%" in s:
            return False
        currency = r"[$€£¥¢]"
        # El símbolo puede estar al inicio o en medio, pero NO al final
        # Debe estar rodeado enteramente de números (no letras)
        # No se aceptan cantidades "00" (ni $00 ni 00$ ni 00$00)
        # Ejemplos válidos: $10.00, 10$00, 1,000$50, $1,000.00, 10$00.50
        # Ejemplos inválidos: 00$, $00, 00$00, 10.00$
        # Patrón para símbolo al inicio
        pattern_start = rf"^{currency}\s*(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?$"
        # Patrón para símbolo en medio, rodeado de números
        pattern_middle = (
            rf"^(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)"
            rf"\s*{currency}\s*"
            rf"(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?$"
        )
        # No aceptar símbolo al final
        pattern_end = rf"^(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?\s*{currency}$"
        # Verificar si hay dos patrones válidos seguidos (ej: "$10.00 $60.00")
        multi_pattern = (
            rf"^(\s*{currency}\s*(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?\s*){{2,}}$"
        )
        # Rechazar si termina con símbolo de moneda
        if re.match(pattern_end, s):
            return False
        # Rechazar si la cantidad es "00" en cualquier parte
        cantidades = re.findall(rf"{currency}?\s*(\d+)(?:[.,]\d+)?\s*{currency}?", s)
        if any(c == "00" for c in cantidades):
            return False
        # Aceptar si cumple patrón de inicio, patrón de en medio, o múltiples patrones válidos
        return (
            bool(re.match(pattern_start, s)) or
            bool(re.match(pattern_middle, s)) or
            bool(re.match(multi_pattern, s))
        )

    def _is_quantitative(self, token: str) -> bool:
        return self._is_currency_amount(token) or self._is_decimal_number(token)
                
    def _save_ocr_raw(self, context: Dict[str, Any], final_results: Dict[str, str], file_name: str):
        from services.output_service import save_json
        import os

        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir: str = os.path.join(path, "semantic_words")
            json_file_name = f"{os.path.splitext(file_name)[0]}.json"
            save_json(final_results, output_dir, json_file_name)
        
        if output_paths:
            logger.debug(f"OCR Raw results para '{file_name}' guardado en {len(output_paths)} ubicaciones.")
            
            