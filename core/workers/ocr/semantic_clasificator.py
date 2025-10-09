# core/workers/ocr/semantic_clasificator.py
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
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        
        logger.debug("Clasificador inciado")
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para preocesar")
                return False
                
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            
            final_results: Dict[str, str] = self._clasify_words(polygons)
            
            classified_count = 0

            for poly_id, semantic_type in final_results.items():
                if poly_id in polygons:
                    polygon = polygons[poly_id]
                    classified_count += 1

            logger.debug(f"Total clasificados: {classified_count}")
            for poly_id, semantic_type in final_results.items():
                if poly_id in polygons:
                    polygon = polygons[poly_id]
                    logger.debug(f"{poly_id}: {semantic_type} | texto: '{polygon.ocr_text or ''}'")
                    
            manager.update_semantic_type(final_results)
            return True

        except Exception as e:
            logger.warning(f"Error en el clasiicador{e}", exc_info=True)
            return False
            
    def _clasify_words(self, polygons: Dict[str, Polygons]) -> Dict[str, str]:
        numeric_range = self.worker_config.get("numeric", [])
        code_range = self.worker_config.get("code", [])

        def norm(r): return (min(r[0], r[1]), max(r[0], r[1])) # type: ignore
        n_min, n_max = norm(numeric_range)
        c_min, c_max = norm(code_range) # type: ignore

        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, str] = {}

        for pid, s in texts.items():
            chars = [ch for ch in s if not ch.isspace()]
            total = len(chars)
            pct = (sum(1 for ch in chars if ch.isdigit()) / total) * 100.0 if total else 0.0

            # Verificar si existe patrón cuantitativo PRIMERO
            tokens = [t for t in (s or "").split() if t]
            if not tokens:  # Si no hay espacios, usar el texto completo
                tokens = [s]
            
            has_quantitative = any(self._is_quantitative(t) for t in tokens)
            
            # 0. Si tiene patrón cuantitativo, clasificar directamente como quantitative
            if has_quantitative:
                semantic = "quantitative"
            # 1. Si el porcentaje está en rango numérico, clasificar como numeric
            elif n_min <= pct <= n_max:
                semantic = "numeric"
            # 2. Si no es numeric, verificar si es descriptive
            elif pct < c_min:
                semantic = "descriptive"
            # 3. Si no es ni numeric ni descriptive, entonces es code
            else:
                semantic = "code"

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
                
    def _has_quantitative_pattern(self, s: str) -> bool:
        """
        Verifica si el texto contiene algún patrón cuantitativo (moneda o decimal).
        Busca patrones dentro del texto, incluso si hay caracteres basura.
        """
        
        if not s or not s.strip():
            return False
        
        # Verificar tokens separados por espacios
        tokens = [t for t in s.split() if t]
        if tokens:
            if any(self._is_quantitative(t) for t in tokens):
                return True
        
        # Si no hay espacios o no se encontró en tokens, buscar patrón dentro del string
        # Patrón para números decimales (con coma o punto como separador)
        decimal_pattern = r'\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}'  # Ej: 1.275.00, 1,275.00
        
        # Patrón para moneda ($ € £ ¥ ¢ seguido de números)
        currency_pattern = r'[$€£¥¢]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?'
        
        # Buscar cualquiera de estos patrones en el texto
        if re.search(decimal_pattern, s) or re.search(currency_pattern, s):
            return True
        
        # Último recurso: verificar el string completo limpio
        return self._is_quantitative(s)
                