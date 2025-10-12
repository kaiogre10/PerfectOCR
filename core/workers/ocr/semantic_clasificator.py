# core/workers/ocr/semantic_clasificator.py
import logging
import re
import dataclasses
from typing import Dict, Any, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, SemanticClassification
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("semantic_clasificator", {})
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter, filter_modified: bool = False) -> bool:
        """
        Clasifica polígonos semánticamente
        """
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
            final_results: Dict[str, SemanticClassification] = self._clasify_words(polygons_to_classify)
            
            classified_count = len(final_results)
            logger.debug(f"Total clasificados: {classified_count}")
            
            # for poly_id, semantic_obj in final_results.items():
            #     if poly_id in all_polygons:
            #         polygon = all_polygons[poly_id]
            #         active_fields = [field for field in ['quantitative', 'umd', 'rfc', 'numeric', 'descriptive', 'code'] 
            #                        if getattr(semantic_obj, field)]
            #         logger.debug(f"{poly_id}: {active_fields} | texto: '{polygon.ocr_text or ''}'")
            
            # Actualizar semantic_type Y resetear was_refined si es modo filtrado
            manager.update_semantic_clasification(final_results, reset_refined=filter_modified)
            
            return True

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
            
    def _clasify_words(self, polygons: Dict[str, Polygons]) -> Dict[str, SemanticClassification]:
        numeric_range = self.worker_config.get("numeric", [])
        code_range = self.worker_config.get("code", [])

        def norm(r: Tuple[float, float]): return (min(r[0], r[1]), max(r[0], r[1]))
        n_min, n_max = norm(numeric_range)
        c_min, c_max = norm(code_range) #type: ignore
        
        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}
        final_results: Dict[str, SemanticClassification] = {}

        for pid, s in texts.items():
            chars = [ch for ch in s if not ch.isspace()]
            total = len(chars)
            pct = (sum(1 for ch in chars if ch.isdigit()) / total) * 100.0 if total else 0.0

            # Verificar si existe patrón cuantitativo PRIMERO
            tokens = [t for t in (s or "").split() if t]
            if not tokens:  # Si no hay espacios, usar el texto completo
                tokens = [s]
            
            has_quantitative = any(self._is_quantitative(t) for t in tokens)
            
            # Crear objeto SemanticClassification con todos los campos en False
            semantic_clasification = SemanticClassification(
                quantitative=False,
                umd=False,
                rfc=False,
                numeric=False,
                descriptive=False,
                code=False
            )
            
            # Si tiene patrón cuantitativo, clasificar directamente como quantitative
            if has_quantitative:
                semantic_clasification = dataclasses.replace(semantic_clasification, quantitative=True)
            # Busca unidades de medida
            elif self.find_umd(s):
                semantic_clasification = dataclasses.replace(semantic_clasification, umd=True)
            # Busca el RFC
            elif self.find_rfc(s):
                semantic_clasification = dataclasses.replace(semantic_clasification, rfc=True)
            # Si el porcentaje está en rango numérico, clasificar como numeric
            elif n_min <= pct <= n_max:
                semantic_clasification = dataclasses.replace(semantic_clasification, numeric=True)
            # Si no es numeric, verificar si es descriptive
            elif pct < c_min:
                semantic_clasification = dataclasses.replace(semantic_clasification, descriptive=True)
            # Si no es ni numeric ni descriptive, entonces es code
            else:
                semantic_clasification = dataclasses.replace(semantic_clasification, code=True)

            final_results[pid] = semantic_clasification

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

        currency_symbols = "$¢"
        currency = r"[$¢]"

        # Simple no-regex shortcut: if contains a currency and at least 1 digit after it
        for sym in currency_symbols:
            idx = s.find(sym)
            if idx != -1:
                after = s[idx+1 : ]
                if any(c.isdigit() for c in after):
# No letras entre el símbolo y el número inmediato y no termina con símbolo y no es "00" después del símbolo y no digitos antes y símbolo al final
                    maybe_amt = after.lstrip()
                    # quick fail for 00 only amount
                    possible_num = ""
                    for ch in maybe_amt:
                        if ch.isdigit() or ch in ",.":
                            possible_num += ch
                        else:
                            break
                    if possible_num == "00":
                        return False
                    # Evitar simbolo al final
                    if idx == len(s)-1:
                        return False
                    # Evitar cantidades tipo "10.00$"
                    if idx == 0 or (s[:idx].strip().isdigit() == False):
                        return True
                    # En casos tipo "1,000$50", permitir
                    before = s[:idx]
                    if any(c.isdigit() for c in before):
                        return True
        # Si no coincide el shortcut, sigue el método previo por regex
        # El símbolo puede estar al inicio o en medio, pero NO al final
        pattern_start = rf"^{currency}\s*(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?$"
        pattern_middle = (
            rf"^(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)"
            rf"\s*{currency}\s*"
            rf"(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?$"
        )
        pattern_end = rf"^(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?\s*{currency}\s*$"
        multi_pattern = (
            rf"^(\s*{currency}\s*(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?\s*){{2,}}$"
        )
        if re.match(pattern_end, s):
            return False
        cantidades = re.findall(rf"{currency}?\s*(\d+)(?:[.,]\d+)?\s*{currency}?", s)
        if any(c == "00" for c in cantidades):
            return False
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

    def find_umd(self, s: str) -> bool:
        try:
            if not s or not s.strip():
                return False

            # Regex Maestra para capturar la unidad y el valor
            # Nota: La diagonal debe ser escapada: \/
            umd_patterns = r'\b(\d+([,\.]\d+)?)\s*(/)?\s*(k(g(r)?|ilo(s)?)|g(r|ramo(s)?|m)?|mg|m(t(r)?|etro(s)?|2|\\^2)?|cm|l(t(r)?|itro(s)?|ml)?|cc|ud(s)?|pza(s)?|cj(s)?)\b'
            if re.search(umd_patterns, s):
                 return True
            return False

        except Exception as e:
            logger.info(f"No se hallaron unidades de medida: {e}", exc_info=True)
            return False

    def find_rfc(self, s: str) -> bool:
        try:
            if not s or not s.strip():
                return False
            rfc_code = r'^([A-ZÑ&]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$'
            rfc_word = r'\b(R\.?F\.?C\.?)\b'

            # Busca primero el patrón corto
            if re.search(rfc_word, s):
                # Si lo encuentra, busca el patrón largo
                if re.search(rfc_code, s):
                    return True
                else:
                    return False
            # Si no encuentra el corto, busca el largo directamente
            if re.search(rfc_code, s):
                return True
            return False
        except Exception as e:
            logger.info(f"Error buscando RFC: {e}", exc_info=True)
            return False