# PerfectOCR/core/workers/ocr/text_corrector.py
import logging
import dataclasses
from typing import Dict, Any
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, SemanticClassification
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class TextCorrector(OCRAbstractWorker):
    """
    Corrector textual quirúrgico que realiza reemplazos especializados de caracteres
    según el tipo semántico de cada polígono.
    
    Operación:
    - Recibe todos los polígonos del manager.
    - Según la clasificación semántica aplica correcciones específicas.
    - Solo hace reemplazos de caracteres, no corrección ortográfica.
    - Es recursivo: itera sobre todos los polígonos aplicando correcciones especializadas.
    """
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("text_corrector", {})
        self.char_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"]
        
    def _load_correction_rules(self):
        """
        Carga las reglas de corrección desde la configuración.
        Cada tipo semántico tiene su propio diccionario de reemplazos.
        """
        # Correcciones para texto numérico
        self.numeric_corrections: Dict[str, str] = {
            "O": "0",
            "o": "0",
            "I": "1",
            "i": "1",
            "|": "1",
            "l": "1",
            "S": "$",
            # "s": "5",
            # "G": "6",
            "B": "8",
            "Z": "2",
            "z": "2",
            "j": "9"
        }
        
        # Correcciones para texto cuantitativo (números con unidades)
        self.quantitative_corrections: Dict[str, str] = self.numeric_corrections
        
        # Correcciones para texto descriptivo
        self.descriptive_corrections: Dict[str, str] = {"$": "S"}
        
        logger.debug("Reglas de corrección quirúrgica cargadas")
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el proceso de corrección quirúrgica sobre todos los polígonos.
        
        Args:
            context: Contexto de ejecución
            manager: DataFormatter con los polígonos a corregir
            
        Returns:
            True si el proceso se completó exitosamente
        """
        confidence_threshold = self.worker_config.get("confidence_threshold")

        conf_threshold = float(confidence_threshold *100.0)
        self._load_correction_rules()
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCorrector: No hay polígonos para procesar.")
            return True
            
        polygons_in: Dict[str, Polygons] = manager.workflow.polygons
        corrected_polygons: Dict[str, Polygons] = {}
        correction_stats = {
            "numeric": 0,
            "quantitative": 0,
            "code": 0,
            "descriptive": 0,
            "total_corrections": 0,
            "skipped_high_confidence": 0
        }
        
        sorted_poly_ids = sorted(polygons_in.keys())
        
        # Procesar cada polígono recursivamente
        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            original_text = polygon.ocr_text or ""
            confidence = polygon.ocr_confidence or 0.0
            
            if not original_text.strip():
                corrected_polygons[poly_id] = polygon
                continue
            
            # Filtro de confianza: si está por encima del umbral, no corregir
            if confidence > conf_threshold:
                corrected_polygons[poly_id] = polygon
                correction_stats["skipped_high_confidence"] += 1
                continue
                
            # Aplicar corrección según tipo semántico
            corrected_text = self._apply_corrections(
                text=original_text,
                semantic_clasification=polygon.semantic_clasification,
                polygon_id=poly_id
            )
            
            # Si hubo cambios, actualizar el polígono
            if corrected_text != original_text:
                updated_polygon = dataclasses.replace(
                    polygon,
                    ocr_text=corrected_text,
                    was_refined=True
                )
                corrected_polygons[poly_id] = updated_polygon
                sc = polygon.semantic_clasification
                semantic_type = "numeric" if sc.numeric else "quantitative" if sc.quantitative else "descriptive" if sc.descriptive else "code"
                correction_stats[semantic_type] += 1
                correction_stats["total_corrections"] += 1
                
                logger.debug(
                    f"Corrección {poly_id}: "
                    f"Tipo: {semantic_type} | "
                    f"Confianza: {confidence:.4f} | "
                    f"Original: '{original_text}' → Corregido: '{corrected_text}'"
                )
            else:
                corrected_polygons[poly_id] = polygon
                
        # Actualizar el manager con los polígonos corregidos
        manager.workflow.polygons = corrected_polygons
        
        logger.debug(
            f"Corrección textual - "
            f"Total: {correction_stats['total_corrections']} | "
            f"Alta confianza omitidos: {correction_stats['skipped_high_confidence']} | "
            f"Numeric: {correction_stats['numeric']} | "
            f"Quantitative: {correction_stats['quantitative']} | "
            f"Code: {correction_stats['code']} | "
            f"Descriptive: {correction_stats['descriptive']}"
        )
                    
        return True

    def _apply_corrections(
        self,
        text: str,
        semantic_clasification: SemanticClassification,
        polygon_id: str
    ) -> str:
        """
        Aplica las correcciones quirúrgicas según el tipo semántico.
        Solo corrige caracteres AISLADOS (sin vecinos del mismo tipo).
        
        Args:
            text: Texto original a corregir
            semantic_clasification: Tipo semántico del polígono
            polygon_id: Identificador del polígono
            
        Returns:
            Texto corregido
        """
        if not text or not semantic_clasification:
            return text
        
        # No corregir
        if not (semantic_clasification.numeric or semantic_clasification.quantitative or semantic_clasification.descriptive):
            semantic_type = "code" if semantic_clasification.code else "umd" if semantic_clasification.umd else "unknown"
            logger.debug(f"Omitiendo corrección para tipo '{semantic_type}' ({polygon_id}: {text} )")
            return text
                    
        # Seleccionar el diccionario de correcciones apropiado
        corrections_map = self._get_corrections_map(semantic_clasification)
        
        if not corrections_map:
            logger.warning(
                f"No hay reglas de corrección para tipo: {semantic_clasification} "
                f"(poly_id: {polygon_id})"
            )
            return text
            
        # Aplicar reemplazos quirúrgicos solo si el carácter está AISLADO
        corrected_chars = list(text)
        
        for i, char in enumerate(text):
            if char not in corrections_map:
                continue

            # Verificar si el carácter está AISLADO (sin vecinos del mismo tipo)
            if not self._is_isolated(text, i):
                continue

            # Log antes de corregir
            logger.info(
                f"{polygon_id} Corrigiendo: '{char}' → '{corrections_map[char]}' "
                f"en texto original: '{text}'"
            )

            # Aplicar corrección
            corrected_chars[i] = corrections_map[char]
            
        return ''.join(corrected_chars)

    def _is_isolated(self, text: str, index: int) -> bool:
        """
        Verifica si un carácter está AISLADO (sin vecinos del mismo tipo).
        Ignora espacios al buscar vecinos.
        
        Args:
            text: Texto completo
            index: Índice del carácter a verificar
            
        Returns:
            True si el carácter está aislado (sin vecinos del mismo tipo)
        """
        if index < 0 or index >= len(text):
            return False
            
        current_char = text[index]
        current_is_digit = current_char in self.char_num
        current_is_alpha = current_char.isalpha()
        
        # Si no es letra ni número, no aplicar corrección
        if not current_is_digit and not current_is_alpha:
            return False
        
        # Buscar vecino izquierdo (ignorando espacios)
        left_neighbor = None
        for i in range(index - 1, -1, -1):
            if text[i] != ' ':
                left_neighbor = text[i]
                break
        
        # Buscar vecino derecho (ignorando espacios)
        right_neighbor = None
        for i in range(index + 1, len(text)):
            if text[i] != ' ':
                right_neighbor = text[i]
                break
        
        # Verificar si NINGÚN vecino es del mismo tipo (está aislado)
        has_left_match = False
        has_right_match = False
        
        if left_neighbor:
            if current_is_digit and left_neighbor in self.char_num:
                has_left_match = True
            elif current_is_alpha and left_neighbor.isalpha():
                has_left_match = True
        
        if right_neighbor:
            if current_is_digit and right_neighbor in self.char_num:
                has_right_match = True
            elif current_is_alpha and right_neighbor.isalpha():
                has_right_match = True
        
        # Está aislado si NO tiene ningún vecino del mismo tipo
        return not (has_left_match or has_right_match)
        
    def _get_corrections_map(self, semantic_clasification: SemanticClassification) -> Dict[str, str]:
        """Devuelve el mapa de correcciones para un tipo semántico dado."""
        if semantic_clasification.numeric:
            return self.numeric_corrections
        if semantic_clasification.quantitative:
            return self.quantitative_corrections
        if semantic_clasification.descriptive:
            return self.descriptive_corrections
        return {}