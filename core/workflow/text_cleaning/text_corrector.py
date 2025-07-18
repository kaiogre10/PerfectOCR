# PerfectOCR/core/text_cleaning/text_corrector.py
import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class TextCorrector:
    """
    Correcto de texto específico para errores OCR.
    Detecta abreviaturas y corrige solo palabras normales.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.use_symspell = self.config.get('use_symspell', True)
        self.symspell_threshold = self.config.get('symspell_threshold', 0.8)
        self._symspell_corrector = None
        self.use_phonotactic_rules = self.config.get('use_phonotactic_rules', True)
        self.dictionary_path = config.get('dictionary_path', 'data/dictionary/diccionario_100k.txt')
        self.names_path = config.get('names_path', 'data/dictionary/nombres_frecuentes.txt')
        self.word_dictionary = self._load_dictionary(self.dictionary_path)
        self.names_dictionary = self._load_dictionary(self.names_path)
        self.common_verb_endings = {
            'ar', 'er', 'ir', 'ando', 'endo', 'ado', 'ido',
            'aba', 'ía', 'ará', 'erá', 'irá', 'aste', 'iste'
        }
        self.letter_freq = {
            'e': 0.1368, 'a': 0.1253, 'o': 0.0868, 'i': 0.0625,
            's': 0.0798, 'n': 0.0671, 'r': 0.0687, 'u': 0.0393,
            'l': 0.0497, 't': 0.0463, 'd': 0.0586, 'c': 0.0468,
            'p': 0.0251, 'm': 0.0315, 'b': 0.0142, 'g': 0.0101,
            'y': 0.0109, 'f': 0.0070, 'v': 0.0090, 'k': 0.0002,
            'h': 0.0070, 'q': 0.0088, 'j': 0.0044, 'x': 0.0022,
            'z': 0.0052, 'w': 0.0001
        }
        

    def _load_symspell(self):
        """Carga SymSpell de forma lazy."""
        if self._symspell_corrector is not None:
            return self._symspell_corrector
        
        try:
            from symspellpy import SymSpell, Verbosity
            self._symspell_corrector = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            logger.info("SymSpell cargado exitosamente")
            return self._symspell_corrector
        except ImportError:
            logger.warning("SymSpell no disponible. Instalar con: pip install symspellpy")
            return None
        except Exception as e:
            logger.error(f"Error cargando SymSpell: {e}")
            return None

    def _is_abbreviation_by_phonotactics(self, word: str) -> bool:
        if not self.use_phonotactic_rules:
            return False
        
        word_lower = word.lower()
        product_abbreviations = [
            'kg', 'cm', 'mm', 'm', 'l', 'ml', 'g', 'mg', 'oz', 'lb', 'pz', 'pcs',
            'tl', 'tl', 'tlf', 'tel', 'fax', 'cel', 'mob', 'sk', 'sku', 'upc',
            'pal', 'red', 'delg', 'punt', 'clav', 'torn', 'tuerc', 'arand',
            'caja', 'bols', 'paq', 'unid', 'doc', 'docena', 'caja', 'caj'
        ]
        
        if word_lower in product_abbreviations:
            return True

        abbreviation_patterns = [
            r'^[a-z]+\d+$',      # PAL0, RED1
            r'^\d+[a-z]+$',      # 1KG, 2CM
            r'^[a-z]+/[a-z0-9]+$',  # C/10, A/B
            r'^[a-z]+\.[a-z\.]*$',  # P.U.C.D, A.B.C
            r'^\d+x\d+$',        # 25X25, 4X60
            r'^[a-z]{1,2}\d{1,2}$',  # A1, B2, AB12
            r'^[a-z]+[0-9]+[a-z]*$',  # PAL0, RED1A
            r'^[0-9]+[a-z]+[0-9]*$',  # 1KG, 2CM3
            r'^[a-z]+[0-9]+[a-z]+$',  # PAL0A, RED1B
        ]
        
        for pattern in abbreviation_patterns:
            if re.match(pattern, word_lower):
                return True
        
        # Palabras sin vocales
        if not re.search(r'[aeiouáéíóúü]', word_lower):
            return True
        
        # Mezcla de mayúsculas y minúsculas
        if re.search(r'[a-z][A-Z]|[A-Z][a-z]', word):
            return True        
        return False

    def _correct_with_symspell(self, word: str) -> Optional[str]:
        if not self.use_symspell:
            return None
        
        corrector = self._load_symspell()
        if corrector is None:
            return None
        
        try:
            from symspellpy import Verbosity
            suggestions = corrector.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            
            if suggestions:
                best_suggestion = suggestions[0]
                if best_suggestion.distance <= 1 and best_suggestion.count > 1:
                    return best_suggestion.term
            
            return None
            
        except Exception as e:
            logger.debug(f"Error en corrección SymSpell para '{word}': {e}")
            return None

    def _generate_semantic_variations(self, word: str) -> List[str]:
        """Genera variaciones semánticas de la palabra usando patrones comunes de OCR."""
        variations = []
        
        char_variations = {
            'a': ['o', 'e'], 'o': ['a', 'e'], 'e': ['a', 'o'],
            'i': ['l', '1'], 'l': ['i', '1'], '1': ['i', 'l'],
            'n': ['m', 'h'], 'm': ['n', 'h'], 'h': ['n', 'm'],
            'r': ['n', 'm'], 'u': ['v', 'w'], 'v': ['u', 'w'],
            's': ['5', 'f'], '5': ['s', 'f'], 'f': ['s', '5'],
            't': ['7', 'f'], '7': ['t', 'f'],
            'g': ['9', 'q'], '9': ['g', 'q'], 'q': ['g', '9'],
            'z': ['2', 's'], '2': ['z', 's'],
            'b': ['8', 'h'], '8': ['b', 'h'],
            'c': ['e', 'o'], 'd': ['b', 'p'], 'p': ['b', 'd']
        }
        
        for i, char in enumerate(word):
            if char in char_variations:
                for replacement in char_variations[char]:
                    variation = word[:i] + replacement + word[i+1:]
                    variations.append(variation)
        
        return variations
    
    def _simple_lemmatize(self, word: str) -> str:
        """
        Lematización simple removiendo sufijos verbales comunes.
        Solo para palabras NO encontradas en el diccionario base.
        """
        word_lower = word.lower()
        if self._word_exists(word):
            return word_lower
        for ending in sorted(self.common_verb_endings, key=len, reverse=True):
            if word_lower.endswith(ending) and len(word_lower) > len(ending) + 2:
                stem = word_lower[:-len(ending)]
                for base_ending in ['ar', 'er', 'ir']:
                    potential_verb = stem + base_ending
                    if potential_verb in self.word_dictionary:
                        return potential_verb        
        return word_lower

    def _word_exists(self, word: str) -> bool:
        word_lower = word.lower()
        if word_lower in self.word_dictionary or word_lower in self.names_dictionary:
            return True
        lemmatized = self._simple_lemmatize(word_lower)
        return lemmatized in self.word_dictionary

    def _weighted_levenshtein(self, s1: str, s2: str) -> float:
        """Implementación completa de distancia Levenshtein con pesos OCR."""
        substitution_costs = {
            ('0', 'o'): 0.3, ('o', '0'): 0.3,
            ('1', 'l'): 0.3, ('l', '1'): 0.3, ('1', 'i'): 0.3, ('i', '1'): 0.3,
            ('5', 's'): 0.3, ('s', '5'): 0.3,
            ('6', 'g'): 0.3, ('g', '6'): 0.3,
            ('8', 'b'): 0.3, ('b', '8'): 0.3,
            ('9', 'g'): 0.3, ('g', '9'): 0.3,
            ('c', 'e'): 0.4, ('e', 'c'): 0.4,
            ('n', 'm'): 0.4, ('m', 'n'): 0.4,
            ('u', 'v'): 0.4, ('v', 'u'): 0.4,
            ('rn', 'm'): 0.3, ('m', 'rn'): 0.3 
        }
        
        len1, len2 = len(s1), len(s2)
        dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Llenar matriz
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Sin costo
                else:
                    # Costo de sustitución (con pesos OCR)
                    sub_cost = substitution_costs.get((s1[i-1], s2[j-1]), 1.0)
                    
                    dp[i][j] = min(
                        dp[i-1][j] + 1.0,           # Eliminación
                        dp[i][j-1] + 1.0,           # Inserción
                        dp[i-1][j-1] + sub_cost     # Sustitución ponderada
                    )
        
        return dp[len1][len2]

    def _generate_weighted_candidates(self, word: str) -> List[Tuple[str, float]]:
        """Genera candidatos optimizados usando solo el diccionario de 100k."""
        candidates = []
        word_lower = word.lower()

        length_threshold = 2
        filtered_dict = [w for w in self.word_dictionary 
                        if abs(len(w) - len(word_lower)) <= length_threshold]
        
        # Buscar candidatos con distancia <= 2.0
        for dict_word in filtered_dict:
            distance = self._weighted_levenshtein(word_lower, dict_word)
            if distance <= 2.0:
                # Bonus por frecuencia de letras
                prob_bonus = self._word_probability(dict_word) * 0.1
                final_score = distance - prob_bonus
                candidates.append((dict_word, final_score))
        
        # Ordenar por menor distancia ajustada
        candidates.sort(key=lambda x: x[1])
        return candidates[:5]

    def correct_text(self, text: str, confidence: Optional[float] = None) -> str:
        """
        Corrección mejorada con diccionario optimizado.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Solo corregir si confianza es baja
        if confidence is None or confidence >= self.low_confidence_threshold:
            return text
        
        try:
            words = text.split()
            if len(words) < 1:
                return text
            
            corrected_words = []
            
            for word in words:
                # Paso 1: Verificar si es abreviatura o código
                if self._is_abbreviation_by_phonotactics(word):
                    corrected_words.append(word)
                    continue
                
                # Paso 2: Verificar si existe en diccionario (con lematización)
                if self._word_exists(word):
                    corrected_words.append(word)
                    continue
                
                # Paso 3: Buscar candidatos por distancia ponderada
                candidates = self._generate_weighted_candidates(word)
                
                if candidates:
                    best_candidate = candidates[0][0]
                    best_distance = candidates[0][1]
                    
                    # Solo aplicar si la distancia es razonable
                    if best_distance <= 1.5:
                        logger.debug(f"Corrección por diccionario: '{word}' → '{best_candidate}' (dist: {best_distance:.2f})")
                        corrected_words.append(best_candidate)
                        continue
                
                # Paso 4: Fallback a SymSpell si está disponible
                symspell_correction = self._correct_with_symspell(word)
                if symspell_correction and symspell_correction != word:
                    corrected_words.append(symspell_correction)
                    continue
                
                # Sin corrección aplicable
                corrected_words.append(word)
            
            return " ".join(corrected_words)
            
        except Exception as e:
            logger.error(f"Error en corrección de texto: {e}")
            return text

    def get_correction_stats(self, original_text: str, corrected_text: str) -> Dict[str, Any]:
        """Obtiene estadísticas de la corrección aplicada."""
        return {
            'original_length': len(original_text),
            'corrected_length': len(corrected_text),
            'text_changed': original_text != corrected_text,
            'abbreviations_preserved': True,
            'correction_type': 'semantic_and_orthographic'
        }

    def _load_dictionary(self, path: str) -> Set[str]:
        """Carga diccionario desde archivo."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except Exception as e:
            logger.warning(f"No se pudo cargar diccionario {path}: {e}")
            return set()
            
    def _word_probability(self, word: str) -> float:
        """Calcula probabilidad lingüística de una palabra."""
        prob = 1.0
        for c in word.lower():
            prob *= self.letter_freq.get(c, 0.001)
        return prob