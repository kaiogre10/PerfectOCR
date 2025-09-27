import os
import re
import logging
import pickle
import unicodedata
import numpy as np
from typing import List, Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str, project_root: str):
        self.model: Dict[str, Any] = self._load_model(model_path)

        # self.project_root = project_root
        # self.model_path = model_path
        self._active = "standard"
        self._apply_active_model()

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if not isinstance(self.model, dict):
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            return self.model
        except Exception as e:
            logger.info(f"Error al cargar el modelo{e}", exc_info=True)

    def available_models(self) -> List[str]:
        return ["standard"]

    def set_active_model(self, name: str) -> bool:
        if name == "standard":
            self._active = "standard"
            return True
        logger.warning("Modelo solicitado no disponible: %s", name)
        return False

    def _apply_active_model(self):
        params: Dict[str, Any] = self.model.get("params", {})
        self.global_words = self.model.get("global_words", {})
        self.global_filter: Dict[str, Any] = self.model.get("global_filter", {})
        self.gngr: Tuple[int, int] = params.get("char_ngram_global", [])
        self.ngr: Tuple[int, int] = params.get("char_ngram_range", [])
        self.threshold = params.get("threshold_similarity", [])
        self.thresholds_by_len: List[Tuple[int, int, float]] = [tuple(item) for item in params.get("thresholds_by_len", [])]
        self.weights_by_n: List[Tuple[int, int, float]] = [tuple(item) for item in params.get("weights_by_n", [])]
        self.window_flex = params.get("window_flexibility", 3)
        self.global_filter_threshold = float(params.get("global_filter_threshold", []))
        self.grams_index = self.model.get("self.grams_index", {})
        raw_global = self.global_filter.get("global_ngrams", [])
        # posibilidad de que el trainer haya incluido un dict de frecuencias y el counter
        self.global_counter = self.global_filter.get("global_counter")
        raw_freqs = self.global_filter.get("global_ngram_freqs")
        try:
            if isinstance(raw_freqs, dict):
                # trainer devolvió un dict ngram -> freq
                self.global_ngrams_freq = {g: float(f) for g, f in raw_freqs.items()}
            else:
                # loader previo (lista de tuplas) o vacío
                self.global_ngrams_freq = {g: float(f) for g, f in raw_global}
            maxf = max(self.global_ngrams_freq.values()) if self.global_ngrams_freq else 1.0
            self.global_ngrams_freq_norm = {g: (f / maxf) for g, f in self.global_ngrams_freq.items()}
        except Exception:
            self.global_ngrams_freq = {}
            self.global_ngrams_freq_norm = {}
        self.variant_to_field = self.model.get("variant_to_field", {})
        try:
            self.global_ngrams_freq = {g: float(f) for g, f in raw_global}
            maxf = max(self.global_ngrams_freq.values()) if self.global_ngrams_freq else 1.0
            self.global_ngrams_freq_norm = {g: (f / maxf) for g, f in self.global_ngrams_freq.items()}
        except Exception:
            self.global_ngrams_freq = []
            self.global_ngrams_freq_norm = []

        # índice precomputado desde el pickle
        self.grams: List[Dict[int, set[str]]] = []
        self.lengths: List[int] = []

        if self.grams_index:
            for entry in self.grams_index:
                length = int(entry.get("len", 0))
                self.gmap_raw: Dict[int, List[str]] = entry.get("grams", {}) 
                self.gmap_sets = {}
                for n in range(self.ngr[0], self.ngr[1] + 1):
                    self.gmap_sets[n] = set(self.gmap_raw.get(n, []))
                self.lengths.append(length)
                self.grams.append(self.gmap_sets)
        else:
            self.grams = []
            self.lengths: List[int]  = []
            for w in self.global_words:
                self.length = len(w)
                self.lengths.append(self.length)
                self.gmap_sets: Dict[int, set[str]] = {}
                for n in range(self.ngr[0], self.ngr[1] + 1):
                    self.gmap_sets[n] = set(self._ngrams(w, n))
                self.grams.append(self.gmap_sets)

        self.buckets: Dict[int, List[int]] = self.model.get("buckets_by_len") or {}
        self.buckets_by_len: Dict[int, List[int]] = {int(k): list(v) for k, v in self.buckets.items()}
        if not self.buckets_by_len:
            for i, length in enumerate(self.lengths):
                self.buckets_by_len.setdefault(length, []).append(i)

    def _len_threshold(self, length: int) -> float:
        """Obtiene umbral por longitud del candidato"""
        try:
            for start, end, value in self.thresholds_by_len:
                if start <= length <= end:
                    return value
            else:
                return 1.0
        except Exception as e:
            logger.error(f"error calculando umrales de largo: {e}", exc_info=True)

    def find_keywords(self, text: List[str] | str, threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        """Busca todas las palabras clave presentes en el string, no solo la mejor."""
        try:
            # Usar el threshold proporcionado o el del modelo como filtro final
            final_threshold = threshold if threshold is not None else self.threshold
            
            single = False
            if isinstance(text, str):
                text = [text]
                single = True

            results: List[Dict[str, Any]] = []
            for s in text:
                q = self._normalize(s)
                if not q:
                    continue

                if not self._is_potential_keyword(q):
                    continue

                for i in range(len(self.global_words)):
                    cand = self.global_words[i]
                    cand_len = len(cand)
                    min_w = max(1, cand_len - self.window_flex)
                    if min_w > len(q):
                        continue
                    max_w = min(len(q), cand_len + self.window_flex)
                    grams_cand = self.grams[i]
                    try:
                        for w in range(min_w, max_w + 1):
                            if w > len(q):
                                break
                            for j in range(0, len(q) - w + 1):
                                sub = q[j:j + w]
                                grams_sub = self._build_query_grams(sub)
                                # Solo asignar similitud perfecta si es la misma palabra completa (misma longitud)
                                if sub == cand and len(sub) == cand_len:
                                    ngram_score: float = 1.0
                                else:
                                    ngram_score: float = self._score_binary_cosine_multi_n(grams_cand, grams_sub)
                                    len_ratio = max(len(sub), cand_len) / max(1, min(len(sub), cand_len))
                                    if len_ratio >= 2.0:
                                        penalizacion = min(len(sub), cand_len) / max(len(sub), cand_len)
                                        ngram_score *= penalizacion                                
                                if ngram_score >= final_threshold:
                                    key_field = self.variant_to_field.get(cand)
                                    results.append({
                                        "key_field": key_field,
                                        "word_found": cand,
                                        "similarity": float(ngram_score),
                                        "query": q
                                    })
                                    return results
                    except Exception as e:
                        logger.error(f"Error en find_keywords: {e}", exc_info=True)

            elapsed = None
            if single:
                return results if results else []
            return results

        except Exception as e:
            logger.error(f"Error en find_keywords: {e}", exc_info=True)

    def _normalize(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
        # s = re.sub(r"[^a-zA-Z0-9]", "", s)
        return s

    def _build_query_grams(self, q: str) -> Dict[int, set[str]]:
        """Construye n-gramas de la consulta"""
        gq: Dict[int, set[str]] = {}
        for n in range(self.ngr[0], self.ngr[1] + 1):
            gq[n] = set(self._ngrams(q, n))
        return gq

    def _ngrams(self, q: str, n: int) -> List[str]:
        try:
            if n <= 0 or not q:
                return []
            if len(q) < n:
                return []
            return [q[i:i+n] for i in range(len(q) - n + 1)]
        except Exception as e:
            logger.info(f"Error construyendo n-gramas: {e}", exc_info=True)
            return []

    def _ngram_similarity(self, a: str, b: str) -> float:
        """Calcula la similitud entre dos n-gramas como la proporción de caracteres iguales."""
        try:
            matches = float(sum(1 for x, y in zip(a, b) if x == y))
        # Normaliza por la longitud máxima para manejar n-gramas de longitudes distintas si fuera el caso.
            return matches / float(max(len(a), len(b)))
        except Exception as e:
            logger.info(f"Error comprobando texto: {e}", exc_info=True)
            return 0.0

    def _binary_cosine(self, size_a: int, size_b: int, soft_intersection: float) -> float:
        """Calcula coseno binario entre dos conjuntos"""
        if size_a == 0 or size_b == 0:
            return 0.0
        return soft_intersection / float((size_a * size_b) ** 0.5)

    def _score_binary_cosine_multi_n(self, grams_a: Dict[int, set[str]], grams_b: Dict[int, set[str]]) -> float:
        """Calcula score ponderado por coseno binario multi-n-grama"""
        num = 0.0
        den = 0.0

        for n in range(self.ngr[0], self.ngr[1] + 1):
            A = grams_a.get(n, set())
            B = grams_b.get(n, set())
            w: float = self._get_weight_by_n(n)

            if not A or not B:
                den += w
                continue

            soft_intersection = 0.0
            for gram_a in A:
                max_sim = 0.0
                for gram_b in B:
                    sim = self._ngram_similarity(gram_a, gram_b)
                    if sim > max_sim:
                        max_sim = sim
                soft_intersection += max_sim

            num += w * self._binary_cosine(len(A), len(B), soft_intersection)
            den += w

        if den <= 0.0:
            return 0.0
        return num / den
    
    def _get_weight_by_n(self, n: int, default: float = 1.0) -> float:
        for start, end, value in self.weights_by_n:
            if start <= n <= end:
                return value
        return default
        
    def _is_potential_keyword(self, q: str) -> bool:
        """Filtro rápido: suma de frecuencias normalizadas de n-gramas globales.
        Opcional: combinar con peso por n-grama (weights_by_n)."""
        try:
            grams: Set[str] = set()
            for n in range(self.gngr[0], self.gngr[1] + 1):
                grams.update(self._ngrams(q, n))
            if not grams:
                return False

            total = 0.0
            for g in grams:
                freq_norm = self.global_ngrams_freq_norm.get(g, 0.0)
                if freq_norm == 0.0:
                    continue
                # Usar la frecuencia normalizada global como peso (no usar weights_by_n aquí)
                w = self.global_ngrams_freq_norm.get(g, 0.0)
                total += freq_norm * w
            # normalizar por la suma de las frecuencias (pesos) consideradas
            denom = sum(self.global_ngrams_freq_norm.get(g, 0.0) for g in grams) or 1.0
            score = total / denom
            return score >= self.global_filter_threshold
        except Exception as e:
            logger.debug("Error en filtro global: %s", e, exc_info=True)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return{
        "total_words": len(self.global_words),
        "threshold_similarity": self.threshold,
        "char_ngram_range": self.ngr,
        "weights_by_n": self.weights_by_n
        }

