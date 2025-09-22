import os
import re
import logging
import pickle
import unicodedata
from typing import List, Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str, project_root: str):
        self.project_root = project_root
        self.model_path = model_path
        self._active = "standard"
        self.model: Dict[str, Any] = self._load_model(self.model_path)
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
        self.key_words: List[str] = [self._normalize(str(w)) for w in self.model.get("key_words", []) or []]
        vt_field: Dict[str, str] = self.model.get("variant_to_field", {}) or {}
        self.variant_to_field: Dict[str, str] = {self._normalize(str(k)): v for k, v in vt_field.items()}
        self.params: Dict[str, Any] = self.model.get("params", {}) or {}
        self.ngr = self.params.get("char_ngram_range", [2, 5]) or [2, 5]
        self.density_encoder: Dict[str, float] = self.model.get("density_encoder", {})
        self.field_stats: Dict[str, Dict[str, List[float]]] = self.model.get("field_stats", {})
        self.normalized_stats: Dict[str, Dict[str, float]] = self.model.get("normalized_stats", {})
        self.MAX_CHAR_COUNT = 25.0
        self.MAX_MEAN = 113.0

        try:
            self.n_min, self.n_max = int(self.ngr[0]), int(self.ngr[1])
        except Exception:
            self.n_min, self.n_max = 2, 5
        if self.n_min < 1 or self.n_max < self.n_min:
            self.n_min, self.n_max = 2, 5

        self.threshold = float(self.params.get("threshold_similarity", 0.70))
        self.thresholds_by_len: Dict[Any, Any] = self.params.get("thresholds_by_len", {}) or {}

        # pesos por n-grama (inversos)
        self.wb = self.params.get("weights_by_n", {})
        self.weights_by_n: Dict[int, float] = {}
        if self.wb and isinstance(self.wb, dict):
            for n in range(self.n_min, self.n_max + 1):
                self.weights_by_n[n] = float(self.wb.get(str(n), 1.0))
        else:
            # pesos por defecto inversos
            for n in range(self.n_min, self.n_max + 1):
                if n == 2: self.weights_by_n[n] = 0.5
                elif n == 3: self.weights_by_n[n] = 0.7
                elif n == 4: self.weights_by_n[n] = 0.85
                elif n == 5: self.weights_by_n[n] = 1.0
                else: self.weights_by_n[n] = 1.0

        # índice precomputado desde el pickle
        self.grams_index = self.model.get("grams_index")
        self.grams: List[Dict[int, set[str]]] = []
        self.lengths: List[int] = []

        if self.grams_index:
            for entry in self.grams_index:
                length = int(entry.get("len", 0))
                self.gmap_raw: Dict[int, List[str]] = entry.get("grams", {}) or {}
                gmap_sets= {}
                for n in range(self.n_min, self.n_max + 1):
                    gmap_sets[n] = set(self.gmap_raw.get(n, []) or [])
                self.lengths.append(length)
                self.grams.append(gmap_sets)
        else:
            self.grams = []
            self.lengths = []
            for w in self.key_words:
                self.length = len(w)
                self.lengths.append(self.length)
                gmap_sets: Dict[int, set[str]] = {}
                for n in range(self.n_min, self.n_max + 1):
                    gmap_sets[n] = set(self._ngrams(w, n))
                self.grams.append(gmap_sets)

        # buckets por longitud
        self.buckets: Dict[int, List[int]] = self.model.get("buckets_by_len") or {}
        self.buckets_by_len: Dict[int, List[int]] = {int(k): list(v) for k, v in self.buckets.items()}
        if not self.buckets_by_len:
            for i, length in enumerate(self.lengths):
                self.buckets_by_len.setdefault(length, []).append(i)

    def _len_threshold(self, length: int) -> float:
        """Obtiene umbral por longitud del candidato"""
        if self.thresholds_by_len:
            self.int_keys = sorted([int(k) for k in self.thresholds_by_len.keys() if str(k).isdigit()])
            for k in self.int_keys:
                if length <= k:
                    try:
                        return float(self.thresholds_by_len.get(str(k), self.thresholds_by_len.get(k, self.threshold)))
                    except Exception:
                        continue
            if "default" in self.thresholds_by_len:
                try:
                    return float(self.thresholds_by_len["default"])
                except Exception:
                    pass
        # umbrales por defecto
        if length <= 3:
            return max(self.threshold, 0.70)
        if length <= 6:
            return max(self.threshold, 0.60)
        if length <= 10:
            return max(self.threshold, 0.50)
        return max(self.threshold, 0.60)

    def find_keywords(self, text: str | List[str]) -> Optional[List[Dict[str, Any]]]:
        """Busca palabras clave usando coseno binario de n-gramas con ventana deslizante."""
        try:
            single = False
            if isinstance(text, str):
                text = [text]
                single = True

            results: List[Dict[str, Any]] = []
            window_flex = int(self.params.get("window_flexibility", 2)) 

            for s in text:
                q = self._normalize(s)
                if not q:
                    continue
                
                if not self._is_potential_keyword(q):
                    continue

                best_idx = None
                best_score = 0.0

                # Evaluamos subcadenas (ventana deslizante) contra cada candidato
                for i in range(len(self.key_words)):
                    cand = self.key_words[i]
                    cand_len = len(cand)

                    # Tamaños de ventana alrededor de la longitud del candidato
                    min_w = max(1, cand_len - window_flex)
                    if min_w > len(q):
                        continue
                    max_w = min(len(q), cand_len + window_flex)

                    grams_cand = self.grams[i]  # precomputado al cargar el modelo

                    # Recorremos subcadenas del query normalizado
                    for w in range(min_w, max_w + 1):
                        # Si la ventana supera el largo de q, no hay subcadenas
                        if w > len(q):
                            break
                        for j in range(0, len(q) - w + 1):
                            sub = q[j:j + w]

                            # n-gramas de la subcadena
                            grams_sub = self._build_query_grams(sub)
                            ngram_score = self._score_binary_cosine_multi_n(grams_cand, grams_sub)

                            # Score de estadísticas (si disponibles) usando la subcadena
                            stats_score = 0.0
                            field = self.variant_to_field.get(cand)
                            if field and cand in self.normalized_stats:
                                query_stats = self._calculate_query_stats(sub, field)
                                candidate_stats = self.normalized_stats[cand]
                                stats_score = self._stats_similarity(query_stats, candidate_stats)

                            combined_score = 0.5 * ngram_score + 0.5 * stats_score

                            if combined_score > best_score:
                                best_score = combined_score
                                best_idx = i

                if best_idx is not None:
                    cand = self.key_words[best_idx]
                    thr: float = self._len_threshold(len(cand))

                    if best_score >= thr:
                        key_field = self.variant_to_field.get(cand)

                        results.append({
                            "key_field": key_field,
                            "word_found": cand,
                            "similarity": float(best_score),
                            "query": q
                        })

            return results if not single else (results[0:1] if results else [])

        except Exception as e:
            logger.error(f"Error en find_keywords: {e}", exc_info=True)

    def _normalize(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^a-zA-Z0-9]", "", s)  # quita espacios, signos y también _
        return s

    def _build_query_grams(self, q: str) -> Dict[int, set[str]]:
        """Construye n-gramas de la consulta"""
        gq: Dict[int, set[str]] = {}
        for n in range(self.n_min, self.n_max + 1):
            gq[n] = set(self._ngrams(q, n))
        return gq

    def _ngrams(self, s: str, n: int) -> List[str]:
        if n <= 0 or not s:
            return []
        if len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s) - n + 1)]

    def _ngram_similarity(self, a: str, b: str) -> float:
        """Calcula la similitud entre dos n-gramas como la proporción de caracteres iguales."""
        if not a or not b:
            return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        # Normaliza por la longitud máxima para manejar n-gramas de longitudes distintas si fuera el caso.
        return matches / max(len(a), len(b))

    def _binary_cosine(self, size_a: int, size_b: int, inter_size: float) -> float:
        """Calcula coseno binario entre dos conjuntos"""
        if size_a == 0 or size_b == 0:
            return 0.0
        return inter_size / ((size_a * size_b) ** 0.5)

    def _score_binary_cosine_multi_n(self, grams_a: Dict[int, set[str]], grams_b: Dict[int, set[str]]) -> float:
        """Calcula score ponderado por coseno binario multi-n-grama"""
        num = 0.0
        den = 0.0

        for n in range(self.n_min, self.n_max + 1):
            A = grams_a.get(n, set())
            B = grams_b.get(n, set())
            w = float(self.weights_by_n.get(n, 1.0))

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
                    if max_sim == 1.0:  # Optimización: si ya encontró un match perfecto, no necesita seguir.
                        break
                soft_intersection += max_sim

            # Usamos la misma fórmula de coseno, pero con la "intersección suave".
            num += w * self._binary_cosine(len(A), len(B), soft_intersection)
            den += w

        if den <= 0.0:
            return 0.0
        return num / den

    def _calculate_query_stats(self, query: str, field: str) -> Dict[str, float]:
        """Calcula estadísticas normalizadas para la query"""
        chars = list(query)
        values = [self.density_encoder.get(c, 0) for c in chars]
        char_count = float(len(query))
        mean_val = sum(values) / char_count if char_count else 0.0
        variance = sum((v - mean_val) ** 2 for v in values) / char_count if char_count else 0.0
        std_dev = variance ** 0.5

        # Normalizar igual que en el modelo
        norm_stats: Dict[str, float] = {}
        norm_stats["char_count_n"] = min(char_count / self.MAX_CHAR_COUNT, 1.0)
        norm_stats["mean_n"] = mean_val / self.MAX_MEAN

        # Normalización por campo para variance y std_dev
        if field in self.field_stats:
            var_min, var_max = min(self.field_stats[field]["variance"]), max(self.field_stats[field]["variance"])
            std_min, std_max = min(self.field_stats[field]["std_dev"]), max(self.field_stats[field]["std_dev"])
            norm_stats["variance_n"] = (variance - var_min) / (var_max - var_min) if var_max != var_min else 1.0
            norm_stats["std_dev_n"] = (std_dev - std_min) / (std_max - std_min) if std_max != std_min else 1.0
        else:
            norm_stats["variance_n"] = 0.0
            norm_stats["std_dev_n"] = 0.0

        return norm_stats

    def _stats_similarity(self, stats_a: Dict[str, float], stats_b: Dict[str, float]) -> float:
        """Calcula similitud coseno entre vectores de estadísticas"""
        vector_a = [stats_a.get("char_count_n", 0), stats_a.get("mean_n", 0),
                    stats_a.get("variance_n", 0), stats_a.get("std_dev_n", 0)]
        vector_b = [stats_b.get("char_count_n", 0), stats_b.get("mean_n", 0),
                    stats_b.get("variance_n", 0), stats_b.get("std_dev_n", 0)]

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = sum(a * a for a in vector_a) ** 0.5
        norm_b = sum(b * b for b in vector_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
        
    def _is_potential_keyword(self, query: str) -> bool:
        """Filtro súper rápido: intersección de n-gramas"""
        global_filter = self.model.get("global_filter", {})
        if not global_filter:
            return True
        
        global_ngrams = global_filter.get("global_ngrams", set())
        if not global_ngrams:
            return True
        
        # Generar n-gramas del query
        query_ngrams = set()
        for n in range(self.n_min, self.n_max + 1):
            query_ngrams.update(self._ngrams(query, n))
        
        # Si hay intersección, es potencial keyword
        return bool(query_ngrams & global_ngrams)

    def get_model_info(self) -> Dict[str, Any]:
        return{
        "total_words": len(self.key_words),
        "threshold_similarity": self.threshold,
        "key_words": len(self.key_words),
        "campos_disponibles": list(self.model.get("key_words", {}).keys()),
        "density_encoder_loaded": len(self.density_encoder) > 0,
        "fields_with_stats": len(self.field_stats),
        "words_with_normalized_stats": len(self.normalized_stats),
        "statistics_enabled": len(self.normalized_stats) > 0,
        "char_ngram_range": self.ngr,
        "weights_by_n": self.weights_by_n
    }

    def _regex_patterns_rfc(self, query: str) -> float:
        """Patrones regex específicos para RFC con formato real"""
        # RFC real: 4 letras + 6 números + 3 alfanuméricos (13 chars total)
        patterns = [
            # RFC completo con ruido leve
            r"[a-z]{3,5}\d{5,7}[a-z0-9]{2,4}",
            # RFC parcial pero reconocible
            r"[a-z]{3,4}.*\d{4,6}.*[a-z0-9]{1,3}",
            # Palabra "rfc" con ruido
            r"r.{0,2}f.{0,2}c",
        ]

        # Si es muy largo, puede ser RFC completo
        if len(query) >= 10:
            for pattern in patterns[:2]:  # patrones de RFC completo
                if re.search(pattern, query, re.IGNORECASE):
                    # Verificar estructura más estricta
                    if re.match(r"[a-z]{4}\d{6}[a-z0-9]{3}", query):
                        return 0.95  # RFC perfecto
                    elif len(query) <= 15:  # RFC con ruido leve
                        return 0.85
                    else:
                        return 0.75

        # Palabra "rfc" corta
        elif 2 <= len(query) <= 5:
            if re.search(patterns[2], query, re.IGNORECASE):
                if re.fullmatch(r"r.{0,1}f.{0,1}c", query):
                    return 0.80
                else:
                    return 0.70

        return 0.0

    def _regex_patterns_fecha(self, query: str) -> tuple[float, str]:
        """Patrones regex específicos para fechas usando datetime"""

        # Intentar parsear como fecha real
        date_patterns = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",  # 31/12/2023
            "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",  # 31/12/23
            "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d",  # 2023/12/31
            "%d%m%Y", "%d%m%y",                  # 31122023, 311223
        ]

        # Limpiar query para parseo de fechas
        clean_query = re.sub(r"[^\d/\-.]", "", query)

        for pattern in date_patterns:
            try:
                datetime.strptime(clean_query, pattern)
                return 0.95, "fecha"  # Es una fecha válida
            except ValueError:
                continue

        # Patrones de palabras "fecha" y "hora" con ruido
        word_patterns = [
            (r"f.{0,2}e.{0,2}c.{0,2}h.{0,2}a", "fecha"),
            (r"h.{0,2}o.{0,2}r.{0,2}a", "hora"),
            (r"[fh].{0,3}[eo].{0,3}[cr].{0,3}[ha]", None),  # ambiguo
        ]

        for pattern, word_type in word_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                if word_type:
                    if re.fullmatch(pattern.replace(".{0,2}", ".{0,1}"), query):
                        return 0.85, word_type  # match casi perfecto
                    else:
                        return 0.75, word_type  # match con ruido
                else:
                    # Decidir entre fecha/hora por contexto
                    if re.search(r"h.*o.*r", query):
                        return 0.70, "hora"
                    else:
                        return 0.70, "fecha"

        return 0.0, ""

    def _search_by_regex(self, query: str) -> Dict[str, Any]:
        """Búsqueda por regex para RFC y fechas específicamente"""
        if len(query) < 2:
            return {}

        # Probar RFC (tanto palabra como formato completo)
        rfc_score = self._regex_patterns_rfc(query)
        if rfc_score > 0:
            return {
                "key_field": "RFCProveedor",
                "word_found": "rfc" if len(query) <= 5 else "registrofederaldecontribuyentes",
                "similarity": float(rfc_score),
                "query": query,
            }

        # Probar fecha/hora
        fecha_score, fecha_type = self._regex_patterns_fecha(query)
        if fecha_score > 0:
            return {
                "key_field": "FechaDocumento",
                "word_found": fecha_type,
                "similarity": float(fecha_score),
                "query": query,
            }

        return {}

    def debug_find_keywords(self, text: str | List[str]) -> Dict[str, Any]:
        """Versión debug que muestra paso a paso lo que hace find_keywords"""
        print(f"\n=== DEBUG find_keywords ===")
        print(f"Input: {text}")

        results = self.find_keywords(text)

        print(f"Output length: {len(results)}")
        print(f"Output value: {results}")

        if results:
            for i, item in enumerate(results):
                print(f"\nResult item {i}:")
                print(f"Content: {item}")

        print(f"=== END DEBUG ===\n")
        return {"debug_info": "complete", "original_result": results}