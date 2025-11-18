import os
import datetime
import logging
import pickle
from datetime import datetime
from typing import List, Any, Dict, Optional, Tuple
from cleantext import clean  # type: ignore

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str):
        self.model: Dict[str, Any] = self._load_model(model_path)
        self.wf_path: str = "C:/word_finder_model/src/word_finder.py"
        self.params = self.model.get("params", {})
        self.global_words: List[str] = self.model["global_words"]
        self.variant_to_field = self.model.get("variant_to_field", {})
        self.noise_words = self.model["noise_words"]
        self.noise_filter = self.model.get("noise_filter", {})
        self.global_filter = self.model.get("global_filter", {})
        self.global_filter_threshold = float(self.params.get("global_filter_threshold"))
        self.global_ngrams: List[Tuple[str, float]] = self.global_filter["global_ngrams"]
        self.noise_grams: List[Tuple[str, float]] = self.noise_filter["noise_grams"]
        self.threshold: float = self.params.get("threshold_similarity")
        self.gngr: Tuple[int, int] = tuple(self.params["char_ngram_global"])
        self.ngr: Tuple[int, int] = tuple(self.params["char_ngram_noise"])
        self.thresholds_by_len: List[Tuple[int, int, float]] = [tuple(item) for item in self.params["thresholds_by_len"]]
        self.weights_by_n: List[Tuple[int, int, float]] = [tuple(item) for item in self.params["weights_by_n"]]
        self.window_flex = self.params.get("window_flexibility")
        self.forb_match: float = self.params.get("forb_match")
        self.max_results: int = self.params.get("max_results_per_query")
        self.field_conversion_map = self.params.get("field_conversion_map", {})
        self.global_counter = self.global_filter.get("global_counter", None)
        self.global_vocab = self.global_filter.get("global_vocab", None)
        self.model_time = self.model.get("model_time")
        timestamp_model = os.path.getmtime(self.wf_path)
        fecha_wf = datetime.fromtimestamp(timestamp_model).isoformat()
        logger.critical(f"FECHA DE GENERACIÓN DEL MODELO: {self.model_time}, FECHA DEL SCRIPT WORD_FINDER.PY: {fecha_wf}")

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if not isinstance(self.model, dict):  # type: ignore
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            return self.model
        except Exception as e:
            logger.error(f"Error al cargar el modelo {e}", exc_info=True)
            raise

    def find_keywords(self, text: List[str] | str, threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
        global_range: Tuple[int, int] = self.gngr
        try:
            final_threshold = threshold if threshold is not None else self.threshold

            single = False
            if isinstance(text, str):
                text = [text]
                single = True

            results: List[Dict[str, Any]] = []
            for s in text:

                if s is None:
                    return None

                q = self._clean_text(s)
                if not q:
                    continue

                if not self._is_potential_keyword(q, global_range):
                    continue

                if self._is_forbidden(q):
                    continue

                # Lista para guardar todos los matches de este string 's'
                found_matches_for_s: List[Dict[str, Any]] = []

                for i in range(len(self.global_words)):
                    cand = self.global_words[i]
                    cand_len = len(cand)
                    min_w = max(1, cand_len - self.window_flex)
                    if min_w > len(q):
                        continue
                    max_w = min(len(q), cand_len + self.window_flex)
                    grams_cand = self.global_ngrams[i]

                    if isinstance(grams_cand, dict):
                        pass

                    else:
                        try:
                            # list/tuple of ngram strings -> group by length
                            if isinstance(grams_cand, (list, tuple, set)) and all(
                                    isinstance(x, str) for x in grams_cand):
                                normalized: Dict[int, set[str]] = {}
                                for g in grams_cand:
                                    normalized.setdefault(len(g), set()).add(g)
                                grams_cand = normalized
                            else:
                                # Unknown format (e.g. (word, freq) or other) -> build on the fly
                                grams_cand = self._build_query_grams(cand, global_range)
                        except Exception:
                            grams_cand = self._build_query_grams(cand, global_range)

                    try:
                        for w in range(min_w, max_w + 1):
                            if w > len(q):
                                break
                            for j in range(0, len(q) - w + 1):
                                sub = q[j:j + w]
                                grams_sub = self._build_query_grams(sub, global_range)
                                # Solo asignar similitud perfecta si es la misma palabra completa (misma longitud)
                                if sub == cand and len(sub) == cand_len:
                                    ngram_score: float = 1.0
                                else:
                                    ngram_score: float = self._score_binary_cosine_multi_n(grams_cand, grams_sub, global_range)
                                    len_ratio = max(len(sub), cand_len) / max(1, min(len(sub), cand_len))
                                    if len_ratio >= 2.0:
                                        penalty = min(len(sub), cand_len) / max(len(sub), cand_len)
                                        ngram_score *= penalty
                                if ngram_score > final_threshold:
                                    if self._is_forbidden(cand):
                                        continue

                                    key_field = self.variant_to_field.get(cand)
                                    # Añadir el match a la lista temporal en lugar de retornar
                                    found_matches_for_s.append({
                                        "key_field": key_field,
                                        "word_found": cand,
                                        "similarity": float(ngram_score),
                                        "text": q
                                    })

                    except Exception as e:
                        logger.error(f"Error en el bucle de búsqueda de find_keywords: {e}", exc_info=True)
                
                # Después de comprobar todos los candidatos, agrupar y seleccionar el mejor por campo
                if found_matches_for_s:
                    best_match_by_field: Dict[str, Dict[str, Any]] = {}
                    
                    for match in found_matches_for_s:
                        field = match["key_field"]
                        
                        # Si es el primer match para este campo, o si es mejor que el guardado
                        if field not in best_match_by_field or match["similarity"] > best_match_by_field[field]["similarity"]:
                            best_match_by_field[field] = match
                    
                    # Desempatar usando la similitud de palabra completa
                    final_matches = self._resolve_ambiguity_by_full_word(list(best_match_by_field.values()))
                    results.extend(final_matches)


            if single:
                return results if results else []
            return results

        except Exception as e:
            logger.error(f"Error principal en find_keywords: {e}", exc_info=True)
            return None

    def _resolve_ambiguity_by_full_word(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dada una lista con el mejor match por campo, resuelve empates o ambigüedades
        calculando la similitud de n-gramas entre el texto original y la palabra clave completa.
        Agrega logs útiles sobre el proceso de desempate y puntajes.
        """

        if not matches:
            return []
        
        if len(matches) == 1:
            return matches

        # Calcular un "score de desempate" para cada match
        for i, match in enumerate(matches):
            original_text = match['text']
            keyword_found = self._clean_text(match['word_found'])

            grams_text = self._build_query_grams(original_text, self.gngr)
            grams_keyword = self._build_query_grams(keyword_found, self.gngr)

            tiebreaker_score = self._score_binary_cosine_multi_n(grams_keyword, grams_text, self.gngr)
            match['score_final'] = tiebreaker_score

            logger.debug(
                "EMPATE: Match #%d: campo: %s, palabra: '%s' | score de desempate: %.4f | texto: '%s'",
                i,
                match.get("key_field"),
                match.get("word_found"),
                tiebreaker_score,
                original_text
            )

        # Ordenar por el score de desempate
        matches.sort(key=lambda x: x['score_final'], reverse=True)

        logger.debug(
            "DESEMPATE: texto '%s': campo: %s, palabra: '%s', score_final: %.4f \n"
            "====================================================================================================================================================================================================",
            matches[0].get("text"),
            matches[0].get("key_field"),
            matches[0].get("word_found"),
            matches[0].get("score_final"),
        )
        return [matches[0]]

    def _build_query_grams(self, q: str, nrange: Tuple[int, int]) -> Dict[int, set[str]]:
        """Construye n-gramas de la consulta"""
        gq: Dict[int, set[str]] = {}
        for n in range(nrange[0], nrange[1] + 1):
            gq[n] = set(self._ngrams(q, n))
        return gq

    def _ngrams(self, q: str, n: int) -> List[str]:
        try:
            if n <= 0 or not q:
                return []
            if len(q) < n:
                return []
            return [q[i:i + n] for i in range(len(q) - n + 1)]
        except Exception as e:
            logger.error(f"Error construyendo n-gramas: {e}", exc_info=True)
            return []

    def _ngram_similarity(self, a: str, b: str) -> float:
        """Calcula la similitud entre dos n-gramas como la proporción de caracteres iguales."""
        try:
            matches = float(sum(1 for x, y in zip(a, b) if x == y))
            # Normaliza por la longitud máxima para manejar n-gramas de longitudes distintas si fuera el caso.
            return matches / float(max(len(a), len(b)))
        except Exception as e:
            logger.warning(f"Error comprobando texto: {e}", exc_info=True)
            return 0.0

    def _binary_cosine(self, size_a: int, size_b: int, soft_intersection: float) -> float:
        """Calcula coseno binario entre dos conjuntos"""
        if size_a == 0 or size_b == 0:
            return 0.0
        return soft_intersection / float((size_a * size_b) ** 0.5)

    def _score_binary_cosine_multi_n(self, grams_a: Dict[int, set[str]], grams_b: Dict[int, set[str]], nrange: Tuple[int, int]) -> float:
        """Calcula score ponderado por coseno binario multi-n-grama"""
        num = 0.0
        den = 0.0
        for n in range(nrange[0], nrange[1] + 1):
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

    def _is_potential_keyword(self, q: str, nrange: Tuple[int, int]) -> bool:
        """Filtro rápido: suma de frecuencias normalizadas de n-gramas globales."""
        try:
            grams: List[str] = []
            for n in range(nrange[0], nrange[1] + 1):
                grams.extend(self._ngrams(q, n))
            if not grams or self.global_counter is None or self.global_vocab is None:
                return False

            # Transforma el texto en matriz de n-gramas usando el vocabulario global
            X = self.global_counter.transform([" ".join(grams)])
            # Suma de frecuencias normalizadas
            total = X.sum()
            score = total / max(1, len(grams))

            if score < self.global_filter_threshold:
                logger.debug(f"Filtro rechazado para '{q}' Score={score:.4f}, ngrams={grams}")
                return False
            else:
                logger.debug(f"Filtro pasado para '{q}' Score={score:.4f}, ngrams={grams}")
                return score > self.global_filter_threshold

        except Exception as e:
            logger.error("Error en filtro global: %s", e, exc_info=True)
            return False

    def _is_forbidden(self, candidate: str) -> bool:
        """Verifica si un candidato coincide con alguna palabra prohibida usando los mismos umbrales"""
        nrange: Tuple[int, int] = self.ngr
        try:
            for i, noise_word in enumerate(self.noise_words):
                if not noise_word:
                    continue

                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)
                if min_w > len(candidate):
                    continue
                max_w = min(len(candidate), noise_len + self.window_flex)

                grams_forbidden = self.noise_grams[i]
                if isinstance(grams_forbidden, dict):
                    pass  # ya está en formato correcto
                elif isinstance(grams_forbidden, (list, tuple, set)) and all(isinstance(x, str) for x in grams_forbidden): # type: ignore
                    normalized: Dict[int, set[str]] = {}
                    for g in grams_forbidden:
                        normalized.setdefault(len(g), set()).add(g)
                    grams_forbidden = normalized
                else:
                    grams_forbidden = self._build_query_grams(noise_word, nrange)

                for w in range(min_w, max_w + 1):
                    if w > len(candidate):
                        break
                    for j in range(0, len(candidate) - w + 1):
                        sub = candidate[j:j + w]
                        grams_sub = self._build_query_grams(sub, nrange)

                        if sub == noise_word and len(sub) == noise_len:
                            similarity = 1.0
                        else:
                            similarity = self._score_binary_cosine_multi_n(grams_forbidden, grams_sub, nrange)
                            len_ratio = max(len(sub), noise_len) / max(1, min(len(sub), noise_len))
                            if len_ratio >= 2.0:
                                penalty = min(len(sub), noise_len) / max(len(sub), noise_len)
                                similarity *= penalty

                        if similarity > self.forb_match:
                            logger.info(f"RUIDO: '{candidate}', Similitud: {similarity:.4f}, n-gramas: {grams_sub}")
                            return True

            return False
        except Exception as e:
            logger.error(f"Error verificando palabra prohibida: {e}", exc_info=True)
            return False

    def _clean_text(self, s: str) -> Optional[str]:
        try:
            if s is None:
                return None

            q = clean(
                s,
                clean_all=False,
                extra_spaces=True,
                stemming=False,
                stopwords=False,
                lowercase=True,
                numbers=True,
                punct=True,
            )

            if q is None:
                return None

            return q

        except Exception as e:
            logger.error(msg=f"Error limpiando texto: {e}", exc_info=True)
        return None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "noise_words": self.noise_words,
            "field_conversion_map": self.field_conversion_map
        }
