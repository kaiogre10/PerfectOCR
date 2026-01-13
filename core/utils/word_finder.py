import os
#ort datetime
import logging
import pickle
import re
import unicodedata
#om datetime import datetime
from typing import List, Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class WordFinder:
    def __init__(self, model_path: str):
        self.model: Dict[str, Any] = self._load_model(model_path)
        self.wf_path: str = "C:/word_finder_model/src/word_finder.py"
        self.params = self.model.get("params", {})
        self.global_words: List[str] = self.model["global_words"]
        self.variant_to_field = self.model.get("variant_to_field", {})
        self.noise_words = self.model["noise_words"]
        noise_filter = self.model.get("noise_filter", {})
        global_filter = self.model.get("global_filter", {})
        self.global_filter_threshold = float(self.params.get("global_filter_threshold"))
        self.noise_grams: List[Tuple[str, float]] = noise_filter["noise_grams"]
        self.threshold: float = self.params.get("threshold_similarity")
        self.ngrams: Tuple[int, int] = self.params["char_ngrams"]
        self.thresholds_by_len: List[Tuple[int, int, float]] = [tuple(item) for item in self.params["thresholds_by_len"]]
        self.weights_by_n: List[Tuple[int, int, float]] = [tuple(item) for item in self.params["weights_by_n"]]
        self.window_flex = self.params.get("window_flexibility")
        self.forb_match: float = self.params.get("forb_match")
        self.global_counter = global_filter.get("global_counter", None)
        self.top_ngrams_slice: set[str] = global_filter.get("global_ngrams", set())
        self.model_time = self.model.get("model_time")
        # timestamp_model = os.path.getmtime(self.wf_path)
        # fecha_wf = datetime.fromtimestamp(timestamp_model).isoformat()
        # logger.critical(f"FECHA DE GENERACIÓN DEL MODELO: {self.model_time}, FECHA DEL SCRIPT WORD_FINDER.PY: {fecha_wf}")

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
        global_range: Tuple[int, int] = self.ngrams
        try:
            final_threshold = threshold if threshold is not None else self.threshold

            single = False
            if isinstance(text, str):
                queue = [text]
                single = True
            else:
                queue = list(text)

            results: List[Dict[str, Any]] = []

            while queue:
                s = queue.pop(0)

                if not s:
                    continue

                q = self._clean_text(s)
                if not q:
                    continue

                    # PRE-FILTRO: Si hay match exacto, saltar el filtro de n-gramas
                if not self._is_exact_global_match(q):
                    if not self._is_potential_keyword(q, global_range):
                        continue

                q_cleaned, removed_noise = self._remove_noise_substrings(q)
                if removed_noise:
                    logger.debug(f"Ruido eliminado: '{removed_noise}' | Texto Limpio: '{q_cleaned}'")
                    q = q_cleaned
                    if not q:
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

                    # CORRECCIÓN: NO accedemos a self.global_ngrams[i] porque es un SET global
                    # En su lugar, generamos los n-gramas para el candidato actual 'cand'
                    # Tu función _build_query_grams es muy rápida.
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
                                    ngram_score: float = self._score_binary_cosine_multi_n(grams_cand, grams_sub,
                                                                                           global_range)
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
                                        "text": q,
                                        "start": j,
                                        "end": j + w
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

                    if final_matches:
                        best_match = final_matches[0]
                        results.append(best_match)

                        # Extraer la palabra encontrada y procesar el resto
                        start = best_match.get("start")
                        end = best_match.get("end")

                        if start is not None and end is not None:
                            left_part = q[:start].strip()
                            right_part = q[end:].strip()

                            if left_part:
                                queue.append(left_part)
                            if right_part:
                                queue.append(right_part)

                            logger.info(f"Extracted '{best_match['word_found']}' from '{q}'. Remaining: '{left_part}', '{right_part}'")

            if single:
                return results if results else []
            return results
        except Exception as e:
            logger.error(f"Error buscando palabras clave: '{e}'", exc_info=True)

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

            grams_text = self._build_query_grams(original_text, self.ngrams)
            grams_keyword = self._build_query_grams(keyword_found, self.ngrams)

            tiebreaker_score = self._score_binary_cosine_multi_n(grams_keyword, grams_text, self.ngrams)
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

    def _score_binary_cosine_multi_n(self, grams_a: Dict[int, set[str]], grams_b: Dict[int, set[str]],
                                     nrange: Tuple[int, int]) -> float:
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
        """
        Filtro rapido con Similitud Suave Directa.
        Compara los n-gramas del input contra el Slice de Top N-gramas
        usando similitud porcentual plana (sin pesos por longitud).
        """
        try:
            if not q or not self.top_ngrams_slice:
                return False

            # Generar todos los n-gramas del input
            input_ngrams: List[str] = []
            for n in range(nrange[0], nrange[1] + 1):
                input_ngrams.extend(self._ngrams(q, n))

            if not input_ngrams:
                return False

            # Calcular score usando similitud suave
            total_max_similarities = 0.0

            for input_gram in input_ngrams:
                max_sim = 0.0
                input_len = len(input_gram)

                # Comparar contra el slice (set de strings)
                for top_gram in self.top_ngrams_slice:
                    # Solo comparar n-gramas de la misma longitud
                    if len(top_gram) != input_len:
                        continue

                    # Reutilizar la función existente
                    sim = self._ngram_similarity(input_gram, top_gram)

                    if sim > max_sim:
                        max_sim = sim
                        if max_sim == 1.0:
                            break

                total_max_similarities += max_sim

            # Score final: promedio simple
            score = total_max_similarities / len(input_ngrams)

            if score < self.global_filter_threshold:
                logger.debug(f"Filtro rechazado para '{q}' Score={score:.4f}")
                return False

            logger.debug(f"Filtro pasado para '{q}' Score={score:.4f}")
            return True

        except Exception as e:
            logger.error("Error en filtro global: %s", e, exc_info=True)
            return False

    def _is_forbidden(self, candidate: str) -> bool:
        """Verifica si un candidato coincide con alguna palabra prohibida usando los mismos umbrales"""
        nrange: Tuple[int, int] = self.ngrams
        try:
            for i, noise_word in enumerate(self.noise_words):
                if not noise_word:
                    continue

                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)
                if min_w > len(candidate):
                    continue
                # Optimizacion: Si candidato es mucho mas grande que la palabra prohibida, no es match prohibido estricto
                # (aunque _remove_noise usaria logica de substring, aqui verificamos identidad de candidato)
                max_w = min(len(candidate), noise_len + self.window_flex)

                # CAMBIO: Ya viene como Dict[int, set[str]] desde el pickle optimizado
                grams_forbidden = self.noise_grams[i]

                for w in range(min_w, max_w + 1):
                    if w > len(candidate):
                        break
                    for j in range(0, len(candidate) - w + 1):
                        sub = candidate[j:j + w]

                        if sub == noise_word:
                            similarity = 1.0
                        else:
                            # Solo calculamos n-gramas del candidato al vuelo
                            grams_sub = self._build_query_grams(sub, nrange)
                            similarity = self._score_binary_cosine_multi_n(grams_forbidden, grams_sub, nrange)

                            len_ratio = max(len(sub), noise_len) / max(1, min(len(sub), noise_len))
                            if len_ratio >= 2.0:
                                penalty = min(len(sub), noise_len) / max(len(sub), noise_len)
                                similarity *= penalty

                        if similarity > self.forb_match:
                            logger.info(
                                f"RUIDO: '{candidate}', Similitud: {similarity:.4f}, n-gramas: {grams_forbidden}")
                            return True

            return False
        except Exception as e:
            logger.error(f"Error verificando palabra prohibida: {e}", exc_info=True)
            return False

    def _clean_text(self, s: str) -> str:
        try:
            if not s:
                return ""

            # Normalizar para separar tildes y convertir a minúsculas
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()

            # CAMBIO: eliminar todo lo que no sean letras a-z y espacios SIN INYECTAR ESPACIOS
            q = re.sub(r"[^a-z\s]+", "", q)

            # dejar solo un espacio entre palabras y quitar extremos
            return re.sub(r"\s+", " ", q).strip()
        except Exception as e:
            logger.error(msg=f"Error limpiando texto: {e}", exc_info=True)
        return ""

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "params": self.params
        }

    def _remove_noise_substrings(self, text: str) -> Tuple[str, List[str]]:
        """
        Elimina todos los substrings que coincidan con palabras prohibidas.
        Retorna: (texto_limpio, lista_de_ruidos_eliminados)
        """
        cleaned = text
        removed_noise: List[str] = []
        nrange: Tuple[int, int] = self.ngrams

        try:
            # PASO 1: Emparejar palabras con sus perfiles de n-gramas precalculados
            candidates: List[Tuple[str, Any]] = []
            for i, word in enumerate(self.noise_words):
                if word and i < len(self.noise_grams):
                    candidates.append((word, self.noise_grams[i]))

            # PASO 2: Ordenar por longitud descendente.
            candidates.sort(key=lambda x: len(x[0]), reverse=True)

            for noise_word, grams_forbidden_tuple in candidates:
                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)

                # grams_forbidden_tuple is Tuple[str, float], but we need Dict[int, set[str]]
                # If your pickle stores Dict[int, set[str]], assign accordingly:
                if isinstance(grams_forbidden_tuple, dict):
                    grams_forbidden = grams_forbidden_tuple
                elif isinstance(grams_forbidden_tuple, tuple) and isinstance(grams_forbidden_tuple[0], dict):
                    grams_forbidden = grams_forbidden_tuple[0]
                else:
                    grams_forbidden = {}

                # Buscar coincidencias (múltiples pases para eliminar repeticiones de la misma palabra)
                found_any = True
                while found_any:
                    found_any = False

                    # Calculamos el max_w dinámicamente sobre el texto que se va reduciendo
                    current_max_w = min(len(cleaned), noise_len + self.window_flex)

                    # Iterar de ventana GRANDE a ventana PEQUEÑA
                    for w in range(current_max_w, min_w - 1, -1):
                        if w > len(cleaned):
                            continue
                        for j in range(0, len(cleaned) - w + 1):
                            sub = cleaned[j:j + w]

                            # Comparación rápida primero
                            if sub == noise_word:
                                similarity = 1.0
                            else:
                                grams_sub = self._build_query_grams(sub, nrange)
                                # grams_forbidden ya es el diccionario óptimo
                                similarity = self._score_binary_cosine_multi_n(grams_forbidden, grams_sub, nrange)

                                len_ratio = max(len(sub), noise_len) / max(1, min(len(sub), noise_len))
                                if len_ratio >= 2.0:
                                    penalty = min(len(sub), noise_len) / max(len(sub), noise_len)
                                    similarity *= penalty

                            if similarity > self.forb_match:
                                # Eliminar el substring agregando un espacio de seguridad para no fusionar palabras
                                cleaned = (cleaned[:j] + " " + cleaned[j + w:]).strip()
                                # Limpiar dobles espacios generados
                                cleaned = re.sub(r"\s+", " ", cleaned).strip()

                                removed_noise.append(sub)
                                logger.debug(
                                    f"SUBSTRING ELIMINADO: '{sub}' | Similitud: {similarity:.4f} | RUIDO ORIG: '{noise_word}'")
                                found_any = True
                                break  # Romper bucle interno j para reiniciar escaneo
                        if found_any:
                            break  # Romper bucle de ventana para reiniciar

            return cleaned, removed_noise

        except Exception as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
            return text, []

    def _is_exact_global_match(self, q: str) -> bool:
        """
        Pre-filtro rápido: verifica si el texto limpio coincide exactamente
        con alguna palabra del vocabulario global.
        Evita rechazos por el filtro de n-gramas cuando hay match directo.
        """
        if q in set(self.global_words):
            logger.info(f"Match exacto para '{q}'")
            return True
        else:
            return False
