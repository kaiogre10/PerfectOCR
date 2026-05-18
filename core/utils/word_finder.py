import os
import logging
import numpy as np
import pickle
import re
import unicodedata
import time
from typing import List, Any, Dict, Tuple, Set, FrozenSet

logger = logging.getLogger(__name__)

_space_pattern = re.compile(r"\s+")
_space_clean_pattern = re.compile(r"[^a-z\s]+")
_nom_pattern= re.compile(r'(?<=[a-zA-Z])[^\w\s]+(?=[a-zA-Z])')

class WordFinder:
    def __init__(self, model_path: str, set_params: bool):
        model: Dict[str, Any] = self._load_model(model_path)
        if set_params:
            """Aquí va una función que obtendría los parametros de configuración del master_config
            pero me da flojera escribirla así que solo dejaré un log y no cambiaré el parametro "set_params"""
            logger.debug(f"Parametros establecidos y cargados manualmente")

        params: Dict[str, Any] = model.get("params", {})
        noise_filter = model.get("noise_filter", {})
        global_filter = model.get("global_filter", {})
        
        self.all_ngrams: Dict[str, Tuple[int, Dict[int, List[str]]]] = model.get("all_ngrams", {})
        self.global_words: FrozenSet[str] = frozenset(model["global_words"])
        self.noise_words: FrozenSet[str] = frozenset(model["noise_words"])
        
        self.global_filter_threshold: float = params.get("global_filter_threshold", {})
        self.noise_grams: List[Dict[int, List[str]]] = noise_filter["noise_grams"]
        
        self.threshold: float = params.get("threshold_similarity", {})
        self.ngrams: Tuple[int, int] = params["char_ngrams"]
        self.window_flex: int = params.get("window_flexibility", {})
        self.forb_match: float = params.get("forb_match", {})
        self.min_diff: float = params.get("min_diff", {})
        
        self.noise_array: List[np.ndarray[Any, np.dtype[np.uint8]]] = noise_filter["noise_array"]
        self.global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = global_filter.get("global_matrices", {})

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model: Dict[str, Any] = pickle.load(f)
            if not isinstance(self.model, dict):  # type: ignore
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            logger.debug(f"Modelo cargado en: '{time.perf_counter() - t0:.6f}s'")
            return self.model
        except ExceptionGroup as e:
            logger.error(f"Error al cargar el modelo {e}", exc_info=True)
            raise

    def find_keywords(self, text: List[str] | str) -> List[Dict[str, Any]]:
        try:
            if not text:
                return []
            
            if text in self.noise_words:
                logger.info(f"Ruido inmediato: {text}")
                return []

            single = False
            if isinstance(text, str):
                queue = [text]
                single = True
            else:
                queue = list(text)

            results: List[Dict[str, Any]] = []
            assigned_fields: Set[int] = set()

            while queue:
                s = queue.pop(0)
                if not s:
                    continue
                
                if s in self.noise_words:
                    # logger.info(f"Ruido temprano: '{list(self.noise_words).pop(list(self.noise_words).index(s))}'")
                    continue
                
                q = self._normalize(s)
                # logger.debug(f"TEXTO NORMALIZADO: '{s}' -> '{q}'")
                if not q:
                    continue

                if q in self.noise_words:
                    # logger.info(f"Ruido temprano 2: '{list(self.noise_words).pop(list(self.noise_words).index(q))}'")
                    continue

                if not self._is_potential_keyword(q):
                    # logger.debug(f"Texto no paso filtro global: {q}")
                    continue

                # ELIMINACIÓN DE RUIDO: No usa assigned_fields
                q_cleaned, removed_noise = self._remove_noise_substrings(q)
                if removed_noise:
                    q = q_cleaned

                if q in self.noise_words:
                    # logger.info(f"Ruido temprano 3: '{list(self.noise_words).pop(list(self.noise_words).index(q))}'")
                    continue

                found_matches_for_s: List[Dict[str, Any]] = []

                # OPTIMIZACIÓN: Construir índice invertido de n-gramas del texto 'q' una sola vez
                q_grams_idx: Dict[str, List[int]] = {}
                for n in range(self.ngrams[0], self.ngrams[1] + 1):
                    for idx, gram in enumerate(self._ngrams(q, n)):
                        q_grams_idx.setdefault(gram, []).append(idx)

                # BÚSQUEDA DE KEYWORDS: Aquí SÍ se usa assigned_fields
                for cand, (key_field, grams_cand) in self.all_ngrams.items():
                    if key_field != 6 and key_field in assigned_fields:
                        continue
                    # if key_field == 5:
                    #     continue

                    cand_len = len(cand)
                    # Encontrar posiciones donde coinciden n-gramas del candidato
                    hit_positions: List[int] = []
                    for n, grams in grams_cand.items():
                        for g in grams:
                            if g in q_grams_idx:
                                hit_positions.extend(q_grams_idx[g])

                    if not hit_positions:
                        continue

                    best_score_for_cand: float = 0.0
                    best_sub_details: Dict[str, int] = {}

                    # Agrupamos posiciones cercanas para no probar la misma zona mil veces
                    sorted_unique_hits = sorted(list(set(hit_positions)))

                    # Definimos el rango de tamaños de ventana a probar
                    min_w = max(1, cand_len - self.window_flex)
                    max_w = cand_len + self.window_flex

                    # Iteramos sobre los puntos de inicio de los n-gramas coincidentes
                    for hit_start_pos in sorted_unique_hits:
                        # Probamos ventanas de diferentes tamaños centradas cerca del 'hit'
                        for w in range(min_w, max_w + 1):
                            # El inicio de la ventana debe permitir que el 'hit' esté dentro
                            # Probamos algunos desplazamientos para la ventana
                            for offset in range(-self.window_flex, 1):
                                start = hit_start_pos + offset
                                end = start + w

                                if start < 0 or end > len(q):
                                    continue

                                sub = q[start:end]
                                if not sub:
                                    continue
                                
                                elif sub == cand:
                                    penalty = self._length_penalty(w, cand_len)
                                    final_score = 1.0 * penalty
                                else:
                                    grams_sub = self._build_query_grams(sub)
                                    final_score = self._score_hybrid_greedy(grams_cand, grams_sub)
                                final_score *= self._length_penalty(w, cand_len)

                                if final_score > best_score_for_cand:
                                    best_score_for_cand = final_score
                                    best_sub_details = {
                                        "start": start,
                                        "end": end
                                    }

                    if best_score_for_cand > self.threshold:
                        found_matches_for_s.append({
                            "key_field": key_field,
                            "key_word": cand,
                            "similarity": best_score_for_cand,
                            "text": s,
                            "norm_ocr_text": q,
                            "start": best_sub_details["start"],
                            "end": best_sub_details["end"]
                        })
                # Después de comprobar todos los candidatos, agrupar y seleccionar el mejor por campo
                if found_matches_for_s:
                    best_match_by_field: Dict[int, Dict[str, Any]] = {}

                    for match in found_matches_for_s:
                        field = match["key_field"]

                        if field not in best_match_by_field:
                            best_match_by_field[field] = match
                        else:
                            if field == 6:
                                best_match_by_field[field] = self._update_best_match(best_match_by_field[field], match)
                            else:
                                best_match_by_field[field] = self._update_best_match(best_match_by_field[field], match)

                    for field in best_match_by_field.keys():
                        if field != 6:
                            assigned_fields.add(field)

                    final_matches = self._resolve_ambiguity_by_full_word(list(best_match_by_field.values()))

                    if final_matches:
                        best_match = final_matches[0]
                        results.append(best_match)

                        start = best_match.get("start")
                        end = best_match.get("end")

                        if start is not None and end is not None:
                            left_part = q[:start].strip()
                            right_part = q[end:].strip()

                            if left_part:
                                queue.append(left_part)
                            if right_part:
                                queue.append(right_part)

                            # logger.debug(f"Extracted '{best_match['key_word']}' from '{q}'. Remaining: '{left_part}', '{right_part}'")
            if single:
                if results:
                    logger.debug(f"RESULTS: {results}")
                return results if results else []
            return results
        except Exception as e:
            logger.error(f"Error buscando palabras clave: '{e}'", exc_info=True)
            return []

    def _resolve_ambiguity_by_full_word(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not matches:
            return []

        if len(matches) == 1:
            return matches

        for i, match in enumerate(matches):
            norm_ocr_text = match['norm_ocr_text']
            word_found = match['key_word']
            grams_text = self._build_query_grams(norm_ocr_text)

            if word_found in self.all_ngrams:
                _, grams_word = self.all_ngrams[word_found]
            else:
                grams_word = self._build_query_grams(word_found)

            # Calcular similitud base
            base_similarity = self._score_hybrid_greedy(grams_word, grams_text)

            # Penalización simétrica: min/max siempre da un valor entre 0 y 1 no importa cuál sea más largo, el resultado es el mismo

            length_penalty = self._length_penalty(len(norm_ocr_text), len(word_found))

            # Score final = similitud base * penalización por longitud
            match['score_final'] = base_similarity * length_penalty

            logger.debug(
                "EMPATE: Match #%d: campo: %s, palabra: '%s' | score de desempate: %.6f | texto: '%s'",
                i, match.get("key_field"), word_found, match['score_final'], norm_ocr_text
            )

        # Encontrar el mejor match usando max() en lugar de sort()
        best_match = max(matches, key=lambda x: (x['score_final'], len(x['key_word'])))

        logger.debug(
            "DESEMPATE: texto '%s': campo: %s, palabra: '%s', score_final: %.6f",
            best_match.get("text"), best_match.get("key_field"), best_match.get("key_word"),
            best_match.get("score_final")
        )
        return [best_match]

    def _build_query_grams(self, q: str) -> Dict[int, List[str]]:
        """Construye n-gramas de la consulta retornando LISTAS (Duplicados permitidos)"""
        gq: Dict[int, List[str]] = {}
        for n in range(self.ngrams[0], self.ngrams[1] + 1):
            gq[n] = self._ngrams(q, n)
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
        """Calcula la similitud entre dos n-gramas."""
        if not a or not b: return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / float(max(len(a), len(b)))

    def _score_hybrid_greedy(self, grams_cand: Dict[int, List[str]], grams_sub: Dict[int, List[str]]) -> float:
        """
        Calcula similitud híbrida "Greedy Unique Match" usando listas.
        No usa pesos por longitud de n-grama.
        """
        total_score = 0.0
        total_ngrams_cand = 0.0

        for n, cand_list in grams_cand.items():
            if not cand_list:
                continue

            num_cand = len(cand_list)
            total_ngrams_cand += num_cand

            sub_list = grams_sub.get(n, [])
            if not sub_list:
                continue

            # 1. Calcular todas las similitudes cruzadas posibles > 0
            possible_matches: List[Tuple[float, int, int]] = []
            for i, gc in enumerate(cand_list):
                for j, gs in enumerate(sub_list):
                    # gc y gs tienen garantizado tener la misma longitud 'n' aquí
                    if gc == gs:
                        sim = 1.0
                    else:
                        sim = self._ngram_similarity(gc, gs)
                    # Penalización simétrica
                    # sim *= self._length_penalty(gc, gs)

                    if sim > 0.0:
                        possible_matches.append((sim, i, j))

            # 2. Ordenar por score descendente (voraz)
            possible_matches.sort(key=lambda x: x[0], reverse=True)

            # 3. Asignar asegurando unicidad de índices
            used_cand: Set[int] = set()
            used_sub: Set[int] = set()
            section_score = 0.0

            for score, i, j in possible_matches:
                if i not in used_cand and j not in used_sub:
                    section_score += score
                    used_cand.add(i)
                    used_sub.add(j)
                    if len(used_cand) == num_cand:
                        break

            total_score += section_score

        if total_ngrams_cand == 0.0:
            return 0.0
        
        return total_score / total_ngrams_cand

    def _is_potential_keyword(self, q: str) -> bool:
        try:
            if not q:
                return False

            if q in self.global_words:
                return True

            # OPTIMIZACIÓN: Convertir string completo a integers UNA VEZ
            q_int = [ord(c) for c in q]

            total_soft_score = 0.0
            total_input_ngrams = 0

            for n, matrix_slice in self.global_matrices.items():
                # Generar n-gramas por slicing (sin ord)
                input_ngrams_int = [q_int[i:i + n] for i in range(len(q_int) - n + 1)]

                if not input_ngrams_int:
                    continue

                num_input = len(input_ngrams_int)
                total_input_ngrams += num_input

                matrix_input: np.ndarray[Any, np.dtype[np.uint8]] = np.array(input_ngrams_int, dtype=np.uint8)

                matches = (matrix_input[:, np.newaxis, :] == matrix_slice[np.newaxis, :, :])
                sim_matrix = matches.sum(axis=2) / n

                rows, cols = np.where(sim_matrix > 0)
                if rows.size > 0:
                    scores = sim_matrix[rows, cols]
                    prio_indices = np.argsort(-scores)

                    used_in: Set[int] = set()
                    used_gl: Set[int] = set()

                    for idx in prio_indices:
                        r, c = rows[idx], cols[idx]
                        if r not in used_in and c not in used_gl:
                            total_soft_score += float(scores[idx])
                            used_in.add(r)
                            used_gl.add(c)
                            if len(used_in) == num_input:
                                break

            if total_input_ngrams == 0:
                return False

            soft_coverage = total_soft_score / total_input_ngrams
            is_valid = soft_coverage > self.global_filter_threshold

            return is_valid

        except Exception as e:
            logger.error(f"Error en filtro matricial único: {e}")
            return False

    def _remove_noise_substrings(self, text: str) -> Tuple[str, List[str]]:
        cleaned = text
        removed_noise: List[str] = []
        try:
            candidates: List[Tuple[str, Dict[int, List[str]]]] = []
            for i, word in enumerate(self.noise_words):
                if word and i < len(self.noise_grams):
                    candidates.append((word, self.noise_grams[i]))

            for noise_word, grams_forbidden in candidates:
                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)

                found_any = True
                while found_any:
                    found_any = False
                    current_max_w = min(len(cleaned), noise_len + self.window_flex)

                    for w in range(current_max_w, min_w - 1, -1):
                        if w > len(cleaned):
                            continue
                        for j in range(0, len(cleaned) - w + 1):
                            sub = cleaned[j:j + w]

                            if sub == noise_word:
                                # 
                                similarity = 1.0 * self._length_penalty(w, noise_len)
                            else:
                                grams_sub = self._build_query_grams(sub)
                                similarity = self._score_hybrid_greedy(grams_forbidden, grams_sub)
                                # Penalización simétrica
                            similarity *= self._length_penalty(w, noise_len)

                            if similarity > self.forb_match:
                                cleaned = (cleaned[:j] + " " + cleaned[j + w:])
                                cleaned = _space_pattern.sub(" ", cleaned).strip()
                                removed_noise.append(sub)
                                logger.debug(f"SUBSTRING ELIMINADO: '{sub}' | Similitud: {similarity:.4f} | RUIDO ORIG: '{noise_word}'")
                                found_any = True
                                break
                        if found_any:
                            break
            return cleaned, removed_noise

        except Exception as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
            return text, []
            
    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            s = s.lower()  # 1. Entra el texto y convertimos a minusuculas
            s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")  # 2. Convertimos letras con con puntuación a su versión estandar, NO ELIMINAMOS PUNTUACIÓN SOLO TRATAMOS CON ALFABÉTICOS
            s = _nom_pattern.sub("", s)  # 3. Eliminar especiales con el patrón definido internos juntando los caracteres.
            # s = _space_clean_pattern.sub(" ", s)  # 4. Ahora sí eliminamos puntuación y números convirtiendolos en espacios sin juntar aún
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')  # 5. Normalizamos ASCII para estandarizar
            return _space_pattern.sub(" ", q).strip()  # 6. Normalizar espacios dobles que se hayan podido generar
        except UnicodeError as e:
            logger.error(f"Error limpiando texto: {e}", exc_info=True)
        return ""
                    
    def _update_best_match(self, current_best: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide si el nuevo match es mejor que el actual según las reglas de similitud y longitud.
        """
        if match["similarity"] > current_best["similarity"]:
            return match
        elif abs(match["similarity"] - current_best["similarity"]) < self.min_diff:
            if len(match["key_word"]) > len(current_best["key_word"]):
                return match
        return current_best
    
    def _length_penalty(self, a: int, b: int) -> float:
        """Penalización simétrica por diferencia de longitud."""
        return min(a, b) / max(a, b)