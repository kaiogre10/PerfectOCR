import logging
import numpy as np
# import time
from typing import List, Any, Dict, Tuple, Set, FrozenSet
from utils.patterns import space_pattern
from domain.model_factory import MatrixFactory
import scipy.sparse as sp # type: ignore

logger = logging.getLogger(__name__)

_space_pattern = space_pattern

class WordFinder:
    __slots__ = (
        "motor",
        "noise_cands",
        "noise_words",
        "all_ngrams",
        "global_words",
        "global_filter_threshold",
        "threshold",
        "ngrams_len",
        "window_flex",
        "forb_match",
        "kf_file"
    )
    def __init__(self, config: Dict[str, Any], motor: MatrixFactory):
        self.motor = motor
        _model = self.motor.model_pkl
        
        self.noise_cands: List[Dict[int, List[str]]] = _model["noise_cands"]
        self.all_ngrams: Dict[str, Tuple[int, Dict[int, List[str]]]] = _model.get("all_ngrams", {})
        self.global_words: FrozenSet[str] = frozenset(_model["global_words"])
        self.noise_words: FrozenSet[str] = frozenset(_model["noise_words"])

        self.global_filter_threshold: float = config.get("global_filter_threshold", {})
        self.threshold: float = config.get("threshold_similarity", {})
        self.ngrams_len = config["char_ngrams"]
        self.window_flex: int = config.get("window_flexibility", {})
        self.forb_match: float = config.get("forb_match", {})
        self.kf_file = config.get("kf_path", "")
    
    # def idx_matrix(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
    #     idx = self.motor.index_matrix
    #     return idx.idx_matrix
        
    def get_sparce_matrix(self, n: int):
        try:
            mtx = self.motor.matrix_registry[n]
            matrix = mtx.matrix
            return sp.csr_matrix((matrix['data'], matrix['indices'], matrix['indptr']), shape=tuple(matrix['shape']))
        except ImportError as e:
            logger.error(f"ERROR ARMANDO MATRIX: {e}", exc_info=True)
        return sp.csr_matrix(0)
    
    def get_keyfield_ngrams(self, key_field: int, n: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        KEYFIELD, len(NGRAM)
        Devuelve el diccionario de keyfielgramas por KeyField separado
        """
        id_matrix = rf"{key_field}_{n}_{self.kf_file}"
        kfngrams = self.motor.kf_registry[n]
        kf_mtx = kfngrams.kf_matrix
        return kf_mtx.get(id_matrix)

    def find_keywords(self, text: List[str] | str) -> List[Dict[str, Any]]:
        try:
            if not text:
                return []

            elif text in self.noise_words:
                return []

            elif text in self.global_words:
                txt = str(text)
                key = self.all_ngrams.get(txt)
                if key and key[0] > 0:
                    # logger.info(f"MATCH TEMPRANO: '{text}'")
                    return [self._set_results(key[0], txt, 1.0, txt, txt, 0, len(txt))]

            single = False
            if isinstance(text, str):
                queue = [text]
                single = True
            else:
                queue = text

            results: List[Dict[str, Any]] = []
            assigned_fields: Set[int] = set()

            while queue:
                q = queue.pop(0)
                if not q:
                    continue

                if q in self.noise_words:
                    # logger.info(f"Ruido temprano 2: '{list(self.noise_words).pop(list(self.noise_words).index(q))}'")
                    continue
                
                if not self._is_potential_keyword(q):
                    # logger.info(f"Texto no paso filtro global: '{q}'")
                    continue

                # ELIMINACIÓN DE RUIDO: No usa assigned_fields
                q_clean, removed_noise = self._remove_noise_substrings(q)
                if removed_noise:
                    if not self._is_potential_keyword(q):
                        continue

                    q = q_clean

                found_matches_for_s: List[Dict[str, Any]] = []

                q_int = q.encode('ascii', 'ignore')
                for cand, (key_field, grams_cand) in self.all_ngrams.items():
                    if key_field != 6 and key_field in assigned_fields:
                        continue

                    if q == cand:
                        # logger.info(f"MATCH TEMPRANO: '{q}'")
                        found_matches_for_s.append(self._set_results(key_field, q, 1.0, str(text), q, 0, len(q)))
                        continue

                    # logger.info(f"INPUT: {q}, CAND: '{cand}'")
                    hit_positions = self.get_hits_pos(grams_cand, q_int)

                    if not hit_positions:
                        continue

                    # logger.info(f"{len(hit_positions)} HIT POSITIONS: {hit_positions}, TIEMPO LOOP : {(time.perf_counter() - timelo) + time_inv:.8f}'s")

                    best_score_for_cand: float = 0.0
                    best_sub_details: Dict[str, int] = {}
                    cand_len = len(cand)
                    # Definimos el rango de tamaños de ventana a probar
                    min_w = max(1, cand_len - self.window_flex)
                    max_w = cand_len + self.window_flex

                    # Iteramos sobre los puntos de inicio de los n-gramas coincidentes
                    for hit_start_pos in hit_positions:
                        # Probamos ventanas de diferentes tamaños centradas cerca del 'hit'
                        for w in range(min_w, max_w + 1):
                            # El inicio de la ventana debe permitir que el 'hit' esté dentro
                            # Probamos algunos desplazamientos para la ventana
                            for offset in range(-self.window_flex, 1):
                                start = hit_start_pos + offset
                                end = start + w

                                sub = q[start:end]
                                if not sub:
                                    continue

                                elif start < 0 or end > len(q):
                                    continue

                                elif sub == cand:
                                    penalty = self._length_penalty(w, cand_len)
                                    final_score = 1.0 * penalty

                                else:
                                    grams_sub = self._build_query_grams(sub)
                                    final_score = self._score_hybrid_greedy(grams_cand, grams_sub)
                                # final_score *= self._length_penalty(w, cand_len)

                                if final_score > best_score_for_cand:
                                    best_score_for_cand = final_score
                                    best_sub_details = {
                                        "start": start,
                                        "end": end
                                    }

                    if best_score_for_cand > self.threshold:
                        found_matches_for_s.append(self._set_results(key_field, cand, best_score_for_cand, str(text), q, best_sub_details["start"], best_sub_details["end"]))
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

                            logger.debug(f"EXRAIDO: '{best_match['key_word']}' DE '{q}'. SOBRAN: '{left_part}', '{right_part}'")
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
        len_text = len(q)
        for n in range(self.ngrams_len[0], self.ngrams_len[1] + 1):
            gq[n] = [] if not q or len(q) < n else [q[i:i + n] for i in range(len_text - n + 1)]
        return gq

    def _ngram_similarity(self, a: str, b: str, nf: int) -> float:
        """Calcula la similitud suave entre dos n-gramas."""
        s = 0.0
        for i in range(nf):
            s += (a[i] == b[i])
        return s / nf

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
            for j, gs in enumerate(sub_list):
                gs_set = set(gs)
                for i, gc in enumerate(cand_list):

                    # gc y gs tienen garantizado tener la misma longitud 'n' aquí
                    if gc == gs:
                        sim = 1.0

                    elif gs_set.isdisjoint(set(gc)):
                        sim = 0.0

                    else:
                        sim = self._ngram_similarity(gc, gs, n)
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

    def _is_potential_keyword(self, q_str: str) -> bool:
        # time_fil = time.perf_counter()
        try:
            if not q_str:
                return False
            
            elif q_str in self.noise_words:
                return  False

            elif q_str in self.global_words:
                return True
        
            q = q_str.encode('ascii', 'ignore')
            word_len = len(q)
            
            if word_len < self.ngrams_len[0]:
                return False

            total_input_ngrams = 0
            total_soft_score_vect: float = 0.0
            q_arr = np.frombuffer(buffer=q, dtype=np.uint8)
            for n in range(self.ngrams_len[0], self.ngrams_len[1] + 1):
                if word_len < n:
                    total_soft_score_vect += 1
                    continue
                    
                matrix_total = np.lib.stride_tricks.sliding_window_view(q_arr, n)
                total_ngrams = matrix_total.shape[0]
                
                _, idx_unique = np.unique(matrix_total, axis=0, return_index=True)
                if idx_unique.size < total_ngrams:
                    matrix_input = matrix_total[np.sort(idx_unique)]
                    num_input = matrix_input.shape[0]
                else:
                    matrix_input = matrix_total
                    num_input = total_ngrams
                    
                total_input_ngrams += num_input
                
                matrix_keywords = self.motor.matrix_registry[n].matrix_ngrams
                sim_idx = np.where((matrix_keywords[:, None] == matrix_input[None, :]).all(axis=2))
                matches_mask = sim_idx[0]  # Índices de los key words con match
                input_idx = sim_idx[1]  # Índices del input con match

                num_match = matches_mask.shape[0]

                if num_match == num_input:
                    total_soft_score_vect += num_match
                    continue

                total_soft_score_vect += num_match
                all_indices = np.arange(num_input, dtype=np.uint8)
                no_match_indices = np.setdiff1d(all_indices, input_idx, assume_unique=True)

                all_kf_idx = np.arange(matrix_keywords.shape[1], dtype=np.uint8)
                no_match_kf_idx = np.setdiff1d(all_kf_idx, matches_mask, assume_unique=True)

                matrix_sparce = self.get_sparce_matrix(n)
                cross_points = matrix_sparce[no_match_indices, :][:, no_match_kf_idx]
                mtx_sims = cross_points.data

                total_sims = mtx_sims.size
                sims_left = num_input - num_match

                if total_sims < 1 or sims_left < 1 or (sims_left - total_sims) < 1:
                    continue

                total_soft_score_vect += sum(mtx_sims)

            if total_input_ngrams == 0:
                # logger.info(f"'{q}' - total_input_ngrams == 0")
                return False

            soft_coverage = total_soft_score_vect / total_input_ngrams
            # logger.info(f"{q} SIMILITUD GLOBAL: {soft_coverage}, score={total_soft_score_vect}, input={total_input_ngrams}")
            # logger.info(f"Tiempo del filtro: {time.perf_counter() - time_fil:.8f}'s")
            return soft_coverage > self.global_filter_threshold

        except Exception as e:
            logger.error(f"Error de {q_str} en filtro matricial: {e}", exc_info=True)
        return False

    def _remove_noise_substrings(self, text: str) -> Tuple[str, List[str]]:
        cleaned = text
        removed_noise: List[str] = []
        try:
            for noise_word, grams_forbidden in self.noise_cands:
                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)

                found_any = True
                while found_any:
                    found_any = False

                    len_clean = len(cleaned)
                    current_max_w = min(len_clean, noise_len + self.window_flex)

                    for w in range(current_max_w, min_w - 1, -1):
                        if w > len_clean:
                            continue
                        for j in range(0, len_clean - w + 1):
                            sub = cleaned[j:j + w].strip()
                            if sub == noise_word:
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
                                # logger.info(f"SUBSTRING ELIMINADO: '{sub}' | Similitud: {similarity:.4f} | RUIDO ORIG: '{noise_word}'")
                                found_any = True
                                break

                        if found_any:
                            break

            return cleaned, removed_noise

        except AttributeError as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
        return text, []

    def _update_best_match(self, current_best: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
        """Decide si el nuevo match es mejor que el actual según las reglas de similitud y longitud."""
        if match["similarity"] > current_best["similarity"]:
            return match
        elif abs(match["similarity"] - current_best["similarity"]) < 0.000009:
            if len(match["key_word"]) > len(current_best["key_word"]):
                return match
        return current_best

    def _length_penalty(self, a: int, b: int) -> float:
        """Penalización simétrica por diferencia de longitud."""
        return min(a, b) / max(a, b)

    def _set_results(self, key_field: int, key_word: str, similarity :float, text: str, norm_ocr_text: str, start: int, end: int) -> Dict[str, Any]:
        """
        Construye un diccionario con los resultados de la búsqueda de palabra clave.
        Parámetros:
            key_field (int): Identificador del campo clave.
            key_word (str): Palabra clave encontrada.
            similarity (float): Puntaje de similitud calculado.
            text (str): Texto original procesado.
            norm_ocr_text (str): Texto normalizado de OCR.
            start (int): Índice de inicio de la coincidencia en el texto.
            end (int): Índice de fin de la coincidencia en el texto.
        """
        return {
            "key_field": key_field,
            "key_word": key_word,
            "similarity": similarity,
            "text": text,
            "norm_ocr_text": norm_ocr_text,
            "start": start,
            "end": end
        }

    def get_hits_pos(self, grams_cand: Dict[int, List[str]], q_int: bytes) -> List[int]:
        # time0 = time.perf_counter()
        idx_pos: Set[int] = set()
        for n in range(self.ngrams_len[0], self.ngrams_len[1] + 1):
            # ngrams_sorted = self.get_ngrams_sorted(n)
            cand_list = [ngram.encode('ascii', 'ignore') for ngram in grams_cand[n]]
            plain_cand = b"".join(cand_list)
            cand_vec = np.frombuffer(plain_cand, np.uint8).reshape(len(cand_list), n)

            input_ngrams_int = [q_int[i:i + n] for i in range(len(q_int) - n + 1)]
            plain_input = b"".join(input_ngrams_int)
            input_vec = np.frombuffer(plain_input, dtype=np.uint8).reshape(len(input_ngrams_int), n)

            gc_idx = np.where((input_vec[:, None] == cand_vec[None, :]).all(axis=2))[0]
            idx_pos.update(gc_idx)
        
        # logger.info(f"TIEMPO HITS: {time.perf_counter() - time0:.6f}s")
        return [] if not idx_pos else sorted(idx_pos)
    
    # def build_vect_grams(self, q: str, len_text: int) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
    #     """Construye n-gramas de la consulta retornando LISTAS (Duplicados permitidos)"""
    #     gq: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    #     q: bytes = q.encode("ascii", 'ignore')
    #     if len_text < self.ngrams_len[1]:
    #         max_ngram_range = len_text
    #     else:
    #         max_ngram_range = self.ngrams_len[1]
    #
    #     if len_text < self.ngrams_len[0]:
    #         return {}
    #
    #     for n in range(self.ngrams_len[0], max_ngram_range + 1):
    #         plain_ngrams = [q[i:i+n] for i in range(len_text - n + 1)]
    #         plain_b = b"".join(plain_ngrams)
    #         gq[n] = np.frombuffer(plain_b, np.uint8).reshape(len(plain_ngrams), n)
    #     return gq
