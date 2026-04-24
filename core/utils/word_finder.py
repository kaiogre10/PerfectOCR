import os
import logging
import numpy as np
import pickle
import re
import unicodedata
import time
from functools import cached_property
from typing import List, Any, Dict, Tuple, Set, FrozenSet

logger = logging.getLogger(__name__)

_space_pattern = re.compile(r"\s+")
_space_clean_pattern = re.compile(r"[^a-z\s]+")
_nom_pattern = re.compile(r'(?<=[a-zA-Z])[^\w\s]+(?=[a-zA-Z])')

class WordFinder:
    def __init__(self, model_path: str, set_params: bool):
        model: Dict[str, Any] = self._load_model(model_path)
        if set_params:
            """Aquí va una función que obtendría los parametros de configuración del master_config pero me da flojera escribirla así que solo dejaré un log y no cambiaré el parametro "set_params"""
            logger.debug(f"Parametros establecidos y cargados manualmente")

        params: Dict[str, Any] = model.get("params", {})
        self.noise_filter = model["noise_filter"]
        self.global_filter = model.get("global_filter", {})
        # Parametros de configuración
        self.threshold: float = params.get("threshold_similarity", {})
        self.global_filter_threshold: float = params.get("global_filter_threshold", {})
        self.ngrams_range: Tuple[int, int] = params["char_ngrams"]
        self.window_flex: int = params.get("window_flexibility", {})
        self.forb_match: float = params.get("forb_match", {})
        self.min_diff: float = params.get("min_diff", {})
        self.primes: Tuple[int, int] = params["primes"]
        # logger.info(f"{self.map_words}")

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

    @cached_property
    def global_filter(self):
        return self.global_filter

    @cached_property
    def noise_filter(self) -> Dict[str, Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]]:
        return self.noise_filter

    @cached_property
    def map_words(self) -> Dict[Tuple[int, int], str]:
        return self.global_filter[0]

    @cached_property
    def global_words(self) -> FrozenSet[str]:
        return frozenset([w for w in self.map_words.values()])

    @cached_property
    def global_matrices(self) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
        return self.global_filter[1]

    @cached_property
    def maped_matrix(self) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
        return self.global_filter[2]

    @cached_property
    def hash_map(self) -> Dict[int, np.ndarray[Any, np.dtype[np.uint32]]]:
        return self.global_filter[3]

    @cached_property
    def inverse_index(self) -> Dict[int, Dict[Tuple, np.ndarray[Any, np.dtype[np.uint32]]]]:
        return self.global_filter[4]

    @cached_property
    def noise_words(self) -> FrozenSet[str]:
        return frozenset([w for w in self.noise_filter.keys()])

    def find_keywords(self, text: List[str] | str) -> List[Dict[str, Any]]:
        try:
            if not text:
                return []

            elif len(text) < 2:
                return []

            single = False
            if isinstance(text, str):
                s = self.text_normalize(text)
                queue = [s]
                single = True
            else:
                queue = [self.text_normalize(s) for s in text if self.text_normalize(s)]

            if s in self.global_words:
                key_word, key_field = self.get_key_field(s, 0)
                #logger.info(f"Match incial: '{key_word}' {key_field[0]}")
                return [self._set_results(key_field[0], key_word, 1.0, text, key_word, 0, len(queue))]

            results: List[Dict[str, Any]] = []
            while queue:
                q = queue.pop(0)
                if not q:
                    continue
                if q in self.noise_words:
                    # logger.info(f"Ruido temprano 2: '{list(self.noise_words).pop(list(self.noise_words).index(q))}'")
                    continue

                if not self._is_potential_keyword(q):
                    # logger.info(f"Texto no paso filtro global: {q}")
                    continue

                # ELIMINACIÓN DE RUIDO: No usa assigned_fields
                q_cleaned, removed_noise = self._remove_noise_substrings(q)
                if removed_noise:
                    # logger.info(f"CLEANED: '{q_cleaned}' REMOVED: '{removed_noise}', INICIAL: '{q}'")
                    q = q_cleaned

                found_matches_for_s: List[Dict[str, Any]] = []
                q_grams = self.build_query_grams(q)
                # logger.info(f"ULTIMAS INSTNACIAS: '{q}'")

                # FASE 1: Intersección Matricial Rápida
                candidate_ids: Set[int] = set()
                # idx_list =  [idx[1] for idx in self.inverse_index.items()]

                q_len = len(q)
                # logger.info(f"{q}")
                matrixes_votes = []
                total_ngrams = 0
                for size_ngram, grams_cand in self.maped_matrix.items():
                    if q_grams is None or grams_cand is None:
                        continue

                    index_cands = []
                    ngram_index = self.inverse_index[size_ngram]
                    q_ngrams = q_grams[size_ngram]

                    for row in q_ngrams:
                        val = ngram_index.get(tuple(row))
                        if val is None or val.size < 1:
                            continue
                        # textstring = ["".join(chr(int(c))) for c in row]
                        # logger.info(f"{textstring}, {row}, {val}")
                        index_cands.extend(val)
                    concat_idx = np.array(index_cands)

                    index_conc = np.unique(concat_idx, return_counts=True)
                    # logger.info("COUNTS:\n"f"{np.column_stack([index_conc[0], index_conc[1]])}")

                    decoded_clas = self.decode_array(index_conc[0])
                    votes = np.column_stack([decoded_clas, index_conc[1]])

                    total_ngrams += len(q_ngrams)
                    matrixes_votes.extend(votes)

                total_votes = np.array(matrixes_votes)
                keys = total_votes[:, :-1]
                unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
                sums = np.bincount(inverse, weights=total_votes[:, -1])
                votes_matrix = np.column_stack([unique_keys, sums])

                complete_match = np.where(votes_matrix[:, -1] == total_ngrams)[0]
                if complete_match is None or complete_match.size < 1:
                    continue
                votes_matrix = votes_matrix[complete_match]
                kf_arr = self.get_key_field_arr(votes_matrix[:, [0, 1]])
                # logger.info(f"POSIBLES MATCHES PARA: '{q}':\n"f"{kf_arr}")
                valid_kw = np.array([len(k[1]) for k in kf_arr], np.uint8)
                votes_matrix = np.column_stack([votes_matrix[:, [0, 1]], valid_kw, votes_matrix[:, -1]])
                # logger.info(f"AÑADIDO LEN(Q):\n"f"{votes_matrix}")
                votes_matrix = self.length_penalty_vect(votes_matrix, q_len)
                # logger.info("Penalizada:\n"f"{np.array2string(votes_matrix, precision=4)}")
                cands = [k[1] for k in kf_arr]

                dict_cands: Dict[str, Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]] = {}
                for cand in cands:
                    ngrams_cand = self.get_key_word_ngrams(cand)
                    dict_cands[cand] = ngrams_cand
                # logger.info("DICT CANDS:\n"f"{dict_cands}")

                kf_matches: Dict[str, Any] = []
                total_score = 0.0
                idxs = []
                for cand_kf, gram_cands in dict_cands.items():
                    for i, cand_ngrams in gram_cands.items():
                        q_ngrams = q_grams[i]

                        rows = cand_ngrams.shape[0]  # type: ignore
                        if q_ngrams.shape[0] < rows:
                            continue

                        windows = np.lib.stride_tricks.sliding_window_view(q_ngrams, (rows, i), (0, 1))
                        for idx, window in enumerate(windows):
                            windowr = window.reshape(rows, i)
                            #    logger.info("WINDOW:\n"f"{windowr}")
                            mean_match = self.ngram_similarity_vec(cand_ngrams, windowr)
                            if mean_match < 0.1:
                                continue

                            total_score += mean_match
                            idxs.append(idx)

                    final_score = total_score / len(list(range(self.ngrams_range[0], self.ngrams_range[1] + 1)))
                    if final_score > self.threshold:
                        #logger.info(f"{idxs}")
                        _, key_field = self.get_key_field(cand_kf, 0)
                        kf_matches = {
                            "key_field": key_field[0],
                            "key_word": cand_kf,
                            "similarity": final_score,
                            "text": text,
                            "norm_ocr_text": q,
                        }
                if not kf_matches:
                    continue
                else:
                    results.append(kf_matches)

                # logger.info(f"{results}")

                # weights = np.ones(size_ngram, dtype=np.int64)
                # weights[0] = self.primes[0]

                # input_mask = np.sum(grams_cand.astype(np.int64) * weights, axis=1, dtype=np.int64)
                # global_mask = np.sum(q_ngrams.astype(np.int64) * weights, axis=1, dtype=np.int64)

                # # Intersección: obtenemos TODOS los índices en grams_cand que hagan hit con algún ngrama de la query
                # match_mask = np.intersect1d(input_mask, global_mask, assume_unique=False, return_indices=True)[1]
                # num_match = match_mask.shape[0]

                # hit_mask = np.isin(input_mask, global_mask)
                # matches_mask = np.where(hit_mask)[0]
                # # logger.info(f"{match_mask}")
                # # logger.info(f"INTERSECT: {num_match}, ISIN: {matches_mask.size}")

                # if matches_mask.size > 0:
                #     # Extraer los IDs reales de esos hits y añadirlos al set
                #     hits_ids = self.hash_map[size_ngram][matches_mask]
                #     candidate_ids.update(hits_ids.tolist())

                if not candidate_ids:
                    continue

                # FASE 2: Scoring de Ventanas y Subcadenas para candidatos filtrados
                found_matches_for_s: List[Dict[str, Any]] = []
                cand_len = len(cand)
                for cand_id in candidate_ids:
                    cand, key_field = self.get_key_field("", cand_id)  # type: ignore

                    grams_cand = self.get_key_word_ngrams(cand)

                    # Buscamos dónde coinciden en la consulta q
                    hit_positions: List[int] = []
                    # max_matches: List[Tuple[float, int]] = []
                    for n, sub_arr in grams_cand.items():
                        if n not in q_grams:
                            continue

                        weights = np.ones(n, dtype=np.int64)
                        weights[0] = self.primes[0]
                        cand_mask = np.sum(sub_arr.astype(np.int64) * weights, axis=1, dtype=np.int64)
                        q_mask = np.sum(q_grams[n].astype(np.int64) * weights, axis=1, dtype=np.int64)
                        # Buscamos las posiciones de hit dentro de la consulta 'q'
                        q_hits = np.intersect1d(q_mask, cand_mask, assume_unique=False, return_indices=True)[1]

                        for idx in q_hits:
                            gram_hit = q_grams[n][idx]
                            sub_str = "".join(chr(c) for c in gram_hit)

                            start_pos = 0
                            while True:
                                pos = q.find(sub_str, start_pos)
                                if pos == -1:
                                    break
                                hit_positions.append(pos)
                                start_pos = pos + 1

                    if not hit_positions:
                        continue

                    best_score_for_cand: float = 0.0
                    best_sub_details: Dict[str, int] = {}

                    # Agrupamos posiciones cercanas para no probar la misma zona mil veces
                    sorted_unique_hits = sorted(list(set(hit_positions)))

                    min_w = max(1, cand_len - self.window_flex)
                    max_w = cand_len + self.window_flex

                    for hit_start_pos in sorted_unique_hits:
                        for w in range(min_w, max_w + 1):
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
                                    grams_sub = self.build_query_grams(sub)
                                    final_score = self._score_hybrid_greedy(grams_cand, grams_sub)
                                    final_score *= self._length_penalty(w, cand_len)

                                if final_score > best_score_for_cand:
                                    best_score_for_cand = final_score
                                    best_sub_details = {
                                        "start": start,
                                        "end": end
                                    }

                    if best_score_for_cand > self.threshold:
                        found_matches_for_s.append(
                            self._set_results(key_field[0], cand, best_score_for_cand, str(text), q,
                                              best_sub_details["start"], best_sub_details["end"]))

                # Después de comprobar todos los candidatos, conservar todos los matches sobre threshold y resolver solo ambigüedades reales (solapamientos/empates).
                if found_matches_for_s:
                    final_matches = self._resolve_ambiguity_by_full_word(found_matches_for_s)

                    if final_matches:
                        results.extend(final_matches)
                        spans: List[Tuple[int, int]] = []
                        for match in final_matches:
                            start = match.get("start")
                            end = match.get("end")
                            if start is None or end is None:
                                continue
                            start_i = int(start)
                            end_i = int(end)
                            if start_i < 0 or end_i > len(q) or start_i >= end_i:
                                continue
                            spans.append((start_i, end_i))

                        if spans:
                            spans.sort(key=lambda x: x[0])
                            merged_spans: List[Tuple[int, int]] = []
                            for start_i, end_i in spans:
                                if not merged_spans or start_i > merged_spans[-1][1]:
                                    merged_spans.append((start_i, end_i))
                                else:
                                    prev_start, prev_end = merged_spans[-1]
                                    merged_spans[-1] = (prev_start, max(prev_end, end_i))

                            cursor = 0
                            for start_i, end_i in merged_spans:
                                fragment = q[cursor:start_i].strip()
                                if fragment:
                                    queue.append(fragment)
                                cursor = end_i
                            tail_fragment = q[cursor:].strip()
                            if tail_fragment:
                                queue.append(tail_fragment)
            if single:
                if results:
                    logger.info(f"RESULTS: {results}")
                return results if results else []
            return results
        except Exception as e:
            logger.debug(f"Error buscando palabras clave: '{e}'", exc_info=True)
            return []

    def _resolve_ambiguity_by_full_word(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not matches:
            return []

        if len(matches) == 1:
            return matches

        scored_matches: List[Dict[str, Any]] = []
        for i, match in enumerate(matches):
            norm_ocr_text = match["norm_ocr_text"]
            word_found = match["key_word"]
            grams_text = self.build_query_grams(norm_ocr_text)
            grams_word = self.get_key_word_ngrams(word_found)
            base_similarity = self._score_hybrid_greedy(grams_word, grams_text)
            length_penalty = self._length_penalty(len(norm_ocr_text), len(word_found))
            score_final = base_similarity * length_penalty
            local_match = dict(match)
            local_match["score_final"] = score_final
            scored_matches.append(local_match)
            logger.debug(
                "EMPATE: Match #%d: campo: %s, palabra: '%s' | score de desempate: %.6f | texto: '%s'",
                i, match.get("key_field"), word_found, score_final, norm_ocr_text
            )

        # Mantener todos; solo descartar en conflictos por solapamiento.
        scored_matches.sort(key=lambda x: (x["score_final"], x["similarity"], len(x["key_word"])), reverse=True)
        selected: List[Dict[str, Any]] = []
        for cand in scored_matches:
            start_c = int(cand.get("start", -1))
            end_c = int(cand.get("end", -1))
            conflict_idx = -1
            for i, prev in enumerate(selected):
                start_p = int(prev.get("start", -1))
                end_p = int(prev.get("end", -1))
                overlaps = (start_c < end_p) and (start_p < end_c)
                if overlaps:
                    conflict_idx = i
                    break

            if conflict_idx == -1:
                selected.append(cand)
                continue

            prev = selected[conflict_idx]
            better = (
                         cand["score_final"],
                         cand["similarity"],
                         len(cand["key_word"]),
                     ) > (
                         prev["score_final"],
                         prev["similarity"],
                         len(prev["key_word"]),
                     )
            if better:
                selected[conflict_idx] = cand

        cleaned_selected: List[Dict[str, Any]] = []
        for match in selected:
            cleaned_match = dict(match)
            cleaned_match.pop("score_final", None)
            cleaned_selected.append(cleaned_match)
        return cleaned_selected

    def build_query_grams(self, q: str) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
        """Construye n-gramas de la consulta retornando LISTAS (Duplicados permitidos)"""
        gq: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        len_text = len(q)
        if len_text < self.ngrams_range[1]:
            max_ngram_range = len_text
        else:
            max_ngram_range = self.ngrams_range[1]

        for n in range(self.ngrams_range[0], max_ngram_range + 1):
            ngrams = [[ord(char) for char in ng] for ng in self._n_grams(q, n)]
            gq[n] = np.array(ngrams)
        return gq

    def _n_grams(self, q: str, n: int) -> List[str]:
        try:
            if n <= 0 or not q:
                return []
            if len(q) < n:
                return []
            return [q[i:i + n] for i in range(len(q) - n + 1)]
        except Exception as e:
            logger.error(f"Error construyendo n-gramas: {e}", exc_info=True)
            return []

    def ngram_similarity_vec(self, a: np.ndarray[Any, np.dtype[np.uint8]],
                             b: np.ndarray[Any, np.dtype[np.uint8]]) -> float:
        ngram_size = a.shape[1]
        if ngram_size != b.shape[1] or b.shape[0] != a.shape[0]:
            return np.empty(1, np.float32)
        matches = np.count_nonzero(a == b, axis=1, keepdims=True)
        return np.mean(matches / ngram_size)

    def _score_hybrid_greedy(self, grams_cand: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]],
                             grams_sub: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]) -> float:
        """
        Calcula similitud híbrida "Greedy Unique Match" usando listas.
        No usa pesos por longitud de n-grama.
        """
        total_score = 0.0
        total_ngrams_cand = 0.0
        try:
            for n, cand_arrays in grams_cand.items():
                if cand_arrays is None or cand_arrays.size < 1:
                    continue

                num_cand = cand_arrays.shape[0]
                total_ngrams_cand += num_cand

                sub_array = grams_sub.get(n)
                if sub_array is None or sub_array.size < 1:
                    continue

                # 1. Calcular similitudes cruzadas en bloque [num_cand, num_sub]
                matches = (cand_arrays[:, np.newaxis, :] == sub_array[np.newaxis, :, :])
                sim_matrix = matches.sum(axis=2) / float(n)
                rows, cols = np.where(sim_matrix > 0.0)
                if rows.size == 0:
                    continue

                # 2. Ordenar por score descendente (voraz)
                scores = sim_matrix[rows, cols]
                prio_indices = np.argsort(-scores)

                # 3. Asignar asegurando unicidad de índices
                used_cand = np.zeros(num_cand, dtype=bool)
                used_sub = np.zeros(sub_array.shape[0], dtype=bool)
                section_score = 0.0

                for idx in prio_indices:
                    i = int(rows[idx])
                    j = int(cols[idx])
                    if not used_cand[i] and not used_sub[j]:
                        section_score += float(scores[idx])
                        used_cand[i] = True
                        used_sub[j] = True
                        if np.count_nonzero(used_cand) == num_cand:
                            break

                total_score += section_score

            if total_ngrams_cand == 0.0:
                return 0.0

            return total_score / total_ngrams_cand
        except Exception as e:
            logger.debug(f"Error: {e}", exc_info=True)
        return 0.0

    def _is_potential_keyword(self, q: str) -> bool:
        # time_fil = time.perf_counter()
        try:
            if not q:
                return False

            word_len = len(q)
            if word_len < 2:
                return False
            # elif word_len < self.ngrams_range[0]:
            q_arr = self.build_query_grams(q)

            total_soft_score = 0.0
            total_input_ngrams = 0
            for n, matrix_slice in self.global_matrices.items():
                if matrix_slice is None or word_len < n:
                    total_soft_score += n
                    continue

                matrix_input = q_arr[n]
                if matrix_input is None or matrix_input.size == 0:
                    continue
                num_input = matrix_input.shape[0]
                total_input_ngrams += num_input

                weights = np.ones(n, dtype=np.int64)
                weights[0] = self.primes[0]

                input_mask = np.sum(matrix_input.astype(np.int64) * weights, axis=1, dtype=np.int64)
                global_mask = np.sum(matrix_slice.astype(np.int64) * weights, axis=1, dtype=np.int64)
                matches_mask = np.intersect1d(input_mask, global_mask, assume_unique=False, return_indices=True)[1]
                num_match = matches_mask.shape[0]
                if num_match == num_input:
                    total_soft_score += num_match
                    continue

                total_soft_score += num_match
                all_indices = np.arange(num_input)
                no_match_indices = np.setdiff1d(all_indices, matches_mask)
                matrix_input = matrix_input[no_match_indices]

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
                # logger.info(f"'{q}' - total_input_ngrams == 0")
                return False

            soft_coverage = total_soft_score / total_input_ngrams
            # logger.info(f"'{q}' SIMILITUD GLOBAL: {soft_coverage}, score={total_soft_score}, input={total_input_ngrams}")
            # logger.info(f"Tiempo del filtro: {time.perf_counter() - time_fil:.6f}'s")
            return soft_coverage > self.global_filter_threshold

        except Exception as e:
            logger.error(f"Error de '{q}' en filtro matricial: {e}", exc_info=True)
            return False

    def _remove_noise_substrings(self, text: str) -> Tuple[str, List[str]]:
        # timer = time.perf_counter()
        try:
            cleaned = text
            removed_noise: List[str] = []
            if not cleaned:
                return cleaned, removed_noise

            clean_grams = self.build_query_grams(cleaned)
            for noise_word, grams_forbidden in self.noise_filter.items():
                noise_len = len(noise_word)
                if noise_len > len(cleaned):
                    continue

                max_matches: List[Tuple[float, int]] = []
                for i, forb_grams in grams_forbidden.items():
                    clean_gram = clean_grams.get(i)
                    if clean_gram is None or forb_grams is None or clean_gram.size == 0 or forb_grams.size == 0:
                        continue

                    rows = forb_grams.shape[0]
                    if clean_gram.shape[0] < rows:
                        continue

                    windows = np.lib.stride_tricks.sliding_window_view(clean_gram, (rows, i), (0, 1))
                    for idx, window in enumerate(windows):
                        windowr = window.reshape(rows, i)
                        mean_match = self.ngram_similarity_vec(forb_grams, windowr)
                        if mean_match > self.forb_match:
                            max_matches.append((float(mean_match), idx))

                if not max_matches:
                    continue

                best_match = max(max_matches, key=lambda x: x[0])
                best_idx = int(best_match[1])
                # logger.info("\n"f"MAX_MATCHES: {best_idx}")

                j = best_idx
                cleaned = (cleaned[:j] + " " + cleaned[j + noise_len:])
                cleaned = _space_pattern.sub(" ", cleaned).strip()
                alpha_count = sum(1 for c in cleaned if c.isalpha())
                if alpha_count < 2:
                    cleaned = ""
                    break
                removed_noise.append(noise_word)
                clean_grams = self.build_query_grams(cleaned)
                # logger.info(f"{cleaned}")

                if not cleaned:
                    break
            # logger.info(f"Tiempo limpiando ruido: {time.perf_counter() - timer:.6f}'s")
            return cleaned, removed_noise

        except Exception as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
            return text, []

    def text_normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            s = s.lower()  # 1. Entra el texto y convertimos a minusuculas
            s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(
                ch) != "Mn")  # 2. Convertimos letras con con puntuación a su versión estandar, NO ELIMINAMOS PUNTUACIÓN SOLO TRATAMOS CON ALFABÉTICOS
            s = _nom_pattern.sub("",
                                 s)  # 3. Eliminar especiales con el patrón definido internos juntando los caracteres.
            s = _space_clean_pattern.sub(" ",
                                         s)  # 4. Ahora sí eliminamos puntuación y números convirtiendolos en espacios sin juntar aún
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode(
                'utf-8')  # 5. Normalizamos ASCII para estandarizar
            return _space_pattern.sub(" ", q).strip()  # 6. Normalizar espacios dobles que se hayan podido generar
        except UnicodeError as e:
            logger.error(f"Error limpiando texto: {e}", exc_info=True)
        return ""

    def _length_penalty(self, la: int, lb: int) -> float:
        """Penalización simétrica por diferencia de longitud."""
        return 1.0 if la == lb else float(min(la, lb) / max(la, lb))

    def length_penalty_vect(self, matrix: np.ndarray[Any, np.dtype[np.uint8]], word_len: int):
        penalty = np.divide(np.minimum(word_len, matrix[:, 2]), np.maximum(word_len, matrix[:, 2]), dtype=np.float32)
        new_sim = np.multiply(penalty, matrix[:, -1], dtype=np.float32)
        penalized_matrix = np.column_stack([matrix[:, [0, 1]], penalty, new_sim])
        return np.sort(penalized_matrix, 0)[::-1]

    def decode_array(self, keys: np.ndarray[Any, np.dtype[np.uint32]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        decoded_kf = keys // self.primes[0]
        decoded_kw = keys - (self.primes[0] * decoded_kf)
        return np.column_stack([decoded_kf, decoded_kw]).astype(np.uint8)

    def get_key_field_arr(self, keys: np.ndarray[Any, np.dtype[np.uint8]]) -> List[Tuple[int, str]]:
        results: List[Tuple[int, str]] = []
        for key in keys:
            word_map = self.map_words.get(tuple(key))
            if word_map is not None:
                results.append((int(key[0]), word_map))
        # logger.info(f"RESULTS: {results}")
        return results

    def get_key_field(self, word: str, encoded_key: int) -> Tuple[str, Tuple[int, int]]:
        if word:
            for word_map, kword in self.map_words.items():
                if kword != word:
                    continue
                return kword, word_map

        elif encoded_key > self.primes[0]:
            code_key = self.primes[0]
            decoded_kf = (encoded_key // code_key)
            decoded_kw = encoded_key - (code_key * decoded_kf)

            for kword, word_map in self.map_words:
                if (decoded_kf, decoded_kw) != word_map:
                    continue
                return kword, (decoded_kf, decoded_kw)
        return ("", (0, 0))

    def get_key_word_ngrams(self, key_word: str) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
        ngrams_len_dict: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        # 1. Obtener el índice de la keyword
        _, idx = self.get_key_field(key_word, 0)
        field_id, word_id = idx
        concatenated_index = (field_id * self.primes[0]) + word_id
        # 2. Para cada tamaño de ngrama
        for n in range(self.ngrams_range[0], self.ngrams_range[1] + 1):
            hash_arr = self.hash_map[n]
            matrix = self.maped_matrix[n]
            # 3. Buscar los índices donde el hash coincide
            indices = np.where(hash_arr == concatenated_index)[0]
            if indices.size > 0:
                ngrams_len_dict[n] = matrix[indices]
            continue
        return ngrams_len_dict

    def _set_results(self, key_field: int, key_word: str, similarity: float, text: str, norm_ocr_text: str, start: int,
                     end: int) -> Dict[str, Any]:
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
