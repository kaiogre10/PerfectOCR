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
_nom_pattern= re.compile(r'(?<=[a-zA-Z])[^\w\s]+(?=[a-zA-Z])')

class WordFinder:
    def __init__(self, model_path: str, set_params: bool):
        model: Dict[str, Any] = self._load_model(model_path)
        if set_params:
            """Aquí va una función que obtendría los parametros de configuración del master_config
            pero me da flojera escribirla así que solo dejaré un log y no cambiaré el parametro "set_params"""
            logger.debug(f"Parametros establecidos y cargados manualmente")

        params: Dict[str, Any] = model.get("params", {})
        self.noised_filter = model["noise_filter"]
        self.globals_filter = model.get("global_filter", {})
        # Parametros de configuración
        self.threshold: float = params.get("threshold_similarity", {})
        self.global_filter_threshold: float = params.get("global_filter_threshold", {})
        self.ngrams_range: Tuple[int, int] = params["char_ngrams"]
        self.window_flex: int = params.get("window_flexibility", {})
        self.forb_match: float = params.get("forb_match", {})
        self.min_diff: float = params.get("min_diff", {})
        self.primes: Tuple[int, int] = params["primes"]
        
        self.global_vocab = self._global_vocab
        self.global_matrices = self._global_matrices
        self.maped_matrix = self._maped_matrix
        
        # self.all_ngrams = self._all_ngrams
        self.map_keys = self._map_keys
        self.global_words: FrozenSet[str] = frozenset(self._global_words)
        self.map_words = self._map_words
        self.hash_map = self._hash_map
        
        self.noise_words = self._noise_words
        
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
        return self.globals_filter
        
    @cached_property
    def noise_filter(self) -> Dict[int, Dict[str, str | Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]]]:
        return self.noised_filter
    
    @cached_property
    def _global_vocab(self) -> Dict[Tuple[int, int], Dict[str, List[str]]]:
        return self.global_filter[0]
        
    @cached_property
    def _map_keys(self) -> List[Tuple[int, int]]:
        return [w for w in self._global_vocab.keys()]
            
    @cached_property
    def _global_words(self) -> List[str]:
        return [list(w.keys())[0] for w in self._global_vocab.values()]
    
    @cached_property
    def _map_words(self):
        return list(zip(self._global_words, self.map_keys,))
                
    @cached_property
    def _global_matrices(self) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
        return self.global_filter[1]
            
    @cached_property
    def _maped_matrix(self) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
        return self.global_filter[2]
    
    @cached_property
    def _hash_map(self) -> Dict[int, np.ndarray[Any, np.dtype[np.uint32]]]:
        return self.global_filter[3]
        
    @cached_property
    def _noise_words(self) -> FrozenSet[str]:
        return frozenset([(w["noise_words"]) for w in self.noise_filter.values()]) #type: ignore
    
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

            results: List[Dict[str, Any]] = []
            
            while queue:
                q = queue.pop(0)
                if not q:
                    continue

                if q in self.noise_words:
                    # logger.info(f"Ruido temprano 2: '{list(self.noise_words).pop(list(self.noise_words).index(q))}'")
                    continue

                if q in self.global_words:
                    key_word, key_field = self.get_key_field(q, 0)
                    results.append(self._set_results(key_field[0], key_word, 1.0, q, q, 0, len(q)))
                    continue

                if not self._is_potential_keyword(q):
                    # logger.info(f"Texto no paso filtro global: {q}")
                    continue

                # ELIMINACIÓN DE RUIDO: No usa assigned_fields
                q_cleaned, removed_noise = self._remove_noise_substrings(q)
                if removed_noise:
                    q = q_cleaned

                found_matches_for_s: List[Dict[str, Any]] = []
                q_grams = self._build_query_grams(q)
                
                # FASE 1: Intersección Matricial Rápida
                candidate_ids: Set[int] = set()
                
                for len_mtx, grams_cand in self.maped_matrix.items():
                    if len_mtx not in q_grams or q_grams[len_mtx].size == 0 or grams_cand.size == 0:
                        continue
                        
                    q_ngrams = q_grams[len_mtx]
                    
                    weights = np.ones(len_mtx, dtype=np.int64)
                    weights[0] = self.primes[0]
                    
                    input_mask = np.sum(grams_cand.astype(np.int64) * weights, axis=1, dtype=np.int64)
                    global_mask = np.sum(q_ngrams.astype(np.int64) * weights, axis=1, dtype=np.int64)
                    
                    # Intersección: obtenemos TODOS los índices en grams_cand que hagan hit con algún ngrama de la query
                    hit_mask = np.isin(input_mask, global_mask)
                    matches_mask = np.where(hit_mask)[0]
                    
                    if matches_mask.size > 0:
                        # Extraer los IDs reales de esos hits y añadirlos al set
                        hits_ids = self.hash_map[len_mtx][matches_mask]
                        candidate_ids.update(hits_ids.tolist())

                if not candidate_ids:
                    continue

                # FASE 2: Scoring de Ventanas y Subcadenas para candidatos filtrados
                found_matches_for_s: List[Dict[str, Any]] = []

                for cand_id in candidate_ids:
                    cand, key_field = self.get_key_field("", cand_id) # type: ignore
                    cand_len = len(cand)
                    grams_cand = self.get_key_word_ngrams(cand)

                    # Buscamos dónde coinciden en la consulta q
                    hit_positions: List[int] = []
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
                        found_matches_for_s.append(self._set_results(key_field[0], cand, best_score_for_cand, str(text), q, best_sub_details["start"], best_sub_details["end"]))

                # Después de comprobar todos los candidatos, conservar todos los matches sobre threshold y resolver solo ambigüedades reales (solapamientos/empates).
                if found_matches_for_s:
                    final_matches = self._resolve_ambiguity_by_full_word(found_matches_for_s)

                    if final_matches:
                        results.extend(final_matches)
                        best_match = max(final_matches, key=lambda x: (x["similarity"], len(x["key_word"])))

                        start = best_match.get("start")
                        end = best_match.get("end")

                        if start is not None and end is not None:
                            left_part = q[:start].strip()
                            right_part = q[end:].strip()

                            if left_part:
                                queue.append(left_part)
                            if right_part:
                                queue.append(right_part)

                            # logger.info(f"Extracted '{best_match['key_word']}' from '{q}'. Remaining: '{left_part}', '{right_part}'")
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
            grams_text = self._build_query_grams(norm_ocr_text)
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

    def _build_query_grams(self, q: str) -> Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]:
        """Construye n-gramas de la consulta retornando LISTAS (Duplicados permitidos)"""
        gq: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        len_text = len(q)
        if len_text < self.ngrams_range[1]:
            max_ngram_range = len_text
        else:
            max_ngram_range = self.ngrams_range[1]
        
        for n in range(self.ngrams_range[0], max_ngram_range + 1):
            ngrams = [[ord(char) for char in ng] for ng in self._n_grams(q, n)]
            if not ngrams:
                continue
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

    def _ngram_similarity(self, a: str, b: str) -> float:
        """Calcula la similitud entre dos n-gramas."""
        if not a or not b: return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / float(max(len(a), len(b)))

    def _score_hybrid_greedy(self, grams_cand: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]], grams_sub: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]) -> float:
        """
        Calcula similitud híbrida "Greedy Unique Match" usando listas.
        No usa pesos por longitud de n-grama.
        """
        total_score = 0.0
        total_ngrams_cand = 0.0
        try:
            for n, cand_arrays in grams_cand.items():
                if cand_arrays.size < 1:
                    continue

                num_cand = cand_arrays.shape[0]
                total_ngrams_cand += num_cand

                sub_array = grams_sub[n]
                if not sub_array:
                    continue
                # input_mask = np.sum((sub_array * p1).astype(np.int64), axis=1, dtype=np.int64)
                # global_mask = np.sum((cand_arrays * p1).astype(np.int64), axis=1, dtype=np.int64)
                # matches_mask = np.intersect1d(input_mask, global_mask, assume_unique=False, return_indices=True)[1]
                # num_match = matches_mask.shape[0]
                # 1. Calcular todas las similitudes cruzadas posibles > 0
                possible_matches: List[Tuple[float, int, int]] = []
                for i, gc in enumerate(cand_arrays):
                    for j, gs in enumerate(sub_array):
                        # gc y gs tienen garantizado tener la misma longitud 'n' aquí
                        if gc == gs:
                            sim = 1.0
                        else:
                            sim = self._ngram_similarity(gc, gs)
                        # Penalización simétrica
                        sim *= self._length_penalty(gc, gs)

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
        except Exception as e:
            logger.debug(f"Error: {e}", exc_info=True)
        return 0.0

    def _is_potential_keyword(self, q: str) -> bool:
        # time_fil = time.perf_counter()
        try:
            if not q:
                return False
            if q in self.global_words:
                return True
                
            word_len = len(q)
            if word_len < 2:
                return False
                
            q_arr = self._build_query_grams(q)
            
            total_soft_score = 0.0
            total_input_ngrams = 0
            for n, matrix_slice in self.global_matrices.items():
                if word_len < n:
                    total_soft_score += n
                    continue
                
                matrix_input = q_arr[n]
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
            
            for noisy_dict in self.noise_filter.values():
                noise_word: str = noisy_dict["noise_words"] # type: ignore
                grams_forbidden: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = noisy_dict.get("noise_grams", {}) # type: ignore
                noise_len = len(noise_word)
                min_w = max(1, noise_len - self.window_flex)
                # logger.info("\n"f"{noise_word}")
                found_any = True
                while found_any:
                    found_any = False
                    len_clean = len(cleaned)
                    current_max_w = min(len_clean, noise_len + self.window_flex)

                    for w in range(current_max_w, min_w - 1, -1):
                        if w > len_clean:
                            continue
                        for j in range(0, len_clean - w + 1):
                            sub = cleaned[j:j + w]

                            if sub == noise_word and w == noise_len:
                                similarity = 1.0
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
            # logger.info(f"Tiempo limpiando ruido: {time.perf_counter() - timer:.6f}'s")
            return cleaned, removed_noise

        except Exception as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
            return text, []

    def text_normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            s = s.lower()
            s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
            s = _nom_pattern.sub("", s)
            s = _space_clean_pattern.sub(" ", s)
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
            return _space_pattern.sub(" ", q).strip()
                        
        except UnicodeError as e:
            logger.error(f"Error limpiando texto: {e}", exc_info=True)
        return ""
    
    def _length_penalty(self, la: int, lb: int) -> float:
        """Penalización simétrica por diferencia de longitud."""
        return 1.0 if la == lb else float(min(la, lb) / max(la, lb))
        
    def get_key_field(self, word: str, encoded_key: int) -> Tuple[str, Tuple[int, int]]:
        if word:
            for kword, word_map in self.map_words:
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
        field_id, id_ = idx
        concatenated_index = (field_id * self.primes[0]) + id_
        # 2. Para cada tamaño de ngrama
        for n in range(self.ngrams_range[0], self.ngrams_range[1] + 1):
            hash_arr = self.hash_map[n]
            matrix = self.maped_matrix[n]
            # 3. Buscar los índices donde el hash coincide
            indices = np.where(hash_arr == concatenated_index)[0]
            if indices.size > 0:
                ngrams_len_dict[n] = matrix[indices]
        return ngrams_len_dict
            
    def _set_results(self, key_field: int, key_word: str, similarity :float, text: str, norm_ocr_text: str, start: int, end: int) -> Dict[str, Any]:
        return {
            "key_field": key_field,
            "key_word": key_word,
            "similarity": similarity,
            "text": text,
            "norm_ocr_text": norm_ocr_text,
            "start": start,
            "end": end
        }
