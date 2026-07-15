import logging
import numpy as np
# import time
from typing import List, Any, Dict, Tuple, Set, FrozenSet
from core.assets.patterns import space_pattern
from domain.model_factory import MatrixFactory
from utils.compiled_utils import ngram_similarity, length_penalty
import scipy.sparse as sp # type: ignore

logger = logging.getLogger(__name__)

kf_list: FrozenSet[int] = frozenset({1, 2, 3, 4, 6})
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
        "kf_file",
        "_all_kf_idx"
    )
    def __init__(self, config: Dict[bytes, Any], motor: MatrixFactory):
        self.motor = motor
        _model = self.motor.model_pkl
        
        self.noise_cands: List[Tuple[bytes, Dict[int, List[bytes]]]] = _model["bnoise_cands"]
        self.all_ngrams: Dict[bytes, Tuple[int, Dict[int, List[bytes]]]] = _model.get("ball_ngrams", {})
        self.global_words: FrozenSet[bytes] = frozenset(_model["bglobal_words"])
        self.noise_words: FrozenSet[bytes] = frozenset(_model["bnoise_words"])
        
        self.global_filter_threshold: float = config.get("global_filter_threshold", {})
        self.threshold: float = config.get("threshold_similarity", {})
        self.ngrams_len = config["char_ngrams"]
        self.window_flex: int = config.get("window_flexibility", {})
        self.forb_match: float = config.get("forb_match", {})
        self.kf_file = config.get("kf_path", "")
        self._all_kf_idx = None
    
    @property
    def all_kf_idx(self):
        if self._all_kf_idx is None:
            return np.arange(((2 + self.ngrams_len[1]) - self.ngrams_len[0]), dtype=np.uint8)
        else:
            return self._all_kf_idx
        
    def get_sparce_matrix(self, n: int):
        try:
            mtx = self.motor.matrix_registry[n]
            matrix = mtx.matrix
            return sp.csr_matrix((matrix['data'], matrix['indices'], matrix['indptr']), shape=tuple(matrix['shape']))
        except ImportError as e:
            logger.error(f"ERROR ARMANDO MATRIX: {e}", exc_info=True)
        return sp.csr_matrix(0)
    
    def keyfield_ngrams(self, key_field: int, n: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        KEYFIELD, len(NGRAM)
        Devuelve el diccionario de keyfielgramas por KeyField separado
        """
        id_matrix = rf"{key_field}_{n}_{self.kf_file}"
        kfngrams = self.motor.kf_registry[n]
        kf_mtx = kfngrams.kf_matrix
        return kf_mtx.get(id_matrix)

    def find_keywords(self, text: List[bytes] | bytes) -> List[Dict[bytes, Any]]:
        if not text:
            return []
        
        elif text in self.noise_words:
            return []
        
        elif text in self.global_words:
            key = self.all_ngrams.get(text) # type: ignore
            if key and key[0] > 0:
                # logger.info(f"MATCH TEMPRANO: '{text}'")
                return [self._set_results(key[0], text, 1.0, text, text, 0, len(text))]
        
        single = False
        if isinstance(text, bytes):
            queue = [text]
            single = True
        else:
            queue = text
        
        results: List[Dict[bytes, Any]] = []
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
                if not self._is_potential_keyword(q_clean):
                    continue
        
                q = q_clean
            
            found_matches_for_s: List[Dict[bytes, Any]] = []
            word_len = len(q)
            
            if q in self.global_words:
                key = self.all_ngrams.get(q)
                if key and key[0] > 0:
                    start = text.find(q)    # type: ignore
                    end = start + word_len
                    found_matches_for_s.append(self._set_results(key[0], text, 1.0, text, q, start, end))
                    continue
            
            arr_q = np.frombuffer(q, np.uint8)
            
            belonged_fields: List[int] = []
            for keyfield in sorted(kf_list):
                if not self.belongs_keyfield(keyfield, arr_q):
                    continue
                else:
                    belonged_fields.append(keyfield)
                
            if not belonged_fields:
                continue
            
            # logger.info(f"'{q}' KF POTENCIALES: {not_belonged_fields}")
            
            for cand, (key_field, grams_cand) in self.all_ngrams.items():
                
                if key_field != 6 and key_field in assigned_fields:
                    continue
                    
                if key_field == 5:
                    continue
                    
                if key_field not in belonged_fields:
                    continue
                    
                cand_len = len(cand)
                if cand_len > word_len:
                    continue

                if q == cand:
                    # logger.info(f"MATCH TEMPRANO: '{q}'")
                    start = text.find(q)    # type: ignore
                    end = start + word_len
                    found_matches_for_s.append(self._set_results(key_field, q, 1.0, text, q, start, end))
                    continue
                    
                cand_arr = np.frombuffer(cand, np.uint8)
                hit_positions = self.get_hits_pos(cand_arr, arr_q)
                
                if not hit_positions:
                    continue
        
                best_score_for_cand: float = 0.0
                best_sub_details: Dict[bytes, int] = {}
                
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
                            
                            if start < 0:
                                continue
                            
                            end = start + w
                            if end > word_len:
                                continue
                                
                            sub = q[start:end].strip()
                            if not sub:
                                continue
        
                            elif sub == cand:
                                final_score = 1.0 * length_penalty(w, cand_len)
        
                            else:
                                grams_sub = self._build_query_grams(sub, len(sub))
                                final_score = self._score_hybrid_greedy(grams_cand, grams_sub)
        
                            if final_score > best_score_for_cand:
                                best_score_for_cand = final_score
                                best_sub_details = {"start": start, "end": end}
        
                if best_score_for_cand > self.threshold:
                    found_matches_for_s.append(self._set_results(key_field, cand, best_score_for_cand, text, q, best_sub_details["start"], best_sub_details["end"]))
            # Después de comprobar todos los candidatos, agrupar y seleccionar el mejor por campo
            if found_matches_for_s:
                best_match_by_field: Dict[int, Dict[bytes, Any]] = {}
        
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
        
                    if start or end:
                        left_part = q[:start].strip()
                        right_part = q[end:].strip()
                        
                        if not left_part and not right_part:
                            continue
                            
                        if left_part:
                            queue.append(left_part)
                        if right_part:
                            queue.append(right_part)
        
                        # logger.info(f"EXRAIDO: '{best_match['key_word']}' DE '{q}'. SOBRAN: '{left_part}', '{right_part}'")

        if single:
            if results:
                logger.debug(f"RESULTS: {text}: {[kf["key_word"] for kf in results]}")
            return results if results else []
        return results
        # except Exception as e:
        #     logger.error(f"Error buscando palabra clave: '{text}' = '{e}'", exc_info=True)
        #     return []

    def _resolve_ambiguity_by_full_word(self, matches: List[Dict[bytes, Any]]) -> List[Dict[bytes, Any]]:
        if not matches:
            return []

        if len(matches) == 1:
            return matches

        for i, match in enumerate(matches):
            norm_ocr_text = match['norm_ocr_text']
            word_found = match['key_word']
            len_ocr = len(norm_ocr_text)
            len_word = len(word_found)
            grams_text = self._build_query_grams(norm_ocr_text, len_ocr)

            if word_found in self.all_ngrams:
                _, grams_word = self.all_ngrams[word_found]
            else:
                grams_word = self._build_query_grams(word_found, len_word)

            # Calcular similitud base
            base_similarity = self._score_hybrid_greedy(grams_word, grams_text)

            # Penalización simétrica: min/max siempre da un valor entre 0 y 1 no importa cuál sea más largo, el resultado es el mismo
            # Score final = similitud base * penalización por longitud
            match['score_final'] = base_similarity * length_penalty(len_ocr, len_word)

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

    def _build_query_grams(self, q: bytes, len_text: int) -> Dict[int, List[bytes]]:
        """Construye n-gramas de la consulta retornando LISTAS (Duplicados permitidos)"""
        gq: Dict[int, List[bytes]] = {}
        for n in range(self.ngrams_len[0], self.ngrams_len[1] + 1):
            if len_text < n:
                continue
                
            gq[n] = [q[i:i + n] for i in range(len_text - n + 1)]
        return gq

    def _score_hybrid_greedy(self, grams_cand: Dict[int, List[bytes]], grams_sub: Dict[int, List[bytes]]) -> float:
        """Calcula similitud híbrida "Greedy Unique Match"""
        total_score = 0.0
        total_ngrams_cand = 0.0

        for n, cand_list in grams_cand.items():
            
            sub_list = grams_sub.get(n, [])
            if not sub_list:
                continue
                
            if not cand_list:
                continue
                
            num_cand = len(cand_list)
            total_ngrams_cand += num_cand

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
                        sim = ngram_similarity(gc, gs)
        
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

    def _is_potential_keyword(self, q: bytes) -> bool:
        # time_fil = time.perf_counter()
        try:
            if not q:
                return False
            
            elif q in self.noise_words:
                return  False

            elif q in self.global_words:
                return True
        
            word_len = len(q)
            
            total_input_ngrams = 0
            total_soft_score_vect: float = 0.0
            q_arr = np.frombuffer(buffer=q, dtype=np.uint8)
            for n in range(self.ngrams_len[0], self.ngrams_len[1] + 1):
                if word_len < n:
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
                
                # all_kf_idx = np.arange(matrix_keywords.shape[1], dtype=np.uint8)
                
                # logger.info(f"N{n}:\n"f"{all_kf_idx}\n"f"NEW: {self.all_kf_idx[:n]}")
                no_match_kf_idx = np.setdiff1d(self.all_kf_idx[:n], matches_mask, assume_unique=True)

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
            logger.error(f"Error de {q} en filtro matricial: {e}", exc_info=True)
        return False

    def _remove_noise_substrings(self, text: bytes) -> Tuple[bytes, List[bytes]]:
        cleaned = text
        removed_noise: List[bytes] = []
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
                            subc = cleaned[j:j + w].strip()
                            if subc == noise_word:
                                similarity = 1.0 * length_penalty(w, noise_len)

                            else:
                                # sub_array = np.frombuffer(subc, dtype=np.uint8)
                                grams_sub = self._build_query_grams(subc, len(subc))
                                similarity = self._score_hybrid_greedy(grams_forbidden, grams_sub)
                                # Penalización simétrica
                            # similarity *= length_penalty(w, noise_len)

                            if similarity > self.forb_match:
                                cleaned = cleaned[:j] + b" " + cleaned[j + w:]
                                cleaned = _space_pattern.sub(b" ", cleaned)
                                removed_noise.append(subc)
                                # logger.info(f"SUBSTRING ELIMINADO: '{sub}' | Similitud: {similarity:.4f} | RUIDO ORIG: '{noise_word}'")
                                found_any = True
                                break

                        if found_any:
                            break

            return cleaned, removed_noise

        except AttributeError as e:
            logger.error(f"Error eliminando substrings de ruido: {e}", exc_info=True)
        return text, []

    def _update_best_match(self, current_best: Dict[bytes, Any], match: Dict[bytes, Any]) -> Dict[bytes, Any]:
        """Decide si el nuevo match es mejor que el actual según las reglas de similitud y longitud."""
        if match["similarity"] > current_best["similarity"]:
            return match
        elif abs(match["similarity"] - current_best["similarity"]) < 0.000009:
            if len(match["key_word"]) > len(current_best["key_word"]):
                return match
        return current_best

    def _set_results(self, key_field: int, key_word: bytes, similarity :float, text: bytes, norm_ocr_text: bytes, start: int, end: int) -> Dict[bytes, Any]:
        """
        Construye un diccionario con los resultados de la búsqueda de palabra clave.
        Parámetros:
            key_field (int): Identificador del campo clave.
            key_word (bytes): Palabra clave encontrada.
            similarity (float): Puntaje de similitud calculado.
            text (bytes): Texto original procesado.
            norm_ocr_text (bytes): Texto normalizado de OCR.
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

    def get_hits_pos(self, cand: np.ndarray[Any, np.dtype[np.uint8]], q_int: np.ndarray[Any, np.dtype[np.uint8]]) -> List[int]:
        # time0 = time.perf_counter()
        idx_pos: Set[int] = set()
        total_nrmas = 0
        total_hits = 0
        for n in range(self.ngrams_len[0], self.ngrams_len[1] + 1):
            if cand.size >= n and q_int.size >= n:
                
                cand_vec = np.lib.stride_tricks.sliding_window_view(cand, n)
                
                total_input = np.lib.stride_tricks.sliding_window_view(q_int, n)
                _, idx_unique = np.unique(total_input, axis=0, return_index=True)
                total_all_ngrams = total_input.shape[0]
            
                if idx_unique.size < total_all_ngrams:
                    input_nrmas = total_input[np.sort(idx_unique)].shape[0]
                else:
                    input_nrmas = total_all_ngrams
                
                gc_idx = np.where((total_input[:, None] == cand_vec[None, :]).all(axis=2))[0]
                
                idx_pos.update(gc_idx)
                total_hits += gc_idx.shape[0]
                total_nrmas += input_nrmas
                
            else:
                continue
        
        if total_hits == 0 or not idx_pos:
            return []
        else:
            score = total_hits / total_nrmas
            return [] if score < 0.25 else sorted(idx_pos)
    
    def belongs_keyfield(self, key_field: int, q_arr: np.ndarray[Any, np.dtype[np.uint8]]) -> bool:
        belong_score = 0.0
        total_input_ngrams = 0
        for n in range(self.ngrams_len[0], self.ngrams_len[1] + 1):
            if q_arr.size < n:
                continue
            
            total_input = np.lib.stride_tricks.sliding_window_view(q_arr, n)
            _, idx_unique = np.unique(total_input, axis=0, return_index=True)
            total_ngrams = total_input.shape[0]
            
            if idx_unique.size < total_ngrams:
                input_vec = total_input[np.sort(idx_unique)]
                posible_matches = input_vec.shape[0]
            else:
                input_vec = total_input
                posible_matches = total_ngrams
            
            total_input_ngrams += posible_matches
            
            matrix_kf = self.keyfield_ngrams(key_field, n)
            matches_idx = np.where((input_vec[:, None] == matrix_kf[None, :]).all(axis=2))
            
            matches_mask = matches_idx[0]
            input_idx = matches_idx[1]
            
            kf_matches = matches_mask.shape[0]
            
            # logger.info(f"MATCHES: {kf_matches} {matches_mask} POSIBLES: {posible_matches}")
            belong_score += kf_matches
            
            if kf_matches == posible_matches:
                continue
            
            all_indices = np.arange(posible_matches, dtype=np.uint8)
            no_match_indices = np.setdiff1d(all_indices, input_idx, assume_unique=True)
            
            all_kf_idx = np.arange(matrix_kf.shape[1], dtype=np.uint8)
            no_match_kf_idx = np.setdiff1d(all_kf_idx, matches_mask, assume_unique=True)
            
            matrix_sparce = self.get_sparce_matrix(n)
            cross_points = matrix_sparce[no_match_indices, :][:, no_match_kf_idx]
            mtx_sims = cross_points.data

            total_sims = mtx_sims.size
            sims_left = posible_matches - kf_matches

            if total_sims < 1 or sims_left < 1 or (sims_left - total_sims) < 1:
                continue

            belong_score += sum(mtx_sims)

        if total_input_ngrams == 0:
            return False

        total_score = belong_score / total_input_ngrams
        
        # logger.info(f"SCORE KF: {key_field}: {total_score}")
        return 0.30 < total_score