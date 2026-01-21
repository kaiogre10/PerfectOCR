import os
import numpy as np
#ort datetime
import logging
import pickle
import re
import unicodedata
import time
#om datetime import datetime
from typing import List, Any, Dict, Optional, Tuple, Set

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str):
        self.model: Dict[str, Any] = self._load_model(model_path)
        self.wf_path: str = "C:/word_finder_model/src/word_finder.py"
        self.params = self.model.get("params", {})
        self.all_ngrams: Dict[str, Dict[int, List[str]]] = self.model.get("all_ngrams", {})
        self.global_words: List[str] = self.model["global_words"]
        self.noise_words = self.model["noise_words"]
        noise_filter = self.model.get("noise_filter", {})
        global_filter = self.model.get("global_filter", {})
        self.global_filter_threshold = float(self.params.get("global_filter_threshold"))
        self.noise_grams: List[Tuple[str, float]] = noise_filter["noise_grams"]
        self.threshold: float = self.params.get("threshold_similarity")
        self.ngrams: Tuple[int, int] = self.params["char_ngrams"]
        self.window_flex = self.params.get("window_flexibility")
        self.forb_match: float = self.params.get("forb_match")
        self.global_matrices: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = global_filter.get("global_matrices", {})
        # timestamp_model = os.path.getmtime(self.wf_path)
        # fecha_wf = datetime.fromtimestamp(timestamp_model).isoformat()
        # logger.critical(f"FECHA DE GENERACIÓN DEL MODELO: {self.model_time}, FECHA DEL SCRIPT WORD_FINDER.PY: {fecha_wf}")

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if not isinstance(self.model, dict):  # type: ignore
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            logger.info(f"Modelo cargado en: '{time.perf_counter() - t0:.6f}s'")
            return self.model
        except Exception as e:
            logger.error(f"Error al cargar el modelo {e}", exc_info=True)
            raise

    def find_keywords(self, text: List[str] | str, threshold: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
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

                q = self._normalize(s)
                if not q:
                    continue 

                # Paso 1: Filtro Global (Matricial - Fast Fail)
                if not self._is_potential_keyword(q):
                    continue
                
                q_cleaned, removed_noise = self._remove_noise_substrings(q)
                if removed_noise:
                    logger.debug(f"Ruido eliminado: '{removed_noise}' | Texto Limpio: '{q_cleaned}'")
                    q = q_cleaned

                if self.check_full_word(text=q, place="noise"):
                    logger.info(f"Input completo es ruido, ignorando: '{q}'")
                    if not q:
                        continue

                # NUEVO: Rastrear campos ya asignados (excluyendo campo 6 que puede tener múltiples)
                assigned_fields: Set[int] = set()
                
                # Lista para guardar todos los matches de este string 's'
                found_matches_for_s: List[Dict[str, Any]] = []

                # Paso 2: Búsqueda Detallada optimizada usando all_ngrams
                for cand, (key_field, grams_cand) in self.all_ngrams.items():
                    # NUEVO: Saltar candidatos de campos ya asignados (excepto campo 6)
                    if key_field != 6 and key_field in assigned_fields:
                        continue
                    
                    cand_len = len(cand)
                    min_w = max(1, cand_len - self.window_flex)
                    size = len(q) 
                    if min_w > size or size == 1:
                        continue

                    max_w = min(len(q), cand_len + self.window_flex)

                    try:
                        for w in range(min_w, max_w + 1):
                            if w > len(q):
                                break
                            for j in range(0, len(q) - w + 1):
                                sub = q[j:j + w]
                                
                                # Exact match shortcut
                                if sub == cand and len(sub) == cand_len:
                                    final_score: float = 1.0
                                else:
                                    # Generamos n-gramas del fragmento del input (On-the-fly)
                                    grams_sub = self._build_query_grams(sub)
                                    
                                    # Calculamos similitud usando lógica híbrida greedy
                                    final_score = self._score_hybrid_greedy(grams_cand, grams_sub)
                                    
                                    # Penalización por diferencia de longitud
                                    len_ratio = max(len(sub), cand_len) / max(1, min(len(sub), cand_len))
                                    if len_ratio >= 2.0:
                                        penalty = min(len(sub), cand_len) / max(len(sub), cand_len)
                                        final_score *= penalty

                                if final_score > final_threshold:
                                    found_matches_for_s.append({
                                        "key_field": key_field,
                                        "word_found": cand,
                                        "similarity": float(final_score),
                                        "text": s,
                                        "start": j,
                                        "end": j + w
                                    })

                    except Exception as e:
                        logger.error(f"Error en el bucle de búsqueda de find_keywords: {e}", exc_info=True)
            
                # Después de comprobar todos los candidatos, agrupar y seleccionar el mejor por campo
                if found_matches_for_s:
                    best_match_by_field: Dict[int, Dict[str, Any]] = {}
                
                    for match in found_matches_for_s:
                        field = match["key_field"]
                        
                        # Si es el primer match para este campo, lo guardamos
                        if field not in best_match_by_field:
                            best_match_by_field[field] = match
                        else:
                            # Campo 6 puede tener múltiples matches, otros campos solo uno
                            if field == 6:
                                # Para campo 6, guardar todos los matches (o el mejor si quieres)
                                # Por ahora mantenemos solo el mejor también
                                current_best = best_match_by_field[field]
                                if match["similarity"] > current_best["similarity"]:
                                    best_match_by_field[field] = match
                                elif abs(match["similarity"] - current_best["similarity"]) < 1e-9:
                                    if len(match["word_found"]) > len(current_best["word_found"]):
                                        best_match_by_field[field] = match
                            else:
                                # Otros campos: solo un match, elegir el mejor
                                current_best = best_match_by_field[field]

                                # 1. Si la similitud es estrictamente mayor, reemplazamos.
                                if match["similarity"] > current_best["similarity"]:
                                    best_match_by_field[field] = match
                                # 2. Si hay empate (diferencia despreciable), preferimos la palabra MÁS LARGA (Maximal Munch)
                                elif abs(match["similarity"] - current_best["similarity"]) < 1e-9:
                                    if len(match["word_found"]) > len(current_best["word_found"]):
                                        best_match_by_field[field] = match
                    
                    # NUEVO: Marcar campos asignados (excepto campo 6)
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

                            logger.debug(f"Extracted '{best_match['word_found']}' from '{q}'. Remaining: '{left_part}', '{right_part}'")
            if single:
                logger.debug(f"RESULTS: {results}")
                return results if results else []
            return results
        except Exception as e:
            logger.error(f"Error buscando palabras clave: '{e}'", exc_info=True)

    def _resolve_ambiguity_by_full_word(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not matches:
            return []
        
        if len(matches) == 1:
            return matches

        for i, match in enumerate(matches):
            # CORREGIDO: Normalizar el texto antes de comparar
            original_text = self._normalize(match['text'])
            keyword_found = self._normalize(match['word_found'])

            # Construir n-gramas de todo el texto (sin ventana deslizante)
            grams_text = self._build_query_grams(original_text)
            
            if keyword_found in self.all_ngrams:
                _, grams_keyword = self.all_ngrams[keyword_found]
            else:
                grams_keyword = self._build_query_grams(keyword_found)

            tiebreaker_score = self._score_hybrid_greedy(grams_keyword, grams_text)
            
            # AÑADIR: Penalización por diferencia de longitud
            len_keyword = len(keyword_found)
            len_text = len(original_text)
            len_ratio = max(len_text, len_keyword) / max(1, min(len_text, len_keyword))
            if len_ratio >= 1.5:  # Usar 1.5 para ser más estricto
                penalty = min(len_text, len_keyword) / max(len_text, len_keyword)
                tiebreaker_score *= penalty

            match['score_final'] = tiebreaker_score

            logger.info(
                "EMPATE: Match #%d: campo: %s, palabra: '%s' | score de desempate: %.4f | texto: '%s'",
                i, match.get("key_field"), match.get("word_found"), tiebreaker_score, original_text
            )

        # Ordenar por score_final, y si hay empate, por longitud de palabra (Maximal Munch)
        matches.sort(key=lambda x: (x['score_final'], len(x['word_found'])), reverse=True)

        logger.info(
            "DESEMPATE: texto '%s': campo: %s, palabra: '%s', score_final: %.4f",
            matches[0].get("text"), matches[0].get("key_field"), matches[0].get("word_found"), matches[0].get("score_final"),
        )
        return [matches[0]]

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
            if not q or not self.global_matrices:
                return False
            
            if self.check_full_word(text=q, place="global"):
                return True

            # OPTIMIZACIÓN: Convertir string completo a integers UNA VEZ
            q_int = [ord(c) for c in q]
            
            total_soft_score = 0.0
            total_input_ngrams = 0

            for n, matrix_slice in self.global_matrices.items():
                # Generar n-gramas por slicing (sin ord)
                input_ngrams_int = [q_int[i:i+n] for i in range(len(q_int) - n + 1)]
                
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
            
            # candidates.sort(key=lambda x: len(x[0]), reverse=True)

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
                                similarity = 1.0
                            else:
                                grams_sub = self._build_query_grams(sub)
                                similarity = self._score_hybrid_greedy(grams_forbidden, grams_sub)
                                
                                len_ratio = max(len(sub), noise_len) / max(1, min(len(sub), noise_len))
                                if len_ratio >= 2.0:
                                    penalty = min(len(sub), noise_len) / max(len(sub), noise_len)
                                    similarity *= penalty

                            if similarity > self.forb_match:
                                cleaned = (cleaned[:j] + " " + cleaned[j + w:]).strip()
                                cleaned = re.sub(r"\s+", " ", cleaned).strip()
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
    
    def check_full_word(self, text: str, place: str) -> bool:
        try:
            if place == "global":
                return text in set(self.global_words)
            if place == "noise":
                return text in set(self.noise_words)
        except Exception as e:
            logger.info(f"Error buscando string inmediatos: {e}")
        return False
        
    def _normalize(self, s: str) -> str:
        try:
            if not s:
                return ""
            q = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()
            # Eliminar cualquier cosa que no sea letra o espacio (SIN inyectar espacios nuevos)
            q = re.sub(r"[^a-z\s]+", "", q)
            # Si quieres seguir limpiando espacios múltiples / extremos:
            q = re.sub(r"\s+", " ", q).strip()
            return q
        except Exception as e:
            logger.error(msg=f"Error limpiando texto: {e}", exc_info=True)
        return ""