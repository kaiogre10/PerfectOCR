import os
import re
import logging
import pickle
import unicodedata
from typing import List, Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class WordFinder:
    def __init__(self, model_path: str):
        self._active = "standard"
        self.model_path = model_path
        self.model: Dict[str, Any] = self._load_model(model_path)
        self._apply_active_model()

    def _normalize(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^a-zA-Z0-9]", "", s)  # quita espacios, signos y también _
        return s

    def _ngrams(self, s: str, n: int) -> List[str]:
        if n <= 0 or not s:
            return []
        if len(s) < n:
            return []
        return [s[i:i+n] for i in range(len(s) - n + 1)]

    def _binary_cosine(self, size_a: int, size_b: int, inter_size: int) -> float:
        """Calcula coseno binario entre dos conjuntos"""
        if size_a == 0 or size_b == 0:
            return 0.0
        return inter_size / ((size_a * size_b) ** 0.5)

    def available_models(self) -> List[str]:
        return ["standard"]

    def set_active_model(self, name: str) -> bool:
        if name == "standard":
            self._active = "standard"
            return True
        logger.warning("Modelo solicitado no disponible: %s", name)
        return False

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, dict):
            raise ValueError("El pickle no tiene el formato esperado (dict).")
        return model

    def _apply_active_model(self):
        # listas
        self.global_words: List[str] = [self._normalize(str(w)) for w in self.model.get("global_words", []) or []]
        self.header_words: List[str] = [self._normalize(str(w)) for w in self.model.get("header_words", []) or []]
        self.combined_words: List[str] = self.model.get("combined_words") or list(dict.fromkeys(self.global_words + self.header_words))
        
        # mapping
        vt: Dict[str, str] = self.model.get("variant_to_field", {}) or {}
        self.variant_to_field: Dict[str, Any] = {self._normalize(str(k)): v for k, v in vt.items()}
        
        # params
        params: Dict[str, Any] = self.model.get("params", {}) or {}
        ngr = params.get("char_ngram_range", [2, 5]) or [2, 5]
        try:
            self.n_min, self.n_max = int(ngr[0]), int(ngr[1])
        except Exception:
            self.n_min, self.n_max = 2, 5
        if self.n_min < 1 or self.n_max < self.n_min:
            self.n_min, self.n_max = 2, 5
            
        self.threshold = float(params.get("threshold_similarity", 0.60))
        self.thresholds_by_len: Dict[Any, Any] = params.get("thresholds_by_len", {}) or {}
        
        # pesos por n-grama (inversos)
        wb = params.get("weights_by_n", {})
        self.weights_by_n: Dict[int, float] = {}
        if wb and isinstance(wb, dict):
            for n in range(self.n_min, self.n_max + 1):
                self.weights_by_n[n] = float(wb.get(str(n), 1.0))
        else:
            # pesos por defecto inversos
            for n in range(self.n_min, self.n_max + 1):
                if n == 2: self.weights_by_n[n] = 0.5
                elif n == 3: self.weights_by_n[n] = 0.7
                elif n == 4: self.weights_by_n[n] = 0.85
                elif n == 5: self.weights_by_n[n] = 1.0
                else: self.weights_by_n[n] = 1.0

        # índice precomputado desde el pickle
        grams_index = self.model.get("grams_index")
        self.grams: List[Dict[int, set[str]]] = []
        self.lengths: List[int] = []
        
        if grams_index:
            for entry in grams_index:
                length = int(entry.get("len", 0))
                gmap_raw: Dict[int, List[str]] = entry.get("grams", {}) or {}
                gmap_sets: Dict[int, set[str]] = {}
                for n in range(self.n_min, self.n_max + 1):
                    gmap_sets[n] = set(gmap_raw.get(n, []) or [])
                self.lengths.append(length)
                self.grams.append(gmap_sets)
        else:
            # fallback: construir en tiempo de carga
            self.grams = []
            self.lengths = []
            for w in self.combined_words:
                length = len(w)
                self.lengths.append(length)
                gmap_sets: Dict[int, set[str]] = {}
                for n in range(self.n_min, self.n_max + 1):
                    gmap_sets[n] = set(self._ngrams(w, n))
                self.grams.append(gmap_sets)

        # buckets por longitud
        buckets: Dict[int, List[int]] = self.model.get("buckets_by_len") or {}
        self.buckets_by_len: Dict[int, List[int]] = {int(k): list(v) for k, v in buckets.items()}
        if not self.buckets_by_len:
            for i, length in enumerate(self.lengths):
                self.buckets_by_len.setdefault(length, []).append(i)

    def _len_threshold(self, length: int) -> float:
        """Obtiene umbral por longitud del candidato"""
        if self.thresholds_by_len:
            int_keys = sorted([int(k) for k in self.thresholds_by_len.keys() if str(k).isdigit()])
            for k in int_keys:
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
            return max(self.threshold, 0.70)
        if length <= 10:
            return max(self.threshold, 0.65)
        return max(self.threshold, 0.60)

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
                
            inter = len(A.intersection(B))
            num += w * self._binary_cosine(len(A), len(B), inter)
            den += w
            
        if den <= 0.0:
            return 0.0
        return num / den

    def _build_query_grams(self, q: str) -> Dict[int, set[str]]:
        """Construye n-gramas de la consulta"""
        gq: Dict[int, set[str]] = {}
        for n in range(self.n_min, self.n_max + 1):
            gq[n] = set(self._ngrams(q, n))
        return gq

    def find_keywords(self, text: str | List[str]) -> List[Dict[str, Any]]:
        """Busca palabras clave usando coseno binario de n-gramas"""
        if not text:
            return []
            
        single = False
        if isinstance(text, str):
            text = [text]
            single = True

        results: List[Dict[str, Any]] = []
        
        for s in text:
            q = self._normalize(s)
            if not q:
                continue
                
            grams_q = self._build_query_grams(q)
            best_idx = None
            best_score = 0.0

            # Comparar contra TODOS los candidatos (sin filtrar por longitud)
            for i in range(len(self.combined_words)):
                score = self._score_binary_cosine_multi_n(self.grams[i], grams_q)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                cand = self.combined_words[best_idx]
                thr = self._len_threshold(len(cand))
                
                if best_score >= thr:
                    field = self.variant_to_field.get(cand)
                    results.append({
                        "field": field,
                        "word_found": cand,
                        "similarity": float(best_score),
                        "query": q
                    })

        return results if not single else (results[0:1] if results else [])

    def get_model_info(self):
        return {
            "total_words": len(self.combined_words),
            "vocabulario_size": None,
            "threshold_similarity": self.threshold,
            "global_words": len(self.global_words),
            "header_words": len(self.header_words),
            "combined_words": len(getattr(self, "combined_words", [])),
            "campos_disponibles": list(self.model.get("key_fields", {}).keys())
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
                "field": "RFCProveedor",
                "word_found": "rfc" if len(query) <= 5 else "registrofederaldecontribuyentes",
                "similarity": float(rfc_score),
                "query": query,
                "method": "regex_rfc"
            }
        
        # Probar fecha/hora
        fecha_score, fecha_type = self._regex_patterns_fecha(query)
        if fecha_score > 0:
            return {
                "field": "FechaDocumento",
                "word_found": fecha_type,
                "similarity": float(fecha_score),
                "query": query,
                "method": "regex_fecha"
            }
        
        return {}