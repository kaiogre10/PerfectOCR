# core/workers/vectorial_transformation/semantic_corrector.py
import re
from typing import Dict, Any
from core.domain.data_models import Polygons
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
import logging

logger = logging.getLogger(__name__)

class SemanticCorrector(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.project_root = project_root
        self.worker_config = config.get('semantic_corrector', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("math_max_corrected", False)

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        
        # Definición de categorías atómicas y su semántica esperada
        self.semantic_types = {
            "quantitative": ["0-9", r"[\d.,]+", r"\$"],
            "descriptivo": [r"[a-zA-Z]", r"[áéíóúñ]"],
            "code": [r"[A-Z]{2,}\d{2,}", r"RFC", r"ID", r"FOLIO"],
        }

        # Rango esperado de densidades promedio (proporción de dígitos/letras)
        self.expected_density = {
            "": (0.7, 1.0),      # alta densidad numérica
            "code": (0.3, 0.7),     # mezcla alfanumérica
            "descriptive": (0.0, 0.3),       # baja densidad numérica
        }

    def atomic_classify(self, word):
        word_norm = self.normalize(word)
        if not word_norm:
            return "vacío"

        digits = len(re.findall(r"\d", word_norm))
        letters = len(re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúüÜ]", word_norm))
        symbols = len(re.findall(r"[$%.,\-]", word_norm))
        total = len(word_norm)

        # proporciones
        p_digits = digits / total
        p_letters = letters / total
        p_symbols = symbols / total

        # Reglas deterministas
        if p_digits > 0.7:
            return "cuantitativo"
        elif 0.3 < p_digits < 0.7 and p_letters > 0.2:
            return "identificador"
        elif p_letters > 0.5:
            return "descriptivo"
        else:
            return "mixto"

    def analyze_column(self, column_values):
        atomic_types = [self.atomic_classify(v) for v in column_values if v.strip()]
        if not atomic_types:
            return "vacía"

        # contar frecuencia de tipos
        freq = {t: atomic_types.count(t) for t in set(atomic_types)}
        dominant = max(freq, key=freq.get)
        return dominant

    def correct_column(self, column_values):
        """
        Ajusta los valores erróneamente clasificados dentro de una columna
        según la coherencia del grupo.
        """
        dominant = self.analyze_column(column_values)
        corrected = []
        for val in column_values:
            t = self.atomic_classify(val)
            if t != dominant:
                # Validar si el valor es ruido o un error de lectura OCR
                val_norm = self.normalize(val)
                if len(val_norm) <= 1:
                    corrected.append("")  # ruido
                    continue

                # Si el valor tiene un 30% de coincidencia con el dominante → corregirlo
                val_digits = len(re.findall(r"\d", val_norm))
                ratio = val_digits / len(val_norm)
                exp_low, exp_high = self.expected_density.get(dominant, (0, 1))

                if exp_low <= ratio <= exp_high:
                    corrected.append(val_norm)
                else:
                    corrected.append("")  # inconsistente
            else:
                corrected.append(self.normalize(val))
        return corrected, dominant

    # ==========================================================
    # ETAPA 5: PROCESAMIENTO GLOBAL
    # ==========================================================
    def correct_table(self, table_rows):
        """
        table_rows: lista de listas (output del geo_matrix)
        Retorna una tabla corregida y las etiquetas semánticas inferidas.
        """
        corrected = []
        sem_labels = []

        n_cols = len(table_rows[0])
        for col_idx in range(n_cols):
            column = [row[col_idx] for row in table_rows]
            corrected_col, label = self.correct_column(column)
            sem_labels.append(label)

            # reintegra columna corregida a filas
            if not corrected:
                corrected = [[v] for v in corrected_col]
            else:
                for i, v in enumerate(corrected_col):
                    corrected[i].append(v)

        return corrected, sem_labels
