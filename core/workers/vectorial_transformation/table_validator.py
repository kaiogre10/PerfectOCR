# core/workers/vectorial_transformation/semantic_corrector.py
from typing import Dict, Any, Tuple, List
import pandas as pd # type: ignore
import time
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
import logging

logger = logging.getLogger(__name__)

class TableCorrector(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('semantic_corrector', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("math_max_corrected", False)

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        try:
            # start_time = time.time()
            if manager.merge_semantics():
                logger.info("Clasificaciones semánticas fusionadas listas")

                df = manager.get_structured_table()
                if df is None or df.empty:
                    logger.error("No hay tabla estructurada para procesar")
                    return False
                # Log simple de cómo recibe la tabla (antes de corregir)
                logger.debug("Tabla recibida para valiación estructural:\n" + df.to_string(index=False))

                validated_df = self.correct_table(df, manager)

        except Exception as e:
            logger.warning(f"Error en el postproceamiento tabular: {e}", exc_info=True)
            return True

    def correct_table(self, df: pd.DataFrame, manager: DataFormatter):
        """
        Retorna una tabla corregida y las etiquetas semánticas inferidas.
        """
        try:
            # Diccionario de valores semánticos
            semantic_values: Dict[str, float] = {
                'numeric': 1.0,
                'descriptive': -1.0,
                'code': 0.0
            }

            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            polygon_semantic_scores = {}

            for poly_id, polygon in polygons.items():
                value = None
                # Si la clasificación ya fue fusionada y es string
                if isinstance(polygon.semantic_clasification, str):
                    value = semantic_values.get(polygon.semantic_clasification, 0.0)
                # Si es objeto con atributos booleanos
                elif polygon.semantic_clasification:
                    if getattr(polygon.semantic_clasification, "numeric", False):
                        value = semantic_values["numeric"]
                    elif getattr(polygon.semantic_clasification, "descriptive", False):
                        value = semantic_values["descriptive"]
                    elif getattr(polygon.semantic_clasification, "code", False):
                        value = semantic_values["code"]
                    else:
                        value = 0.0
                else:
                    value = 0.0

                polygon_semantic_scores[poly_id] = value
                logger.info(f"Polígono {poly_id}: valor = {value}")

            # Aquí puedes continuar con la lógica de corrección usando polygon_semantic_scores
            # Por ejemplo, podrías devolver la tabla y las etiquetas inferidas:
            # return df, polygon_semantic_scores

            return df, polygon_semantic_scores

        except Exception as e:
            logger.error(f"Error en correct_table: {e}", exc_info=True)
            return df, {}
            

    #             logger.debug("Tabla recibida para corrección matemática:\n" + df.to_string(index=False))

    
    #         # return corrected, sem_labels
    #     columns: List[str] = list(df.columns)

    #     corrected = []
    #     sem_labels = []

    #     n_cols = len(table_rows[0])
    #     for col_idx in range(n_cols):
    #         column = [row[col_idx] for row in table_rows]
    #         corrected_col, label = self.correct_column(column)
    #         sem_labels.append(label)

    #         # reintegra columna corregida a filas
    #         if not corrected:
    #             corrected = [[v] for v in corrected_col]
    #         else:
    #             for i, v in enumerate(corrected_col):
    #                 corrected[i].append(v)

    #     return corrected, sem_labels


    # def analyze_column(self, column_values):
    #     atomic_types = [self.atomic_classify(v) for v in column_values if v.strip()]
    #     if not atomic_types:
    #         return "vacía"

    #     # contar frecuencia de tipos
    #     freq = {t: atomic_types.count(t) for t in set(atomic_types)}
    #     dominant = max(freq, key=freq.get)
    #     return dominant

    # def correct_column(self, column_values):
    #     """
    #     Ajusta los valores erróneamente clasificados dentro de una columna
    #     según la coherencia del grupo.
    #     """
    #     dominant = self.analyze_column(column_values)
    #     corrected = []
    #     for val in column_values:
    #         t = self.atomic_classify(val)
    #         if t != dominant:
    #             # Validar si el valor es ruido o un error de lectura OCR
    #             val_norm = self.normalize(val)
    #             if len(val_norm) <= 1:
    #                 corrected.append("")  # ruido
    #                 continue

    #             # Si el valor tiene un 30% de coincidencia con el dominante → corregirlo
    #             val_digits = len(re.findall(r"\d", val_norm))
    #             ratio = val_digits / len(val_norm)
    #             exp_low, exp_high = self.expected_density.get(dominant, (0, 1))

    #             if exp_low <= ratio <= exp_high:
    #                 corrected.append(val_norm)
    #             else:
    #                 corrected.append("")  # inconsistente
    #         else:
    #             corrected.append(self.normalize(val))
    #     return corrected, dominant

