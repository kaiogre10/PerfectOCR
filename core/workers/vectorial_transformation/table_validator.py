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
            if manager.merge_semantics():
                logger.info("Clasificaciones semánticas fusionadas listas")

                # Obtener la copia del contexto
                df = context.get("table_copy")
                if df is None or df.empty:
                    logger.error("No hay tabla estructurada para procesar")
                    return False
                    
                logger.debug("Tabla recibida para validación estructural:\n" + df.to_string(index=False))

                validated_df = self.correct_table(df, manager)
                
                # Guardar resultado en contexto si es necesario
                context["validated_table"] = validated_df

        except Exception as e:
            logger.warning(f"Error en el postproceamiento tabular: {e}", exc_info=True)
            return True

    def correct_table(self, df: pd.DataFrame, manager: DataFormatter) -> pd.DataFrame:
        """
        Retorna una tabla corregida y las etiquetas semánticas inferidas.
        """
        try:
            # Diccionario de valores semánticos
            self.semantic_values: Dict[str, float] = {
                'numeric': 1.0,
                'descriptive': -1.0,
                'code': 0.0
            }

            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            polygon_semantic_scores: Dict[str, float] = {}

            for poly_id, polygon in polygons.items():
                value = None
                # Si la clasificación ya fue fusionada y es string
                if isinstance(polygon.semantic_clasification, str):
                    value = self.semantic_values.get(polygon.semantic_clasification, 0.0)
                # Si es objeto con atributos booleanos
                elif polygon.semantic_clasification:
                    if getattr(polygon.semantic_clasification, "numeric", False):
                        value = self.semantic_values["numeric"]
                    elif getattr(polygon.semantic_clasification, "descriptive", False):
                        value = self.semantic_values["descriptive"]
                    elif getattr(polygon.semantic_clasification, "code", False):
                        value = self.semantic_values["code"]
                    else:
                        value = 0.0
                else:
                    value = 0.0

                polygon_semantic_scores[poly_id] = value
                logger.info(f"Polígono {poly_id}: valor = {value}")

            # Crear tabla con valores semánticos
            table_with_values = self._create_table_with_semantic_values(df, polygon_semantic_scores, polygons)
            logger.info(f"avalores: {table_with_values}")
            
            return df

        except Exception as e:
            logger.error(f"Error en correct_table: {e}", exc_info=True)
            return df

    def _create_table_with_semantic_values(self, df: pd.DataFrame, polygon_semantic_scores: Dict[str, float], 
                                        polygons: Dict[str, Polygons]) -> pd.DataFrame:
        """
        Crea una tabla donde cada celda tiene el valor semántico del polígono correspondiente.
        """
        try:
            # Crear diccionario texto -> polygon_id para búsqueda rápida
            text_to_polygon = {}
            for poly_id, polygon in polygons.items():
                text = polygon.ocr_text or ""
                if text.strip():
                    text_to_polygon[text.strip()] = poly_id
            
            # Crear nueva tabla con valores semánticos
            table_with_values = df.copy()
            
            for row_idx, (_, row) in enumerate(table_with_values.iterrows()):
                for col_idx, col_name in enumerate(table_with_values.columns):
                    cell_text = str(row[col_name]).strip()
                    
                    # Buscar polígono que coincida con este texto
                    matching_polygon_id = None
                    for text, poly_id in text_to_polygon.items():
                        if text in cell_text or cell_text in text:
                            matching_polygon_id = poly_id
                            break
                    
                    # Reemplazar texto con valor semántico
                    if matching_polygon_id and matching_polygon_id in polygon_semantic_scores:
                        semantic_value = polygon_semantic_scores[matching_polygon_id]
                        table_with_values.iloc[row_idx, col_idx] = semantic_value
                    else:
                        table_with_values.iloc[row_idx, col_idx] = 0.0  # default
            
            return table_with_values
            
        except Exception as e:
            logger.error(f"Error creando tabla con valores semánticos: {e}", exc_info=True)
            return df