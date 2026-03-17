# core/workers/vectorial_transformation/table_validator.py
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd # type: ignore
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
import logging

logger = logging.getLogger(__name__)

class TableCorrector(VectorizationAbstractWorker):
    """
    Corrige la estructura de la tabla aislando cuantitativos (SC=2).
    Regla fundamental: Solo los cuantitativos deben ir aislados en su propia celda.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('semantic_corrector', {})
        self.enabled_outputs = self.config.get("vectorize", {})
        self.output = self.enabled_outputs.get("table_validator_corrected", False)
        
        # SC=2 es cuantitativo, el único que debe ir aislado
        self.quantitative_sc = 2

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        try:
            df = manager.get_structured_table()
            if df is None or df.empty:
                logger.warning("No hay tabla estructurada para validar")
                return False
                
            logger.debug("Tabla recibida para validación estructural:\n" + df.to_string(index=False))

            corrected_df = self._isolate_quantitatives(df, manager)
            
            logger.info("Tabla tras aislamiento de cuantitativos:\n" + corrected_df.to_string(index=False))
            
            # Guardar resultado
            context["validated_table"] = corrected_df
            manager.save_structured_table(df=corrected_df, columns=list(corrected_df.columns))
            
            return True

        except Exception as e:
            logger.warning(f"Error en el postprocesamiento tabular: {e}", exc_info=True)
            return False

    def _isolate_quantitatives(self, df: pd.DataFrame, manager: DataFormatter) -> pd.DataFrame:
        """
        Aísla los cuantitativos (SC=2) en celdas individuales.
        Si hay cuantitativos mezclados con otros tokens, los extrae y 
        los mueve a celdas vacías disponibles.
        """
        try:
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.warning("No hay polígonos disponibles para validación")
                return df
            
            # Crear mapeo token -> (poly_id, sc_list)
            token_to_semantic = self._build_token_semantic_map(polygons)
            
            corrected_df = df.copy()
            num_cols = len(corrected_df.columns)
            
            for row_idx in range(len(corrected_df)):
                row_data = corrected_df.iloc[row_idx].tolist()
                corrected_row = self._process_row(row_data, num_cols, token_to_semantic)
                
                for col_idx, value in enumerate(corrected_row):
                    corrected_df.iloc[row_idx, col_idx] = value
            
            return corrected_df

        except Exception as e:
            logger.error(f"Error en _isolate_quantitatives: {e}", exc_info=True)
            return df

    def _build_token_semantic_map(self, polygons: Dict[str, Polygons]) -> Dict[str, Tuple[str, List[int]]]:
        """
        Construye un mapeo de token -> (poly_id, semantic_classification).
        Para polígonos con múltiples tokens, mapea cada token individualmente.
        """
        token_map: Dict[str, Tuple[str, List[int]]] = {}
        
        for poly_id, polygon in polygons.items():
            text = polygon.ocr_text or ""
            sc_list = polygon.semantic_clasification or [0]
            
            if not text.strip():
                continue
            
            tokens = text.strip().split()
            
            # Si hay igual número de tokens que clasificaciones, mapear 1:1
            if len(tokens) == len(sc_list):
                for i, token in enumerate(tokens):
                    token_clean = token.strip()
                    if token_clean:
                        token_map[token_clean] = (poly_id, [sc_list[i]])
            else:
                # Mapear el texto completo con toda la clasificación
                token_map[text.strip()] = (poly_id, sc_list)
                # También mapear tokens individuales con la clasificación promedio/dominante
                for token in tokens:
                    token_clean = token.strip()
                    if token_clean and token_clean not in token_map:
                        token_map[token_clean] = (poly_id, sc_list)
        
        return token_map

    def _process_row(self, row_data: List[str], num_cols: int, token_to_semantic: Dict[str, Tuple[str, List[int]]]) -> List[str]:
        """
        Procesa una fila para aislar cuantitativos.
        1. Identifica tokens cuantitativos mezclados con otros
        2. Los extrae y busca celdas vacías para colocarlos
        """
        # Analizar cada celda y separar tokens
        cells_analysis: List[Dict[str, Any]] = []
        
        for col_idx, cell_value in enumerate(row_data):
            cell_str = str(cell_value).strip() if cell_value else ""
            tokens = cell_str.split() if cell_str else []
            
            quantitatives: List[str] = []
            non_quantitatives: List[str] = []
            
            for token in tokens:
                token_clean = token.strip()
                if not token_clean:
                    continue
                    
                sc = self._get_token_sc(token_clean, token_to_semantic)
                
                if sc == self.quantitative_sc:
                    quantitatives.append(token_clean)
                else:
                    non_quantitatives.append(token_clean)
            
            cells_analysis.append({
                'col_idx': col_idx,
                'original': cell_str,
                'quantitatives': quantitatives,
                'non_quantitatives': non_quantitatives,
                'is_empty': len(tokens) == 0,
                'needs_isolation': len(quantitatives) > 0 and len(non_quantitatives) > 0
            })
        
        # Construir fila corregida
        result_row: List[str] = [""] * num_cols
        pending_quantitatives: List[str] = []
        
        # Primera pasada: colocar no-cuantitativos y recolectar cuantitativos que necesitan aislarse
        for cell in cells_analysis:
            col_idx = cell['col_idx']
            
            if cell['needs_isolation']:
                # Celda mixta: dejar no-cuantitativos, extraer cuantitativos
                result_row[col_idx] = " ".join(cell['non_quantitatives'])
                pending_quantitatives.extend(cell['quantitatives'])
            elif len(cell['quantitatives']) > 0 and len(cell['non_quantitatives']) == 0:
                # Celda solo con cuantitativos: si hay más de uno, quedarse con el primero y mover el resto
                if len(cell['quantitatives']) == 1:
                    result_row[col_idx] = cell['quantitatives'][0]
                else:
                    result_row[col_idx] = cell['quantitatives'][0]
                    pending_quantitatives.extend(cell['quantitatives'][1:])
            else:
                # Celda normal (solo no-cuantitativos o vacía)
                result_row[col_idx] = cell['original']
        
        # Segunda pasada: colocar cuantitativos pendientes en celdas vacías (izquierda a derecha, preservando orden)
        if pending_quantitatives:
            empty_cols = [i for i in range(num_cols) if result_row[i] == ""]
            
            for quant in pending_quantitatives:
                if empty_cols:
                    target_col = empty_cols.pop(0)
                    result_row[target_col] = quant
                else:
                    logger.warning(f"No hay celdas vacías para cuantitativo: '{quant}'")
        
        return result_row

    def _get_token_sc(self, token: str, token_to_semantic: Dict[str, Tuple[str, List[int]]]) -> int:
        """
        Obtiene la clasificación semántica de un token.
        Si el token tiene múltiples SC, retorna el dominante.
        SC=2 (cuantitativo) tiene prioridad si está presente.
        """
        if token in token_to_semantic:
            _, sc_list = token_to_semantic[token]
            if self.quantitative_sc in sc_list:
                return self.quantitative_sc
            # Retornar el más común o el primero
            return sc_list[0] if sc_list else 0
        
        # Heurística: si parece número con decimales, probablemente es cuantitativo
        if self._looks_like_quantitative(token):
            return self.quantitative_sc
        
        return 0  # default: descriptivo

    def _looks_like_quantitative(self, token: str) -> bool:
        """
        Heurística conservadora: solo detecta precios/montos explícitos.
        No clasifica enteros sueltos para evitar falsos positivos (ej: "25" en "VINCI 25 AM").
        """
        try:
            cleaned = token.replace(",", "").strip()
            if not cleaned:
                return False
            
            has_currency = "$" in cleaned
            cleaned_num = cleaned.replace("$", "").replace("%", "").strip()
            
            if not cleaned_num:
                return False
            
            # Precio explícito con signo de moneda
            if has_currency:
                float(cleaned_num)
                return True
            
            # Decimal sin moneda (ej: 85.50, 9.49)
            if "." in cleaned_num:
                float(cleaned_num)
                return True
                
            return False
        except ValueError:
            return False