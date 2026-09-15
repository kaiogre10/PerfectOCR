# domain/data_models.py
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
    
@dataclass(slots=True)
class Polygons:
    polygon_id: str
    poly_index: int
    bounding_box: List[int]
    centroid: List[float]
    cropped_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
    ocr_text: Optional[str]
    key_field: Optional[List[int]]
    semantic_clasification: List[int]
    cuant_chars: int

@dataclass(slots=True)
class AllLines:
    lineal_id: str
    line_index: int
    text: str
    polygon_ids: List[str]
    polygons_index: List[int]
    line_centroid: List[float]
    line_bbox: List[float]
    tabular_line: bool
    header_line: Optional[int]
    footer_line: Optional[int]
    
@dataclass(slots=True)
class Metadata:
    image_name: str
    img_dims: Tuple[int, int]

@dataclass(slots=True)
class StructuredData:
    df_table: Optional[pd.DataFrame]
    global_data: Dict[str, Any]

@dataclass(slots=True)
class WorkflowData:
    metadata: Optional[Metadata]
    polygons: Optional[Dict[str, Polygons]]
    all_lines: Optional[Dict[str, AllLines]]
    table_data: Optional[StructuredData]

@dataclass(slots=True)
class Payload:
    payload: Optional[str]
    buff_size: int

class FullImageKey:
    img_ptr: Optional[int] = None  

    def __new__(cls, img_ptr: int):
        # Bloquea floats (1.0), strings ("1"), y explícitamente booleanos (True/False).
        if type(img_ptr) is not int:
            raise TypeError(f"Tipo inválido: se esperaba 'int', se recibió '{type(img_ptr).__name__}'.")
            
        # 2. VALIDACIÓN DE NEGATIVOS: Un puntero de memoria real no puede ser menor a cero.
        if img_ptr < 0:
            raise TypeError("Validación fallida: El puntero no puede ser un entero negativo.")

        # Guarda el entero en el namespace de la clase
        cls.img_ptr = img_ptr
        
        # Evita crear instancias en memoria
        return None 

    @classmethod
    def clear(cls) -> None:
        """Resetea el puntero a None. Cero residuos entre tareas de la cola."""
        cls.img_ptr = None