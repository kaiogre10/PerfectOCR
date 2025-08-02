# PerfectOCR/core/workspace/domain/vector_models.py
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class MorphologicalProfile:
    """Contiene el perfil morfológico de un polígono, que es una cadena
    basada en la clasificación de sus caracteres (a, n, e).
    Su identificador relacional es la coordenada del centroide"""
    x: float
    y: float
    perfil_str: str

@dataclass
class ElementalVector:
    """Representa un polígono con todos sus escalares geométricos y morfológicos."""
    # Geometría
    x: float
    y: float
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    # Atributos morfológicos calculados
    confianza_ocr: float
    l_c: int  # Cantidad de caracteres
    l_a: int  # Cantidad de caracteres alfabéticos
    l_n: int  # Cantidad de caracteres numéricos
    l_e: int  # Cantidad de caracteres especiales
    r_a: float # Ratio de alfabéticos
    r_n: float # Ratio de numéricos
    r_e: float # Ratio de especiales
    z_m: float # Densidad media del polígono
    
    # Atributos relacionales (se calculan durante el agrupamiento en líneas) 
    k: int = 0             # Identificador de la línea (S_k) a la que pertenece, indice de línea
    h_x: float = 0.0       # Distancia horizontal con el polígono siguiente a la derecha con respecto de x_min x _max
    d_arctan: float = 0.0  # Derivada del arcotangente
    d_arccot: float = 0.0  # Derivada del arcocotangente
    o_x: float = 0.0       # Distancia ortogonal al eje x respecto de y_min

@dataclass
class DifferetiatorVector:
    """Contiene las estadísticas agregadas de la geometría de una línea completa (S_k)."""
    y_sk_min: float      # Promedio de los y_min de todos los polígonos de la línea 
    y_sk_max: float      # Promedio de los y_max de todos los polígonos de la línea 
    x_sk_min: float      # El valor x_min más a la izquierda de la línea 
    x_sk_max: float      # El valor x_max más a la derecha de la línea
    o_xm: float          # Distancia ortogonal al promedio al eje x
    o_sk: float          # Distancia vertical promedio con la línea siguiente (S_k+1) 

@dataclass
class AtomicVector:
    """Es el vector de la forma más básica de unidad de un polígono."""
    z_i: int             # La densidad dada en el density_map 
    x_ri: int            # Posición horizontal relativa progresiva de cada letra
