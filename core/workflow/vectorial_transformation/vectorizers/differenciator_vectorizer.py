# PerfectOCR/core/workflow/vectorial_transformation/vectorizers/differenciator_vectorizer.py

import os
import math
import json
from typing import List, Tuple, Dict, Any, Optional
from core.workspace.domain.vector_models import DifferetiatorVector
from core.workflow.vectorial_transformation.vectorizers.elemental_vectorizer import ElementalVectorizer

class DifferenciatorVectorizer:

    def __init__(self, density_map: Dict[str, Any]):
        """
        Inicializa el vectorizador diferenciador con el mapa de densidad.
        
        Args:
            density_map: Diccionario con el mapa de densidad de caracteres
        """
        self.density_map = density_map
        self.elemental_vectorizer = ElementalVectorizer(density_map)

    def generate_differential_scalars(self, linea_actual: List[Dict[str, Any]], 
                                   linea_siguiente: Optional[List[Dict[str, Any]]] = None) -> DifferetiatorVector:
        """
        Genera los escalares para el vector diferenciador usando las definiciones exactas.
        
        Returns:
            DifferetiatorVector con todos los campos requeridos
        """
        diff = self.elemental_vectorizer.calculate_differential_scalars(linea_actual, linea_siguiente)
        
        return DifferetiatorVector(
            y_sk_min=diff['y_sk_min'], 
            y_sk_max=diff['y_sk_max'],
            x_sk_min=diff['x_sk_min'], 
            x_sk_max=diff['x_sk_max'],
            o_xm=diff['o_xm'],  # Ahora incluido correctamente
            o_sk=diff['o_sk']
        )