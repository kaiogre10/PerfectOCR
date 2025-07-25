# PerfectOCR/core/workflow/vectorial_transformation/tensorizer.py
import json
import os
import math
from typing import List, Tuple, Dict, Any, Optional
from core.workflow.vectorial_transformation.vectorizers.atomic_vectorizer import AtomicVectorizer
from core.workflow.vectorial_transformation.vectorizers.differenciator_vectorizer import DifferenciatorVectorizer
from core.workflow.vectorial_transformation.vectorizers.elemental_vectorizer import ElementalVectorizer
from core.workspace.domain.vector_models import AtomicVector, ElementalVector, DifferetiatorVector, MorphologicalProfile

class VectorTensorizer:

    def __init__(self, config: Dict, project_root: str):
        self.config = config
        self.project_root = project_root
        self.density_map = config.get('density_map', {})
        self._atomic_vectorizer = AtomicVectorizer(self.density_map)
        self._differenciator_vectorizer = DifferenciatorVectorizer(self.density_map)
        self._elemental_vectorizer = ElementalVectorizer(self.density_map)

    def generate_density_sequences(self, grouped_lines: List[List[Dict[str, Any]]]) -> List[List[int]]:
        """
        Delega la generación de secuencias de densidad al AtomicVectorizer.
        """
        return self._atomic_vectorizer.generate_density_sequences(grouped_lines)

    def generate_elemental_vectors_for_line(self, line: List[Dict[str, Any]], k: int) -> List[ElementalVector]:
        """
        Genera vectores elementales para todos los polígonos de una línea.
        """
        return self._elemental_vectorizer.generate_elemental_vectors_for_line(line, k)

    def generate_atomic_vectors_for_line(self, line: List[Dict[str, Any]]) -> List[AtomicVector]:
        """
        Genera vectores atómicos para una línea.
        """
        return self._atomic_vectorizer.generate_atomic_vectors_for_line(line)

    def generate_morphological_profiles_for_line(self, line: List[Dict[str, Any]]) -> List[MorphologicalProfile]:
        """
        Genera perfiles morfológicos para todos los polígonos de una línea.
        """
        return [self._elemental_vectorizer.generate_morphological_profile(poligono) for poligono in line]

    def generate_differential_vector_for_line(self, current_line: List[Dict[str, Any]], 
                                            next_line: Optional[List[Dict[str, Any]]] = None) -> DifferetiatorVector:
        """
        Genera el vector diferenciador para una línea.
        """
        return self._differenciator_vectorizer.generate_differential_scalars(current_line, next_line)
