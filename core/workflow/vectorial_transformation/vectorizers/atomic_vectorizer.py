# PerfectOCR/core/vectorization/vectors/atomic_vectorizer.py

import json
import os
import math
from typing import List, Dict, Any, Optional
from core.workspace.domain.vector_models import AtomicVector

class AtomicVectorizer:

    def __init__(self, density_map: Dict[str, int]):
        """
        Inicializa el vectorizador atómico con el mapa de densidad.
        """
        self.density_map = density_map

    def get_density(self, char: str) -> int:
        """
        Obtiene la densidad de un carácter desde el mapa de densidad.
        """
        return self.density_map.get(char, 116)  # 116 por defecto para caracteres no mapeados

    def _get_centroide_x(self, polygon: Dict[str, Any]) -> float:
        """
        Calcula el centroide X de un polígono.
        
        Args:
            polygon: Diccionario con las coordenadas del polígono
            
        Returns:
            Coordenada X del centroide
        """
        # Extraer coordenadas del polígono
        bbox = polygon.get('bbox', [])
        if len(bbox) >= 4:
            # bbox = [x1, y1, x2, y2] - calcular centro
            return (bbox[0] + bbox[2]) / 2.0
        
        # Fallback: usar coordenadas individuales si están disponibles
        coords = polygon.get('coords', [])
        if coords and len(coords) >= 2:
            return coords[0]  # Primera coordenada X
        
        return 0.0  # Fallback final

    def get_density_list(self, text: str) -> List[int]:
        """
        Genera una lista de densidades para cada carácter en el texto.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de densidades correspondientes a cada carácter
        """
        return [self.get_density(char) for char in text]

    def generate_density_sequences(self, grouped_lines: List[List[Dict[str, Any]]]) -> List[List[int]]:
        """
        Genera secuencias de densidades z_i para cada línea, para detección tabular.
        
        Args:
            grouped_lines: Lista de líneas, cada línea es una lista de polígonos
            
        Returns:
            Lista de líneas, cada línea es una lista de densidades z_i ordenadas
        """
        secuencias = []
        for linea in grouped_lines:
            # Ordenar polígonos por posición X
            poligonos_ordenados = sorted(linea, key=lambda p: self._get_centroide_x(p))
            texto_linea = "".join([poligono.get('text', '') for poligono in poligonos_ordenados])
            # Generar secuencia de densidades
            densidades = self.get_density_list(texto_linea)
            secuencias.append(densidades)

        return secuencias

    def generate_atomic_vectors_for_line(self, line: List[Dict[str, Any]]) -> List[AtomicVector]:
        """
        Genera vectores atómicos para una línea completa.
        
        Args:
            line: Lista de polígonos que forman una línea
            
        Returns:
            Lista de AtomicVector con z_i y x_ri para cada carácter
        """
        # Ordenar polígonos por posición X
        poligonos_ordenados = sorted(line, key=lambda p: self._get_centroide_x(p))
        
        # Concatenar todo el texto de la línea
        texto_linea = "".join([poligono.get('text', '') for poligono in poligonos_ordenados])
        
        # Generar vectores atómicos
        atomic_vectors = []
        for i, char in enumerate(texto_linea):
            atomic_vector = AtomicVector(
                z_i=self.get_density(char),
                x_ri=i + 1  # Posición horizontal relativa (empezando en 1)
            )
            atomic_vectors.append(atomic_vector)
        
        return atomic_vectors