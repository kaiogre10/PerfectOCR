# PerfectOCR/core/workflow/vectorial_transformation/vectorizers/elemental_vectorizer.py
import json
import os
import math
from typing import List, Tuple, Dict, Any, Optional
from core.workspace.domain.vector_models import ElementalVector, MorphologicalProfile

class ElementalVectorizer:

    def __init__(self, density_map: Dict[str, int]):
        """Inicializa el vectorizador elemental con el mapa de densidad."""
        self.density_map = density_map

    def get_density(self, char: str) -> int:
        """Obtiene la densidad de un carácter desde el mapa de densidad."""
        return self.density_map.get(char, 0)

    def get_char_type(self, char: str) -> str:
        """Devuelve 'n' si es numérico, 'e' si es especial, 'a' si es alfabético.
        Según los rangos de densidad definidos en la documentación."""
        z = self.get_density(char)
        if 0 <= z <= 9:
            return 'n'  # Números
        elif 10 <= z <= 55:
            return 'e'  # Especiales
        elif 56 <= z <= 116:
            return 'a'  # Alfabéticos
        else:
            return 'a'  # Por defecto, alfabético

    def calculate_geometric_scalars(self, poligono: Dict[str, Any]) -> Dict[str, float]:
        """Calcula todos los escalares geométricos de un polígono."""
        vertices = poligono.get('polygon_coords', [])
        if len(vertices) != 4:
            return {
                'x': 0.0, 'y': 0.0,
                'x_min': 0.0, 'x_max': 0.0, 'y_min': 0.0, 'y_max': 0.0,
                'x1': 0.0, 'y1': 0.0, 'x2': 0.0, 'y2': 0.0,
                'x3': 0.0, 'y3': 0.0, 'x4': 0.0, 'y4': 0.0
            }
        
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        x4, y4 = vertices[3]

        # Calcular centroide
        x = (x1 + x2 + x3 + x4) / 4
        y = (y1 + y2 + y3 + y4) / 4

        # Calcular extremos
        xs = [x1, x2, x3, x4]
        ys = [y1, y2, y3, y4]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        return {
            'x': x, 'y': y,
            'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4
        }

    def calculate_morphological_scalars(self, poligono: Dict[str, Any]) -> Dict[str, float]:
        """Calcula todos los escalares morfológicos de un polígono."""
        texto = poligono.get('text', '')
        confianza_ocr = float(poligono.get('overall_confidence_avg_lines', 0.0))

        # Contadores de caracteres
        l_c = len(texto)
        l_a = l_n = l_e = 0
        z_total = 0
        
        for c in texto:
            tipo = self.get_char_type(c)
            if tipo == 'a':
                l_a += 1
            elif tipo == 'n':
                l_n += 1
            else:
                l_e += 1
            z_total += self.get_density(c)
        
        # Calcular ratios
        r_a = l_a / l_c if l_c else 0.0
        r_n = l_n / l_c if l_c else 0.0
        r_e = l_e / l_c if l_c else 0.0
        z_m = z_total / l_c if l_c else 0.0

        return {
            'l_c': l_c, 'l_a': l_a, 'l_n': l_n, 'l_e': l_e,
            'r_a': r_a, 'r_n': r_n, 'r_e': r_e, 'z_m': z_m,
            'confianza_ocr': confianza_ocr
        }

    def calculate_relational_scalars(self, poligonos_linea: List[Dict[str, Any]], k: int) -> List[Dict[str, float]]:
        """Calcula escalares relacionales para todos los polígonos de una línea."""
        resultados = []
        
        for i, poligono in enumerate(poligonos_linea):
            # Calcular h_x (distancia horizontal con el siguiente polígono)
            h_x = 0.0
            if i < len(poligonos_linea) - 1:
                geo_actual = self.calculate_geometric_scalars(poligono)
                geo_siguiente = self.calculate_geometric_scalars(poligonos_linea[i + 1])
                h_x = geo_siguiente['x_min'] - geo_actual['x_max']
            
            # Calcular escalares angulares (simplificado por ahora)
            geo = self.calculate_geometric_scalars(poligono)
            d_arctan = math.atan(geo['y'] / geo['x']) if geo['x'] != 0 else 0.0
            d_arccot = math.atan(geo['x'] / geo['y']) if geo['y'] != 0 else 0.0
            o_x = geo['y_min']  # Distancia ortogonal al eje x
            
            resultados.append({
                'k': k,
                'h_x': h_x,
                'd_arctan': d_arctan,
                'd_arccot': d_arccot,
                'o_x': o_x
            })
        
        return resultados

    def generate_elemental_vector(self, poligono: Dict[str, Any], k: int = 0, 
                                relational_scalars: Optional[Dict[str, float]] = None) -> ElementalVector:
        """Genera un vector elemental completo para un polígono."""
        geo = self.calculate_geometric_scalars(poligono)
        morph = self.calculate_morphological_scalars(poligono)
        
        # Usar escalares relacionales si se proporcionan, sino valores por defecto
        rel = relational_scalars or {
            'k': k, 'h_x': 0.0, 'd_arctan': 0.0, 'd_arccot': 0.0, 'o_x': 0.0
        }

        return ElementalVector(
            x=geo['x'], y=geo['y'],
            x1=geo['x1'], y1=geo['y1'], x2=geo['x2'], y2=geo['y2'],
            x3=geo['x3'], y3=geo['y3'], x4=geo['x4'], y4=geo['y4'],
            x_min=geo['x_min'], x_max=geo['x_max'],
            y_min=geo['y_min'], y_max=geo['y_max'],
            confianza_ocr=morph['confianza_ocr'],
            l_c=morph['l_c'], l_a=morph['l_a'], l_n=morph['l_n'], l_e=morph['l_e'],
            r_a=morph['r_a'], r_n=morph['r_n'], r_e=morph['r_e'], z_m=morph['z_m'],
            k=rel['k'], h_x=rel['h_x'], d_arctan=rel['d_arctan'],
            d_arccot=rel['d_arccot'], o_x=rel['o_x']
        )

    def generate_elemental_vectors_for_line(self, line: List[Dict[str, Any]], k: int) -> List[ElementalVector]:
        """Genera vectores elementales para todos los polígonos de una línea."""
        vectors = []
        relational_scalars_list = self.calculate_relational_scalars(line, k)
        
        for i, poligono in enumerate(line):
            relational_scalars = relational_scalars_list[i]
            vector = self.generate_elemental_vector(poligono, k, relational_scalars)
            vectors.append(vector)
        
        return vectors

    def generate_morphological_profile(self, poligono: Dict[str, Any]) -> MorphologicalProfile:
        texto = poligono.get('text', '')
        geo = self.calculate_geometric_scalars(poligono)
        perfil_chars = []
        for char in texto:
            perfil_chars.append(self.get_char_type(char))
        
        perfil_str = ''.join(perfil_chars)
        
        return MorphologicalProfile(
            x=geo['x'],
            y=geo['y'],
            perfil_str=perfil_str
        )

    def calculate_differential_scalars(self, linea_actual: List[Dict[str, Any]], 
                                     linea_siguiente: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Calcula escalares diferenciales para una línea."""
        if not linea_actual:
            return {
                'y_sk_min': 0.0, 'y_sk_max': 0.0,
                'x_sk_min': 0.0, 'x_sk_max': 0.0, 
                'o_xm': 0.0, 'o_sk': 0.0
            }
        
        # Calcular escalares geométricos para todos los polígonos de la línea
        geo_scalars = [self.calculate_geometric_scalars(p) for p in linea_actual]
        
        # Promedios y extremos
        y_min_values = [g['y_min'] for g in geo_scalars]
        y_max_values = [g['y_max'] for g in geo_scalars]
        x_min_values = [g['x_min'] for g in geo_scalars]
        x_max_values = [g['x_max'] for g in geo_scalars]
        
        y_sk_min = sum(y_min_values) / len(y_min_values)
        y_sk_max = sum(y_max_values) / len(y_max_values)
        x_sk_min = min(x_min_values)
        x_sk_max = max(x_max_values)
        
        # Calcular o_xm (distancia ortogonal al promedio al eje x)
        o_xm = y_sk_min  # Simplificado por ahora
        
        # Distancia con la siguiente línea
        o_sk = 0.0
        if linea_siguiente:
            geo_siguiente = [self.calculate_geometric_scalars(p) for p in linea_siguiente]
            next_y_min_values = [g['y_min'] for g in geo_siguiente]
            next_y_sk_min = sum(next_y_min_values) / len(next_y_min_values)
            o_sk = next_y_sk_min - y_sk_max
        
        return {
            'y_sk_min': y_sk_min, 'y_sk_max': y_sk_max,
            'x_sk_min': x_sk_min, 'x_sk_max': x_sk_max, 
            'o_xm': o_xm, 'o_sk': o_sk
        }