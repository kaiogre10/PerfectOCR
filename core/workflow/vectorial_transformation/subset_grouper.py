# PerfectOCR/core/vectorization/subset_grouper.py
import logging
import numpy as np
from typing import List, Dict, Any

Poligono = Dict[str, Any]

logger = logging.getLogger(__name__)

class SubsetConstructor:
    def _calcular_centroide(self, poligono: Poligono) -> tuple[float, float]:
    
        box = poligono['polygon_coords']
        if not box:
            return 0.0, 0.0
            
        if isinstance(box[0], list):  # Formato de lista de puntos
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = sum(xs) / len(xs) if xs else 0.0
            cy = sum(ys) / len(ys) if ys else 0.0
        else:  # Formato de rectángulo contenedor [xmin, ymin, xmax, ymax]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
        return cx, cy

    def agrupar_poligonos(self, poligonos: List[Poligono]) -> Dict[str, Any]:
        """
        Agrupa los polígonos usando bandas Y adaptativas.
        """
        if not poligonos:
            return {"grouped_lines": []}

        # Calcula centroides para todos los polígonos
        for p in poligonos:
            p['centroide'] = self._calcular_centroide(p)

        # Ordena por centroide Y (de arriba a abajo)
        poligonos_ordenados = sorted(poligonos, key=lambda p: (p['centroide'][1], p['centroide'][0]))
        lineas = []
        linea_actual = []
        if not poligonos_ordenados:
            return {"grouped_lines": []}

        # Inicializa con el primer polígono
        p0 = poligonos_ordenados[0]
        y_min = min([y for x, y in p0['polygon_coords']])
        y_max = max([y for x, y in p0['polygon_coords']])
        linea_actual.append(p0)

        for p in poligonos_ordenados[1:]:
            cy = p['centroide'][1]
            p_y_min = min([y for x, y in p['polygon_coords']])
            p_y_max = max([y for x, y in p['polygon_coords']])
            # Si el centroide cae dentro del intervalo, es la misma línea
            if y_min <= cy <= y_max:
                linea_actual.append(p)
                # Actualiza el intervalo de la línea
                y_min = min(y_min, p_y_min)
                y_max = max(y_max, p_y_max)
            else:
                lineas.append(linea_actual)
                linea_actual = [p]
                y_min = p_y_min
                y_max = p_y_max
        if linea_actual:
            lineas.append(linea_actual)

        # Output clásico (solo las words agrupadas)
        lineas_finales = []
        for linea in lineas:
            linea_ordenada = sorted(linea, key=lambda p: p['centroide'][0])
            for p in linea_ordenada:
                del p['centroide']
            lineas_finales.append(linea_ordenada)

        return {
            "grouped_lines": lineas_finales
        }
