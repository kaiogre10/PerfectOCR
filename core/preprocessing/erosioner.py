
import cv2
import numpy as np
from typing import List, Dict, Tuple

class ResolutionAssessor:
    """
    Estima la resolución efectiva (DPI) de una imagen mediante una prueba de
    "punto de fractura" progresiva en polígonos de texto.
    """

    def __init__(self, calibration_data: Dict[int, float]):
        """
        Inicializa el asesor con los datos de calibración.

        Args:
            calibration_data: Un diccionario que mapea el número de iteraciones de
                              erosión al DPI correspondiente.
                              Ej: {2: 200, 5: 300, 9: 400, 14: 600}
        """
        self.calibration_data = calibration_data
        # Kernel de erosión pequeño y consistente para la prueba de estrés.
        self.erosion_kernel = np.ones((3, 3), np.uint8)
        # Umbral de binarización agresivo.
        self.binary_threshold = 127

    def _select_best_candidate(
        self, polygon_images: List[np.ndarray]
    ) -> np.ndarray:
        """
        Selecciona el mejor polígono candidato para el análisis.
        Elige el polígono con la varianza más baja (texto más limpio).

        Args:
            polygon_images: Una lista de imágenes de polígonos en escala de grises.

        Returns:
            La imagen del polígono más adecuado para la prueba.
        """
        if not polygon_images:
            raise ValueError("La lista de polígonos no puede estar vacía.")

        min_variance = float('inf')
        best_candidate = None

        for poly_img in polygon_images:
            # Asegurarse de que el polígono no sea demasiado pequeño
            if poly_img.size < 500: # Umbral de área mínima
                continue
            
            variance = np.var(poly_img)
            if variance < min_variance:
                min_variance = variance
                best_candidate = poly_img

        if best_candidate is None:
             # Si ningún polígono cumplió los criterios, usar el más grande.
            best_candidate = max(polygon_images, key=lambda img: img.size)

        return best_candidate

    def find_fracture_point(
        self, polygon_candidate: np.ndarray, max_iterations: int = 20, stop_threshold: float = 0.1
    ) -> int:
        """
        Aplica erosión iterativa y encuentra el "punto de fractura".

        Args:
            polygon_candidate: La imagen del polígono seleccionado (escala de grises).
            max_iterations: El número máximo de iteraciones a probar.
            stop_threshold: El porcentaje de texto restante en el que se considera
                              que la imagen está completamente "rota".

        Returns:
            El número de iteraciones en el que el texto se "rompió".
        """
        # 1. Binarizar el polígono intacto para obtener la referencia inicial.
        _, intact_binary = cv2.threshold(
            polygon_candidate, self.binary_threshold, 255, cv2.THRESH_BINARY
        )

        # 2. Medir el estado inicial.
        initial_text_pixels = np.sum(intact_binary == 255)
        if initial_text_pixels == 0:
            return 0

        degraded_image = intact_binary.copy()
        
        # 3. Bucle de estrés progresivo.
        for i in range(1, max_iterations + 1):
            # Aplicar una iteración de erosión.
            degraded_image = cv2.erode(degraded_image, self.erosion_kernel, iterations=1)
            
            # Medir el texto restante.
            remaining_text_pixels = np.sum(degraded_image == 255)
            
            # Calcular qué proporción del texto original sobrevive.
            survival_ratio = remaining_text_pixels / initial_text_pixels
            
            # 4. Comprobar si se alcanzó el punto de fractura.
            if survival_ratio < stop_threshold:
                # El texto ha desaparecido casi por completo.
                return i
        
        # Si sobrevive a todas las iteraciones, es de muy alta calidad.
        return max_iterations

    def assess_dpi(self, polygon_images: List[np.ndarray]) -> int:
        """
        Realiza el proceso completo para estimar los DPI de la imagen original.

        Args:
            polygon_images: Una lista de los polígonos detectados en la imagen.

        Returns:
            El valor de DPI estimado.
        """
        # Paso 1: Seleccionar el mejor polígono para la prueba.
        candidate = self._select_best_candidate(polygon_images)
        
        # Paso 2: Encontrar su punto de fractura.
        fracture_iteration = self.find_fracture_point(candidate)
        
        # Paso 3: Consultar los datos de calibración para encontrar el DPI.
        # Encuentra el DPI correspondiente a la iteración más cercana que sea
        # mayor o igual al punto de fractura.
        
        # Ordenar las claves de calibración (iteraciones)
        calibrated_iterations = sorted(self.calibration_data.keys())
        
        # Encontrar la mejor coincidencia
        for iteration_threshold in calibrated_iterations:
            if fracture_iteration <= iteration_threshold:
                return self.calibration_data[iteration_threshold]
        
        # Si supera todas las iteraciones de calibración, devolver el DPI más alto.
        return self.calibration_data[max(calibrated_iterations)]


# --- Ejemplo de Uso ---

# 1. Datos de Calibración (usted debe generar esto una vez)
# Mapeo: {Iteración de Fractura: DPI}
CALIBRATION_TABLE = {
    2: 200,   # Si se rompe en <= 2 iteraciones, es ~200 DPI
    5: 300,   # Si se rompe en <= 5 iteraciones, es ~300 DPI
    9: 400,   # ...
    14: 600   # Si resiste hasta 14 iteraciones, es ~600 DPI
}

# 2. Inicializar el asesor con la tabla
assessor = ResolutionAssessor(calibration_data=CALIBRATION_TABLE)

# 3. Simular una lista de polígonos de una imagen de entrada
#    (en su pipeline real, esta lista vendría de la salida de PaddleOCR)
#    Aquí cargamos una imagen de ejemplo y la dividimos para simular polígonos.
#    (Reemplace 'ruta/a/su/imagen.png' con una imagen real para probar)
try:
    image = cv2.imread('ruta/a/su/imagen.png', cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    # Simular 4 polígonos
    sample_polygons = [
        image[0:h//2, 0:w//2],
        image[0:h//2, w//2:w],
        image[h//2:h, 0:w//2],
        image[h//2:h, w//2:w]
    ]

    # 4. Estimar los DPI
    estimated_dpi = assessor.assess_dpi(sample_polygons)

    print(f"El DPI estimado de la imagen es: {estimated_dpi}")

except Exception as e:
    print(f"No se pudo cargar la imagen de ejemplo. Asegúrese de que la ruta sea correcta. Error: {e}")