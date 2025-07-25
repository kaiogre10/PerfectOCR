# core/workflow/geometry/segmenter.py
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
from skimage.measure import label, regionprops
from typing import Dict, List, Any, Tuple

class Segmenter:
    def segment_words(self, binarized_img: np.ndarray, contexto: dict,
                      min_area: int = 30, min_aspect: float = 0.1, max_aspect: float = 10.0) -> Dict[str, Any]:
        """
        Detecta palabras en la imagen binarizada y devuelve solo las coordenadas de los bboxes.
        Filtra posibles regiones de ruido por área y aspecto.
        Retorna:
            {
                "line_bbox": (min_row, min_col, max_row, max_col),
                "word_bboxes": [ (minr, minc, maxr, maxc), ... ]
            }
        """
        line_bbox = contexto.get("line_bbox", None)
        
        labeled = label(binarized_img)
        regions = regionprops(labeled)
        word_bboxes = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            area = region.area
            width = maxc - minc
            height = maxr - minr
            aspect_ratio = width / height if height > 0 else 0

            # Filtro de ruido: área mínima y aspecto razonable
            if area >= min_area and min_aspect <= aspect_ratio <= max_aspect:
                word_bboxes.append((minr, minc, maxr, maxc))
        
        return {
            "line_bbox": line_bbox,
            "word_bboxes": word_bboxes
        }