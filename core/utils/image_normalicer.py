import cv2
import numpy as np
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def normalice_image(img: Optional[np.ndarray[Any, Any]]) -> Optional[np.ndarray[Any, np.dtype[np.uint8]]]:
    """
    Normaliza una imagen de entrada:
    - Asegura que sea ndarray
    - Convierte BGR->GRAY si viene con 3/4 canales
    - Convierte a dtype uint8 (escala floats en [0,1] a 0-255)
    - Garantiza que el array sea C-contiguo
    Retorna ndarray uint8 o None si ocurre un error / imagen vacía.
    """
    try:
        if img is None:
            logger.error("normalice_image: imagen None recibida")
            return None
        
        if not validate_image(img):
            logger.error("Imagen blanca/negra completamente")
            return None

        try:
            img_arr = np.asarray(img)
        except Exception as e:
            logger.error(f"normalice_image: no se pudo convertir a ndarray: {e}", exc_info=True)
            return None

        if img_arr.size == 0:
            logger.error("normalice_image: ndarray vacío.")
            return None

        # Si llega en color, convertir a gris (BGR->GRAY)
        if img_arr.ndim == 3 and img_arr.shape[2] in (3, 4):
            try:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                logger.info("normalice_image: convertida imagen BGR->GRAY")
            except Exception as e:
                logger.warning(f"normalice_image: no se pudo convertir BGR->GRAY: {e}", exc_info=True)

        # Asegurar dtype uint8 (flotantes se escalan si están en [0,1], otros se recortan)
        if img_arr.dtype != np.uint8:
            try:
                if np.issubdtype(img_arr.dtype, np.floating):
                    mx = float(img_arr.max()) if img_arr.size > 0 else 0.0
                    if mx <= 1.0:
                        img_arr = (img_arr * 255.0).round().astype(np.uint8)
                    else:
                        img_arr = np.clip(img_arr, 0, 255).round().astype(np.uint8)
                    logger.info("normalice_image: convertida imagen float->uint8 (escalada si hizo falta)")
                else:
                    img_arr = img_arr.astype(np.uint8, copy=False)
                    logger.info("normalice_image: casteada imagen a uint8")
            except Exception as e:
                logger.error(f"normalice_image: fallo al convertir dtype: {e}", exc_info=True)
                try:
                    img_arr = np.array(img_arr, dtype=np.uint8)
                except Exception:
                    return None

        # Asegurar contigüidad para OpenCV
        if not img_arr.flags['C_CONTIGUOUS']:
            img_arr = np.ascontiguousarray(img_arr)
            logger.info("normalice_image: imagen hecha contigua en memoria")

        # Logueo detallado para trazabilidad
        try:
            vmin = int(img_arr.min()); vmax = int(img_arr.max()); vmean = float(img_arr.mean())
        except Exception:
            vmin = vmax = None; vmean = None

        logger.debug(
            "normalice_image: id=%d shape=%s dtype=%s min=%s max=%s mean=%s",
            id(img_arr), getattr(img_arr, "shape", None), getattr(img_arr, "dtype", None),
            vmin, vmax, f"{vmean:.2f}" if vmean is not None else None
        )
        
        return img_arr
        
    except Exception  as e:
        logger.error(f"Error normalizando imagen: {e}", exc_info=True)
        return None

def validate_image(img: np.ndarray[Any, Any]) -> bool:
    min_threshold = 5
    max_threshold = 250
    img_mean =  np.mean(img).astype(np.uint8)
    if img_mean < min_threshold or max_threshold < img_mean:
        return False
    else:
        return True