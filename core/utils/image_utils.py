import cv2
import numpy as np
from typing import Any, Optional, List
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
            img_arr = np.asarray(img, dtype=np.uint8)
        except Exception as e:
            logger.error(f"normalice_image: no se pudo convertir a ndarray: {e}", exc_info=True)
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
    
        return img_arr # type: ignore
        
    except Exception  as e:
        logger.error(f"Error normalizando imagen: {e}", exc_info=True)
    return None

def calculate_img_values(img: np.ndarray[Any, Any]):
    img_mean = np.mean(img).astype(np.uint8)
    img_dims = img.shape[:2]
    return int(img_mean), img_dims

def validate_image(img: Optional[np.ndarray[Any, Any]]) -> bool:
    min_threshold = 10
    max_threshold = 250
    if img is None:
        return False
    
    img_mean, img_dims = calculate_img_values(img)
    img_size = img_dims[0] * img_dims[1] 

    if img_mean < min_threshold or max_threshold < img_mean or img_size==0:
        return False
    
    else:
        return True

def validate_full_image(img: np.ndarray[Any, Any]):
    _, img_dims = calculate_img_values(img)
    img_size = img_dims[0] * img_dims[1]
    
    if np.all(img)==255 or np.all(img)==0 or img_size==0:
        return img_dims
    
    else:
        return [0, 0]

def cropp_img(full_img: np.ndarray[Any, np.dtype[np.uint8]], all_bboxes: List[np.ndarray[Any, Any]] | np.ndarray[Any, Any], padding: Optional[int] = None) -> Optional[np.ndarray[Any, np.dtype[np.uint8]]]:
    img_h = full_img.shape[0]
    img_w = full_img.shape[1]

    if padding is None:
        padding = 0

    bboxes_array = np.array(all_bboxes)
    
    if bboxes_array.ndim == 1 and bboxes_array.shape[0] == 4:
        bboxes_array = bboxes_array.reshape(1, 4)

    x1, y1, x2, y2 = bboxes_array[0, 0], bboxes_array[0, 1], bboxes_array[0, 2], bboxes_array[0, 3]

    # Aplicar padding y clipping
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(img_w, x2 + padding)
    py2 = min(img_h, y2 + padding)

    crop_x1, crop_y1 = int(px1), int(py1)
    crop_x2, crop_y2 = int(px2), int(py2)

    cropped: np.ndarray[Any, np.dtype[np.uint8]] = full_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    return cropped

def use_bilateral_filter(img: np.ndarray[Any, np.dtype[np.uint8]], d: int, sigma_color: int, sigma_space: int)-> np.ndarray[Any, np.dtype[np.uint8]]:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space).astype(np.uint8)

def use_sobel(img: np.ndarray[Any, np.dtype[np.uint8]], ksize: int) -> float:
    return np.mean(np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize)))