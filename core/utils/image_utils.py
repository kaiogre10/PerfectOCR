import cv2
import numpy as np
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

min_threshold = 10
max_threshold = 250

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

def validate_image(img: np.ndarray[Any, Any]) -> bool:
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
 
def correct_img(full_img: np.ndarray[Any, np.dtype[np.uint8]], config: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    worker_config = config.get('cleaner', {})
    std_low = worker_config.get("std_low")
    sp_thr: float = worker_config.get("sp_thr")
    clahe_clip_base = worker_config.get("clahe_clip")
    clahe_grid = worker_config["clahe_grid"]
    dimension_thresholds_px = worker_config["dimension_thresholds_px"]
    kernel_size: int = worker_config.get("kernel_size")
    try:
        size = full_img.size
            
        ext_low = float((full_img <= 5).sum())
        ext_high = float((full_img >= 250).sum())
        sp_ratio = (ext_low + ext_high) / size

        # 1) Desruido sal‑y‑pimienta (rápido, solo si aplica)
        if sp_ratio > sp_thr:
            den = cv2.medianBlur(src=full_img, ksize=kernel_size)
            full_img[...] = den

        # Recalcular contraste
        std1 = float(np.std(full_img))

        # 2) Contraste local con CLAHE (solo si contraste bajo)
        if std1 < std_low:
            h, w = full_img.shape[:2]
            max_dim = max(h, w)
            
            if max_dim < dimension_thresholds_px[0]:
                grid_size = tuple(clahe_grid[0])
            elif max_dim < dimension_thresholds_px[1]:
                grid_size = tuple(clahe_grid[1])
            else:
                grid_size = tuple(clahe_grid[2])
            
            logger.debug(f"CLAHE para full_img con grid: {grid_size} basado en max_dim: {max_dim}")

            clahe = cv2.createCLAHE(clipLimit=clahe_clip_base, tileGridSize=grid_size)
            en1 = clahe.apply(full_img)
            full_img[...] = en1

            # Si siguió bajo, subir ligeramente el clipLimit
            std2 = float(np.std(full_img))
            if std2 < std_low:
                clahe2 = cv2.createCLAHE(clipLimit=clahe_clip_base + 0.5, tileGridSize=grid_size)
                en2 = clahe2.apply(full_img)
                full_img[...] = en2

        # 3) Nitidez local (unsharp adaptativo)
        lap = cv2.Laplacian(full_img, cv2.CV_64F)
        lap_var = float(lap.var())
        stdf = float(np.std(full_img))

        if lap_var < 20.0 or stdf <= 25.0:
            alpha, beta = 1.2, -0.2  # suave
        elif lap_var < 60.0:
            alpha, beta = 1.4, -0.4  # medio
        else:
            alpha, beta = 1.1, -0.1  # mínimo

        blur = cv2.GaussianBlur(full_img, (3, 3), 0)
        sharp = cv2.addWeighted(full_img, alpha, blur, beta, 0)
        np.clip(sharp, 0, 255, out=sharp)
        if sharp.dtype != np.uint8:
            sharp = sharp.astype(np.uint8, copy=False)

            full_img[...] = sharp
            
        return full_img
        
    except Exception as e:
        logger.error(f"Cleaner: {e}", exc_info=True)
        return full_img
