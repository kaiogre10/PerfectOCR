# bidings/image/image.pyi
import numpy as np
from typing import Any, Optional, List

def load(filepath: str) -> Optional[int]:
    """Carga una imagen desde disco.
    Returns:
        Un handle opaco (dirección de memoria) que debe liberarse con release().

    Raises:
        RuntimeError: Si la imagen no se puede cargar o no es de 1 canal.
    """

def get_image_array(ptr_addr: int, dtype: bytes) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Devuelve un array NumPy de la imagen.

    Warning:
        El array NO es dueño de la memoria. No llames a release() mientras
        el array siga vivo. Haz una copia con .copy() si necesitas independencia.
    """

def release_img(ptr_addr: int) -> None:
    """Libera la imagen apuntada por ptr_addr.

    Warning:
        Llamar dos veces con el mismo puntero es Segmentation Fault.
        Llamar mientras un array de get_array() sigue vivo es undefined behavior.
    """

def rotate(ptr_addr: int, angle: float, center_x: float, center_y: float) -> Optional[int]:
    """Manda a rotar la imagen y recibe puntero de la imagen actualizada"""

def copy_image(ptr_addr: int, dtype: bytes) -> Optional[int]:
    """Crea una copia independiente de la imagen y devuelve su handle opaco.

    La copia conserva sus propios bytes y debe liberarse con release().
    """
def get_crops_ptr(grouped_crops: int, total_buffers: int) -> List[int]:
    """Devuelve la lista de punteros de los crops"""

def release_crops(cropped_buffer_ptr: int, addr_aos: int, total_crops: int) -> None:
    """Libera las imagenes en memoria, usese co precacunion"""