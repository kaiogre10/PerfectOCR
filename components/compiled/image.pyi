# components/compiled/image.pyi
import numpy as np
from typing import Any

def load(filepath: str) -> int:
    """Carga una imagen desde disco.
    Returns:
        Un handle opaco (dirección de memoria) que debe liberarse con release().

    Raises:
        RuntimeError: Si la imagen no se puede cargar o no es de 1 canal.
    """

def get_array(ptr_addr: int) -> np.ndarray[Any, np.dtype[np.uintp]]:
    """Devuelve un array NumPy de la imagen.

    Warning:
        El array NO es dueño de la memoria. No llames a release() mientras
        el array siga vivo. Haz una copia con .copy() si necesitas independencia.
    """

def release(ptr_addr: int) -> None:
    """Libera la imagen apuntada por ptr_addr.

    Warning:
        Llamar dos veces con el mismo puntero es undefined behavior.
        Llamar mientras un array de get_array() sigue vivo es undefined behavior.
    """
