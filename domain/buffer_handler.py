from typing import Tuple, Optional, Any
import numpy as np
from numpy import typing
from components.compiled import image, arrays

class ImageBuffer:
    """Contenedor de direcciones en memoria para imágenes en búfer de C."""
    __slots__ = ('_img_ptr', '_data', '_shape')
    def __init__(self, img_ptr: int):
        self._img_ptr: int = img_ptr
        self._data: Optional[np.ndarray[Any, Any]] = None
        self._shape: Optional[Tuple[int, ...]] = None

    @property
    def data(self) -> Optional[np.ndarray[Any, Any]]:
        if self._data is None:
            if not isinstance(self._img_ptr, int):  # type: ignore
                raise ValueError("Se requiere 'img_ptr' o 'data' para obtener el arreglo.")
            self._data = image.get_image_array(self._img_ptr)
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is None:
            self._shape = self.data.shape # type: ignore
        return self._shape

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def channels(self) -> int:
        # Los arreglos de NumPy no tienen atributo .channels
        return self.shape[2] if len(self.shape) > 2 else 1

    @property
    def size(self) -> int:
        return self.data.size # type: ignore
    
class ArrayBuffer:
    """Contenedor de direcciones en memoria para estructuras de datos en buffer de C"""
    __slots__ = ('_arr_ptr', '_data', '_shape', "_dims")
    def __init__(self, arr_ptr: int, dims: Optional[int] = None):
        self._arr_ptr: int = arr_ptr
        self._dims: Optional[int] = dims
        self._data: Optional[np.ndarray[Any, Any]] = None
        self._shape: Optional[Tuple[int, ...]] = None

    @property
    def data(self) -> Optional[np.ndarray[Any, Any]]:
        if self._data is None:
            if not isinstance(self._arr_ptr, int):  # type: ignore
                raise ValueError("Se requiere 'arr_ptr' o 'data' para obtener el arreglo.")
            
            if self._dims is None or self._dims == 3:
                self._data = arrays.get_3d_array(self._arr_ptr)
                
            elif self._dims == 2:
                self._data = arrays.get_2d_array(self._arr_ptr)
                self._shape = self._data.shape[:2]

        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is None:
            self._shape = self.data.shape
        return self._shape

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def depth(self) -> int:
        # Los arreglos de NumPy no tienen atributo .channels
        return self.shape[2]

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> typing.DTypeLike:
        return self.data.dtype
