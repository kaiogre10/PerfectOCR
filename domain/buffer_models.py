from typing import Optional
from bidings import arrays, image

def PointerError():
    return ValueError("Validación fallida: El puntero no puede ser un entero negativo.")

class FullImageKey:
    img_ptr: Optional[int] = None

    def __new__(cls, img_ptr: int):
        # Bloquea floats (1.0), strings ("1"), y explícitamente booleanos (True/False).
        if type(img_ptr) is not int:
            raise TypeError(f"Tipo inválido: se esperaba 'int', se recibió '{type(img_ptr).__name__}'.")

        # 2. VALIDACIÓN DE NEGATIVOS: Un puntero de memoria real no puede ser menor a cero.
        if img_ptr < 0:
            raise PointerError()

        # Guarda el entero en el namespace de la clase
        cls.img_ptr = img_ptr

        # Evita crear instancias en memoria
        return None

    @classmethod
    def clear_img(cls) -> None:
        """Resetea el puntero a None. Cero residuos entre tareas de la cola."""
        image.release_img(ptr_addr=cls.img_ptr) # type: ignore
        cls.img_ptr = None

class BoundingBoxesKey:
    bboxes_ptr: Optional[int] = None

    def __new__(cls, bboxes_ptr: int):
        # Bloquea floats (1.0), strings ("1"), y explícitamente booleanos (True/False).
        if type(bboxes_ptr) is not int:
            raise TypeError(f"Tipo inválido: se esperaba 'int', se recibió '{type(bboxes_ptr).__name__}'.")

        # 2. VALIDACIÓN DE NEGATIVOS: Un puntero de memoria real no puede ser menor a cero.
        if bboxes_ptr < 0:
            raise PointerError()

        # Guarda el entero en el namespace de la clase
        cls.bboxes_ptr = bboxes_ptr

        # Evita crear instancias en memoria
        return None

    @classmethod
    def clear_bboxes(cls) -> None:
        """Resetea el puntero a None. Cero residuos entre tareas de la cola."""
        arrays.release_array(ptr_addr=cls.bboxes_ptr) # type: ignore
        cls.bboxes_ptr = None

class CentroidsKey:
    centroids_ptr: Optional[int] = None

    def __new__(cls, centroids_ptr: int):
        # Bloquea floats (1.0), strings ("1"), y explícitamente booleanos (True/False).
        if type(centroids_ptr) is not int:
            raise TypeError(f"Tipo inválido: se esperaba 'int', se recibió '{type(centroids_ptr).__name__}'.")

        # 2. VALIDACIÓN DE NEGATIVOS: Un puntero de memoria real no puede ser menor a cero.
        if centroids_ptr < 0:
            raise PointerError()

        # Guarda el entero en el namespace de la clase
        cls.centroids_ptr = centroids_ptr

        # Evita crear instancias en memoria
        return None

    @classmethod
    def clear_centroids(cls) -> None:
        """Resetea el puntero a None. Cero residuos entre tareas de la cola."""
        arrays.release_array(ptr_addr=cls.centroids_ptr) # type: ignore
        cls.centroids_ptr = None

class CroppedBufferKey:
    cropped_buffer_ptr: Optional[int] = None

    def __new__(cls, cropped_buffer_ptr: int):
        # Bloquea floats (1.0), strings ("1"), y explícitamente booleanos (True/False).
        if type(cropped_buffer_ptr) is not int:
            raise TypeError(f"Tipo inválido: se esperaba 'int', se recibió '{type(cropped_buffer_ptr).__name__}'.")

        # 2. VALIDACIÓN DE NEGATIVOS: Un puntero de memoria real no puede ser menor a cero.
        if cropped_buffer_ptr < 0:
            raise PointerError()

        # Guarda el entero en el namespace de la clase
        cls.cropped_buffer_ptr = cropped_buffer_ptr

        # Evita crear instancias en memoria
        return None

    @classmethod
    def free_cropped_buffer_images(cls) -> None:
        """Resetea el puntero a None. Cero residuos entre tareas de la cola."""
        cls.cropped_buffer_ptr = None

class AoSCroppedImageKey:
    cropped_aos_ptr: Optional[int] = None

    def __new__(cls, cropped_aos_ptr: int):
        # Bloquea floats (1.0), strings ("1"), y explícitamente booleanos (True/False).
        if type(cropped_aos_ptr) is not int:
            raise TypeError(f"Tipo inválido: se esperaba 'int', se recibió '{type(cropped_aos_ptr).__name__}'.")

        # 2. VALIDACIÓN DE NEGATIVOS: Un puntero de memoria real no puede ser menor a cero.
        if cropped_aos_ptr < 0:
            raise PointerError()

        # Guarda el entero en el namespace de la clase
        cls.cropped_aos_ptr = cropped_aos_ptr

        # Evita crear instancias en memoria
        return None

    @classmethod
    def free_cropped_asos(cls) -> None:
        """Resetea el puntero a None. Cero residuos entre tareas de la cola."""
        cls.cropped_aos_ptr = None
