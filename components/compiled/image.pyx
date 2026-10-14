# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libc.string cimport memcpy, memset
from image cimport Image, destroy_image
from image_loader cimport load_image, rotate_image
import numpy as np
cimport numpy as np

cdef inline Image* _new_buffer(int rows, int cols, int channels) except *:
    return new Image(rows, cols, channels)

cdef inline void _set_array_vals(size_t img_ptr) except *:
    cdef Image* image = <Image*>img_ptr
    memset(image.data(), 0, image.size())

def load(str filepath):
    cdef Image* img = load_image(filepath.encode('utf-8')) # type: ignore
    if img == NULL:
        raise RuntimeError(f"Fallo al cargar imagen: {filepath}")
    
    return <size_t>img

def get_array(size_t ptr_addr):
    cdef Image* img = <Image*>ptr_addr
    cdef int h = img.height()
    cdef int w = img.width()
    cdef uint8_t* data = <uint8_t*>img.data()
    cdef np.uint8_t[:, ::1] view = <np.uint8_t[:h, :w]> data # type: ignore
    return np.ascontiguousarray(np.asarray(view))
    
def release(size_t ptr_addr):
    if ptr_addr == 0:
        return
    
    cdef Image* img = <Image*>ptr_addr
    destroy_image(img)

def rotate(size_t ptr_addr, double angle, double center_x, double center_y):
    cdef Image* source = <Image*>ptr_addr
    cdef Image* rotated

    if source == NULL:
        return None

    rotated = rotate_image(source, angle, center_x, center_y)
    if rotated == NULL:
        return None
        
    return <size_t>rotated

def copy_image(size_t ptr_addr):
    cdef Image* image
    cdef Image* copy

    if ptr_addr == 0:
        raise ValueError("El puntero de la imagen no puede ser nulo")

    image = <Image*>ptr_addr
    copy = _new_buffer(image.width(), image.height(), image.channels())
    memcpy(copy.data(), image.data(), image.size()) # type: ignore
    return <size_t>copy

def get_new_array(size_t arr_ptr):
    cdef Image* image = <Image*>arr_ptr
    cdef int h = image.height()
    cdef int w = image.width()
    cdef int channels = image.channels()
    cdef uint8_t* data = <uint8_t*>image.data()
    cdef np.uint8_t[:, :, ::1] view = <np.uint8_t[:h, :w, :channels]> data # type: ignore
    return np.ascontiguousarray(np.asarray(view))

def create_buffer_array(int rows, int cols, int n_sc):
    cdef Image* image
    if rows <= 0 or cols <= 0:
        raise ValueError("Las dimensiones del array deben ser positivas")

    image = _new_buffer(cols, rows, n_sc)
    _set_array_vals(<size_t>image)
    return <size_t>image

def clear_buffer(size_t ptr_addr):
    if ptr_addr == 0:
        raise ValueError("El puntero del buffer no puede ser nulo")
    _set_array_vals(ptr_addr)