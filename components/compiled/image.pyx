# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libc.string cimport memcpy
from image cimport Image, destroy_image
from image_loader cimport load_image, rotate_image
import numpy as np
cimport numpy as np

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
    copy = new Image(image.width(), image.height(), image.channels())
    memcpy(copy.data(), image.data(), image.size())
    return <size_t>copy