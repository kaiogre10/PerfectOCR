# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from image cimport Image, destroy_image
from image_loader cimport load_image
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

def send_array_data(tuple metadata, size_t ptr_addr):
    cdef Image* img = <Image*>ptr_addr
    cdef uint8_t* img_data = <uint8_t*>img.data()
    cdef void* arr_data = metadata[0]
    cdef size_t ndim = metadata[1]
    cdef size_t width = metadata[2]
    cdef size_t height = metadata[3]
    cdef size_t itemsize = metadata[4]
    crop_polygons(img_data, arr_data, ndim, width, height, itemsize) # type: ignore