# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t
from image cimport Image
from image_loader cimport load_image
import numpy as np
cimport numpy as np

def load(str filepath):
    cdef Image* img = load_image(filepath.encode('utf-8'))
    if img == NULL:
        raise RuntimeError(f"Fallo al cargar imagen: {filepath}")
    return <size_t>img

def get_array(size_t ptr_addr):
    cdef Image* img = <Image*>ptr_addr
    cdef int h = img.height()
    cdef int w = img.width()
    cdef uint8_t* data = img.data()
    return np.asarray(<np.uint8_t[:h, :w]> data)

def release(size_t ptr_addr):
    if ptr_addr == 0:
        return
    cdef Image* img = <Image*>ptr_addr
    destroy_image(img)