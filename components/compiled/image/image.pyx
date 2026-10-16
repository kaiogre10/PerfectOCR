# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t
from libc.string cimport memcpy
from image cimport Image, destroy_image, new_buffer, get_image_ptr
from image_loader cimport load_image, rotate_image
import numpy as _numpy

def load(str filepath):
    cdef Image* img = load_image(filepath.encode('utf-8')) # type: ignore
    if img == NULL: # type: ignore
        return None
    
    return <size_t>img

def get_image_array(size_t ptr_addr):
    cdef Image* img = get_image_ptr(ptr_addr)
    cdef int h = img.height()
    cdef int w = img.width()
    cdef uint8_t* data = <uint8_t*> img.data()
    cdef uint8_t[:, ::1] view = <uint8_t[:h, :w]> data
    return _numpy.asarray(view)
    
def release_img(size_t ptr_addr):
    cdef Image* img = get_image_ptr(ptr_addr)
    destroy_image(img)

def rotate(size_t ptr_addr, double angle, double center_x, double center_y):
    cdef Image* source = get_image_ptr(ptr_addr)
    cdef Image* rotated

    if source == NULL:
        return None
        
    rotated = rotate_image(source, angle, center_x, center_y)
    if rotated == NULL:
        return None
        
    return <size_t>rotated

def copy_image(size_t ptr_addr):
    cdef Image* image = get_image_ptr(ptr_addr)
    cdef Image* copy
    cdef size_t dtype = sizeof(uint8_t)

    copy = new_buffer(image.width(), image.height(), image.channels(), dtype)
    memcpy(copy.data(), image.data(), image.size()) # type: ignore
    return <size_t>copy