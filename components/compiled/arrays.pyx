# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stddef cimport size_t
from libc.stdint cimport int32_t
from libc.string cimport memset
from image cimport Image, destroy_image
import numpy as _numpy
cimport numpy as cnp

cdef inline Image* new_buffer(int width, int height, int channels, size_t dtype) except *:
    return new Image(width, height, channels, dtype) # type: ignore

cdef inline Image* get_image_ptr(size_t ptr_addr) noexcept:
    if ptr_addr == 0:
        return NULL
    return <Image*>ptr_addr

cdef inline void _set_array_vals(size_t ptr_addr) except *:
    cdef Image* image  = get_image_ptr(ptr_addr)
    memset(image.data(), 0, image.size())

cdef inline size_t _get_arr_dtype(object dtype_raw) except *:
    cdef object dtype_obj = _numpy.dtype(dtype_raw)
    return <size_t>dtype_obj.itemsize

cdef inline bint _validate_arr_vals(int rows, int cols, int depth) noexcept:
    return (rows > 0) and (cols > 0) and (depth > 0)

def create_new_array(int rows, int cols, int depth, object raw_dtype):
    if not _validate_arr_vals(rows, cols, depth):
        raise ValueError("rows, cols y depth deben ser mayores que cero")

    cdef Image* new_array
    cdef size_t dtype = _get_arr_dtype(raw_dtype)
    
    new_array = new_buffer(rows, cols, depth, dtype)
    # _set_array_vals(<size_t>image)
    return <size_t>new_array

def get_3d_array(size_t arr_ptr, object raw_dtype):
    cdef Image* new_array = get_image_ptr(arr_ptr)
    if new_array == NULL:
        raise ValueError("arr_ptr no puede ser cero")

    cdef int rows = new_array.height()
    cdef int cols = new_array.width()
    cdef int depth = new_array.channels()

    if not _validate_arr_vals(rows, cols, depth):
        return _numpy.empty(0, dtype=_numpy.int32)

    # cdef size_t dtype = _numpy.dtype(raw_dtype).itemsize
    cdef int32_t* data = <int32_t*>new_array.data()

    cdef cnp.int32_t[:, :, ::1] view = <cnp.int32_t[:rows, :cols, :depth]> data

    return _numpy.asarray(view)

def release(size_t ptr_addr):
    if ptr_addr == 0:
        return
    cdef Image* img = get_image_ptr(ptr_addr)
    destroy_image(img)