# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint8_t
from libc.string cimport memset, memcpy
from image cimport Image, destroy_image, new_buffer, get_image_ptr
import numpy as _numpy

cdef inline int _list_len(list py_list) except *:
    return len(py_list)

cdef inline void _set_array_vals(Image* image) except *:
    memset(image.data(), 0, image.size())

cdef inline size_t _get_arr_dtype(object dtype_raw) except *:
    cdef object dtype_obj = _numpy.dtype(dtype_raw)
    return <size_t>dtype_obj.itemsize

cdef inline bint _validate_arr_vals(int rows, int cols, int channels) noexcept:
    return (0 < rows) and (0 < cols) and (channels > 0)

def create_new_array(int rows, int cols, int channels, object raw_dtype):
    if not _validate_arr_vals(rows, cols, channels):
        raise ValueError("rows, cols y depth deben ser mayores que cero")

    cdef Image* new_array
    cdef size_t dtype = _get_arr_dtype(raw_dtype)
    
    new_array = new_buffer(rows, cols, channels, dtype)
    # _set_array_vals(<size_t>image)
    return <size_t>new_array

def get_2d_array(size_t arr_ptr):
    cdef Image* new_array = get_image_ptr(arr_ptr)
    if new_array == NULL:
        return None

    cdef int cols = new_array.height()
    cdef int rows = new_array.width()
    cdef int channels = new_array.channels()

    if not _validate_arr_vals(rows, cols, channels):
        return _numpy.empty(0)

    # cdef size_t dtype = _numpy.dtype(raw_dtype).itemsize
    cdef int32_t* data = <int32_t*> new_array.data()
    cdef int32_t[:, ::1] view = <int32_t[:rows, :cols]> data

    return _numpy.asarray(view)

def get_3d_array(size_t arr_ptr):
    cdef Image* new_array = get_image_ptr(arr_ptr)
    if new_array == NULL:
        return _numpy.empty(0, dtype=_numpy.uint8)

    cdef int cols = new_array.height()
    cdef int rows = new_array.width()
    cdef int channels = new_array.channels()

    if not _validate_arr_vals(rows, cols, channels):
        return _numpy.empty(0, dtype=_numpy.uint8)

    # cdef size_t dtype = _numpy.dtype(raw_dtype).itemsize
    cdef uint8_t* data = <uint8_t*> new_array.data()
    cdef uint8_t[:, :, ::1] view = <uint8_t[:rows, :cols, :channels]> data

    return _numpy.asarray(view)

def release_array(size_t ptr_addr):
    cdef Image* img = get_image_ptr(ptr_addr)
    destroy_image(img)

def reset_vals(size_t ptr_addr):
    cdef Image* image = get_image_ptr(ptr_addr)
    _set_array_vals(image)