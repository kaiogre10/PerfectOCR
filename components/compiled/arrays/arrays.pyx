# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint8_t, uint16_t, uint32_t, int8_t, int16_t
from libc.string cimport memset
from image cimport Image, destroy_image, new_buffer, get_image_ptr
import numpy as _numpy

cdef inline object _view(void* data, int w, int h, object dtype, int channels, bool tridimentional):
    if tridimentional == False:
        if dtype == _numpy.dtype("uint8"): return _numpy.asarray(<uint8_t[:w, :h]> data)
        if dtype == _numpy.dtype("uint16"): return _numpy.asarray(<uint16_t[:w, :h]> data)
        if dtype == _numpy.dtype("uint32"): return _numpy.asarray(<uint32_t[:w, :h]> data)
        if dtype == _numpy.dtype("int8"): return _numpy.asarray(<int8_t[:w, :h]> data)
        if dtype == _numpy.dtype("int16"): return _numpy.asarray(<int16_t[:w, :h]> data)
        if dtype == _numpy.dtype("int32"): return _numpy.asarray(<int32_t[:w, :h]> data)
        if dtype == _numpy.dtype("float32"): return _numpy.asarray(<float[:w, :h]> data)
        if dtype == _numpy.dtype("float64"): return _numpy.asarray(<double[:w, :h]> data)

    else:
        if dtype == _numpy.dtype("uint8"): return _numpy.asarray(<uint8_t[:w, :h, :channels]> data)
        if dtype == _numpy.dtype("uint16"): return _numpy.asarray(<uint16_t[:w, :h, :channels]> data)
        if dtype == _numpy.dtype("uint32"): return _numpy.asarray(<uint32_t[:w, :h, :channels]> data)
        if dtype == _numpy.dtype("int8"): return _numpy.asarray(<int8_t[:w, :h, :channels]> data)
        if dtype == _numpy.dtype("int16"): return _numpy.asarray(<int16_t[:w, :h, :channels]> data)
        if dtype == _numpy.dtype("int32"): return _numpy.asarray(<int32_t[:w, :h, :channels]> data)
        if dtype == _numpy.dtype("float32"): return _numpy.asarray(<float[:w, :h, :channels]> data)
        if dtype == _numpy.dtype("float64"): return _numpy.asarray(<double[:w, :h, :channels]> data)

    raise TypeError(f"dtype no soportado: {dtype}")

cdef inline void _set_array_vals(Image* image) except *:
    memset(image.data(), 0, image.size())

cdef inline object _get_arr_dtype(str dtype_raw) except *:
    return _numpy.dtype(dtype_raw)

cdef inline size_t _get_arr_dtype_size(str dtype_raw) except *:
    cdef object dtype_obj = _get_arr_dtype(dtype_raw)
    return <size_t>dtype_obj.itemsize

cdef inline bint _validate_arr_vals(int rows, int cols, int channels) noexcept:
    return (0 < rows) and (0 < cols) and (channels > 0)

def create_new_array(int rows, int cols, int channels, str raw_dtype):
    if not _validate_arr_vals(rows, cols, channels):
        raise ValueError("rows, cols y depth deben ser mayores que cero")

    cdef Image* new_array
    cdef size_t dtype = _get_arr_dtype_size(raw_dtype)
    
    new_array = new_buffer(rows, cols, channels, dtype)
    return <size_t>new_array

def get_array(size_t arr_ptr, str dtype_raw, int dims):
    cdef Image* new_array = get_image_ptr(arr_ptr)
    if new_array == NULL:
        return None

    cdef int cols = new_array.height()
    cdef int rows = new_array.width()
    cdef int channels = new_array.channels()

    if not _validate_arr_vals(rows, cols, channels):
        return None

    cdef bool tridimentional = True

    if dims == 0 or dims == 2:
        tridimentional = False

    cdef object dtype = _get_arr_dtype(dtype_raw)
    cdef void* data = <void*> new_array.data()
    return _view(data=data, w=rows, h=cols, dtype=dtype, channels=channels, tridimentional=tridimentional)

def release_array(size_t ptr_addr):
    cdef Image* img = get_image_ptr(ptr_addr)
    destroy_image(img)

def reset_vals(size_t ptr_addr):
    cdef Image* image = get_image_ptr(ptr_addr)
    _set_array_vals(image)