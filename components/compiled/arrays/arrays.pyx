# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint8_t, uint16_t, uint32_t, int8_t, int16_t, int64_t, uint64_t
from image cimport Image, destroy_image, new_buffer, get_image_ptr, get_rows, get_cols, get_channels, get_data, get_size
from image_loader cimport get_centroids, get_bboxes
import numpy as _numpy

cdef inline bint _check_null_ptr(Image* ptr_addr):   # Returna True si es NULL
    return 1 if ptr_addr == NULL else 0

cdef inline float* _float32_cast(Image* image):
    return <float*>get_data(image)

cdef inline uint8_t* _uint8_cast(Image* image):
    return <uint8_t*>get_data(image)

cdef inline int32_t* _int32_cast(Image* image):
    return <int32_t*>get_data(image)

cdef inline object _view(void* data, int rows, int cols, int channels, object dtype, bint tridimentional):
    if tridimentional == 0:
        if dtype == _numpy.dtype("uint8"): return _numpy.asarray(<uint8_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("uint16"): return _numpy.asarray(<uint16_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("uint32"): return _numpy.asarray(<uint32_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("int8"): return _numpy.asarray(<int8_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("int16"): return _numpy.asarray(<int16_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("int32"): return _numpy.asarray(<int32_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("int64"): return _numpy.asarray(<int64_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("uint64"): return _numpy.asarray(<uint64_t[:rows, :cols]> data)
        if dtype == _numpy.dtype("float32"): return _numpy.asarray(<float[:rows, :cols]> data)
        if dtype == _numpy.dtype("float64"): return _numpy.asarray(<double[:rows, :cols]> data)

    else:
        if dtype == _numpy.dtype("uint8"): return _numpy.asarray(<uint8_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("uint16"): return _numpy.asarray(<uint16_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("uint32"): return _numpy.asarray(<uint32_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("int8"): return _numpy.asarray(<int8_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("int16"): return _numpy.asarray(<int16_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("int32"): return _numpy.asarray(<int32_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("int64"): return _numpy.asarray(<int64_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("uint64"): return _numpy.asarray(<uint64_t[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("float32"): return _numpy.asarray(<float[:rows, :cols, :channels]> data)
        if dtype == _numpy.dtype("float64"): return _numpy.asarray(<double[:rows, :cols, :channels]> data)

    raise TypeError(f"dtype no soportado: {dtype}")

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
    if _check_null_ptr(new_array):
        raise ReferenceError("PUNTERO INVALIDO, NO SE PUEDE CREAR NUEVO ARRAY")

    cdef int rows = get_rows(new_array)
    cdef int cols = get_cols(new_array)
    cdef int channels = get_channels(new_array)

    if not _validate_arr_vals(rows, cols, channels):
        return None

    cdef bint tridimentional = 1

    if dims == 0 or dims == 2:
        tridimentional = 0

    cdef object dtype = _get_arr_dtype(dtype_raw)
    cdef void* data =  get_data(new_array)
    return _view(data=data, rows=rows, cols=cols, channels=channels, dtype=dtype, tridimentional=tridimentional)

def release_array(size_t ptr_addr):
    cdef Image* img = get_image_ptr(ptr_addr)
    if _check_null_ptr(img):
        raise ReferenceError("PUNTERO INVALIDO, YA HABIA SIDO LIBERADO")

    destroy_image(img)

def compute_polygons_geometry(size_t arr_ptr, size_t img_ptr, int padd):
    cdef Image* bboxes_ptr = get_image_ptr(arr_ptr)
    if _check_null_ptr(bboxes_ptr):
        raise RuntimeError("PUNTERO DE ARRAY BBOXES CORRUPTO")

    cdef int bbox_count = get_rows(bboxes_ptr)
    cdef int32_t* bboxes_data = _int32_cast(bboxes_ptr)

    cdef Image* img = get_image_ptr(img_ptr)
    if _check_null_ptr(img):
        raise RuntimeError("PUNTERO DE IMAGEN ORIGINAL NULO")

    cdef int img_rows = get_rows(img)
    cdef int img_cols = get_cols(img)
    cdef int padding = padd

    if not _validate_arr_vals(img_rows, img_cols, padding):
        raise ValueError("ERROR EN DIMENSIONES PARA ARRAY DE BBOXES")

    cdef Image* bboxs = get_bboxes(bboxes_data, bbox_count, img_rows, img_cols, padding)

    if not _check_null_ptr(bboxes_ptr):
        print(f"REFERENCIA A BBOXES ANTERIORES NO LIBERADA_ {arr_ptr}")
        destroy_image(bboxes_ptr)
        bboxes_data = NULL

    if _check_null_ptr(bboxs):
        destroy_image(img)
        raise RuntimeError("PUNTERO DE BBOXES FILTRADOS NULO")

    cdef int bbox_filtered_count = get_rows(bboxs)
    cdef int32_t* bboxes_filtered = _int32_cast(bboxs)

    cdef Image* centroids = get_centroids(bboxes_filtered, bbox_filtered_count)
    if _check_null_ptr(centroids):
        destroy_image(bboxs)
        destroy_image(img)
        raise RuntimeError("PUNTERO DE CENTROIDES NULO, BBOXES LIBERADAS")
    
    return <size_t>centroids, <size_t>bboxs
