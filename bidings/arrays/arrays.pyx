# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, embedsignature=True
from ..container.container cimport Buffer, get_buffer_ptr, get_array_view, create_new_buffer, check_null_ptr, free_buffer, recover_class
from libc.stddef cimport size_t
from ..container.workers cimport get_bboxes

cdef enum:
    INVALID_PTR = 0

def create_new_array(int rows, int cols, int channels, const char* dtype_raw):
    return create_new_buffer(rows, cols, channels, dtype_raw)
    
def get_array(size_t arr_ptr, const char* dtype, int dims):
    return get_array_view(arr_ptr, dtype, dims)

def release_array(size_t ptr_addr):
    free_buffer(ptr_addr)
    return None

def compute_polygons_geometry(size_t arr_ptr, size_t image_ptr, int padding):
    cdef Buffer* bboxes_ptr = get_buffer_ptr(arr_ptr)
    if check_null_ptr(bboxes_ptr):
        free_buffer(image_ptr)
        raise TypeError(f"PUNTERO A BBOXES VACIO")

    cdef Buffer* img = get_buffer_ptr(image_ptr)
    if check_null_ptr(img):
        free_buffer(arr_ptr)
        raise TypeError(f"PUNTERO A FULL IMG VACIO")

    cdef size_t total_buffers = 0
    cdef size_t* total_buffers_ref = &total_buffers
    # calculate_geometry
    cdef Buffer* geometry = get_bboxes(bboxes_ptr, img, padding, total_buffers_ref)
    if check_null_ptr(geometry):
        free_buffer(image_ptr)
        raise TypeError(f"PUNTERO PACK GEOMETRIA NULL")

    if total_buffers < 3:
        free_buffer(image_ptr)
        raise TypeError(f"CONTEO POR REFERENCIA ERRONEO")

    cdef Buffer* bboxs = recover_class(total_buffers-2, geometry)
    cdef Buffer* centroids = recover_class(total_buffers-1, geometry)

    if check_null_ptr(bboxs) or check_null_ptr(centroids):
        free_buffer(image_ptr)
        raise TypeError("PUNTEROS NULOS DE BBOXES O CENTROIDES")

    cdef size_t total_buffs = total_buffers-3
    crop_ptr = get_all_buffer_addrs(geometry, total_buffs)
    return <size_t>centroids, <size_t>bboxs, crop_ptr

cdef get_all_buffer_addrs(Buffer* aos, size_t total_buffs):
    cdef size_t n = total_buffs
    cdef list result = [0] * n
    cdef size_t i = 0

    while i < n:
        size_buffer = aos.buffer_addr_sized(i)
        result[i]= size_buffer
        i += 1
    return result