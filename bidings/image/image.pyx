# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, embedsignature=True
from libc.stddef cimport size_t
from libc.string cimport memcpy
from libc.stdint cimport uint8_t
from libcpp cimport bool
from ..container.container cimport Buffer, get_buffer_ptr, create_new_buffer, get_array_view, check_null_ptr, get_rows, get_cols, get_size, get_data, free_buffer, get_channels, get_all_buffer_addrs_size
from ..container.workers cimport load_image, rotate_image

cdef inline size_t NULL_PYTHON_PTR = 0

def load(const char* filepath):
    cdef Buffer* buffer = load_image(filepath)
    if check_null_ptr(buffer):
        return NULL_PYTHON_PTR
    
    return <size_t>buffer

def get_image_array(size_t ptr_addr, const char* dtype_raw):
    return get_array_view(ptr_addr, dtype_raw, 0)

cdef inline void _release_img(size_t ptr_addr):
    free_buffer(ptr_addr)

def release_img(size_t ptr_addr):
    _release_img(ptr_addr)
    return None

def rotate(size_t ptr_addr, double angle, double center_x, double center_y):
    cdef Buffer* source = get_buffer_ptr(ptr_addr)
    if check_null_ptr(source):
        return NULL_PYTHON_PTR

    cdef Buffer* rotated

    rotated = rotate_image(source, angle, center_x, center_y)
    if check_null_ptr(rotated):
        return NULL_PYTHON_PTR

    return <size_t>rotated

def copy_image(size_t ptr_addr, const char* dtype):
    cdef Buffer* buffer = get_buffer_ptr(ptr_addr)
    if check_null_ptr(buffer):
        return NULL_PYTHON_PTR

    cdef int cols = get_cols(buffer)
    cdef int rows = get_rows(buffer)
    cdef int channels = 1

    cdef size_t copy = create_new_buffer(rows, cols, channels, dtype)
    cdef Buffer* new_buffer = get_buffer_ptr(copy)

    if check_null_ptr(new_buffer):
        return NULL_PYTHON_PTR

    cdef void* old_data = get_data(buffer)
    cdef void* new_data = get_data(new_buffer)
    cdef size_t total_bytes = get_size(buffer)

    memcpy(new_data, old_data, total_bytes)
    return <size_t>new_buffer

def get_crops_ptr(size_t grouped_crops, size_t total_buffers):
    """No acepta individuales"""
    cdef Buffer* grouped_buffers = get_buffer_ptr(grouped_crops)  # Crops
    cdef size_t* buffer_list_size = get_all_buffer_addrs_size(grouped_buffers, total_buffers)   # Indice de punteros
    buffer_list = [buffer_list_size[i] for i in range(total_buffers)]   # LIsta de punteros para python

    if len(buffer_list) != total_buffers:
        return NULL_PYTHON_PTR

    cdef Buffer* last_array = get_buffer_ptr(buffer_list[total_buffers - 1])
    if check_null_ptr(last_array):
        return NULL_PYTHON_PTR

    if not _validate_image(last_array):
        return NULL_PYTHON_PTR

    return buffer_list

cdef inline bool _validate_image(Buffer* last_array):
    cdef int rows = get_rows(last_array)
    cdef int cols = get_cols(last_array)
    cdef int channels = get_channels(last_array)
    cdef size_t size = get_size(last_array)
    cdef size_t vol = <size_t>rows * <size_t>cols * <size_t>channels
    return vol == (size // sizeof(uint8_t))