# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint8_t
from container.container cimport Buffer, get_buffer_ptr, get_rows, get_cols, get_data, get_array_view, create_new_buffer, check_null_ptr, free_buffer
from container.workers cimport get_centroids, get_bboxes

cdef inline float* _float32_cast(Buffer* buffer):
    return <float*>get_data(buffer)

cdef inline uint8_t* _uint8_cast(Buffer* buffer):
    return <uint8_t*>get_data(buffer)

cdef inline int32_t* _int32_cast(Buffer* buffer):
    return <int32_t*>get_data(buffer)

def create_new_array(int rows, int cols, int channels, const char* raw_dtype):
    return create_new_buffer(rows, cols, channels, raw_dtype)
    
def get_array(size_t arr_ptr, const char* dtype_raw, int dims):
    return get_array_view(arr_ptr, dtype_raw, dims)

def release_array(size_t ptr_addr):
    return free_buffer(ptr_addr)

def compute_polygons_geometry(size_t arr_ptr, size_t img_ptr, int padd):
    cdef Buffer* bboxes_ptr = get_buffer_ptr(arr_ptr)
    if check_null_ptr(bboxes_ptr):
        return <size_t>0, <size_t>0

    cdef int bbox_count = get_rows(bboxes_ptr)
    cdef int32_t* bboxes_data = _int32_cast(bboxes_ptr)

    cdef Buffer* img = get_buffer_ptr(img_ptr)
    if check_null_ptr(img):
        return <size_t>0, <size_t>0

    cdef int img_rows = get_rows(img)
    cdef int img_cols = get_cols(img)
    cdef int padding = padd

    if not (0 < img_rows) and (0 < img_cols) and (0 < padding):
        return <size_t>0, <size_t>0

    cdef Buffer* bboxs = get_bboxes(bboxes_data, bbox_count, img_rows, img_cols, padding)

    if not check_null_ptr(bboxes_ptr):
        free_buffer(arr_ptr)
        bboxes_data = NULL

    if check_null_ptr(bboxs):
        free_buffer(img_ptr)
        return <size_t>0, <size_t>0

    cdef int bbox_filtered_count = get_rows(bboxs)
    cdef int32_t* bboxes_filtered = _int32_cast(bboxs)

    centroids = get_centroids(bboxes_filtered, bbox_filtered_count)
    if check_null_ptr(centroids):
        return <size_t>0, <size_t>0
    
    return <size_t>centroids, <size_t>bboxs
