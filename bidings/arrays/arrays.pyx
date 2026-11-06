# bidings/arrays/arrays.pyx
from libc.stddef cimport size_t # type: ignore
from libc.stdint cimport int32_t, uint8_t # type: ignore
from container cimport Buffer, get_buffer_ptr, get_rows, get_data, get_array_view, create_new_buffer, check_null_ptr, free_buffer # type: ignore
from workers cimport get_centroids, get_bboxes # type: ignore

cdef enum:
    INVALID_PTR = 0

cdef inline float* _float32_cast(Buffer* buffer):
    return <float*>get_data(buffer)

cdef inline uint8_t* _uint8_cast(Buffer* buffer):
    return <uint8_t*>get_data(buffer)

cdef inline int32_t* _int32_cast(Buffer* buffer):
    return <int32_t*>get_data(buffer)

def create_new_array(int rows, int cols, int channels, const char* dtype_raw):
    return create_new_buffer(rows, cols, channels, dtype_raw)
    
def get_array(size_t arr_ptr, const char* dtype, int dims):
    return get_array_view(arr_ptr, dtype, dims)

def release_array(size_t ptr_addr):
    return free_buffer(ptr_addr)

def compute_polygons_geometry(size_t arr_ptr, size_t img_ptr, int padd):
    cdef Buffer* bboxes_ptr = get_buffer_ptr(arr_ptr)
    if check_null_ptr(bboxes_ptr):
        return INVALID_PTR, INVALID_PTR

    cdef Buffer* img = get_buffer_ptr(img_ptr)
    if check_null_ptr(img):
        return INVALID_PTR, INVALID_PTR

    # calculate_geometry
    cdef Buffer* bboxs = get_bboxes(bboxes_ptr, img, padd)

    if not check_null_ptr(bboxes_ptr):
        free_buffer(arr_ptr)
        bboxes_data = NULL

    if check_null_ptr(bboxs):
        free_buffer(img_ptr)
        return INVALID_PTR, INVALID_PTR

    cdef int bbox_filtered_count = get_rows(bboxs)
    cdef int32_t* bboxes_filtered = _int32_cast(bboxs)

    centroids = get_centroids(bboxes_filtered, bbox_filtered_count)
    if check_null_ptr(centroids):
        return INVALID_PTR, INVALID_PTR
    
    return <size_t>centroids, <size_t>bboxs
