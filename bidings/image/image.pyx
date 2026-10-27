# bidings/image/image.pyx
from libc.stddef cimport size_t
from libc.string cimport memcpy
from container cimport Buffer, get_buffer_ptr, create_new_buffer, get_array_view, check_null_ptr, get_rows, get_cols, get_size, get_data, free_buffer
from workers cimport load_image, rotate_image

cdef inline size_t NULL_PYTHON_PTR = 0

def load(str filepath):
    cdef Buffer* buffer = load_image(filepath.encode('utf-8')) # type: ignore
    if check_null_ptr(buffer):
        return NULL_PYTHON_PTR
    
    return <size_t>buffer

def get_image_array(size_t ptr_addr, const char* dtype_raw):
    return get_array_view(ptr_addr, dtype_raw, 0)
        
def release_img(size_t ptr_addr):
    free_buffer(ptr_addr)

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
    
    
    copy = create_new_buffer(rows, cols, channels, dtype)

    cdef Buffer* new_buffer = get_buffer_ptr(copy)

    if check_null_ptr(new_buffer):
        return NULL_PYTHON_PTR

    cdef void* old_data = get_data(buffer)
    cdef void* new_data = get_data(new_buffer)
    cdef size_t total_bytes = get_size(buffer)

    memcpy(new_data, old_data, total_bytes)
    return <size_t>new_buffer