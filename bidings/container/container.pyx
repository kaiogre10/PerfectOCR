# bidings/container/container.pyx
from container cimport Buffer, destroy_buffer
from libc.stddef cimport size_t
from bidings.compiled_utils.compiled_utils cimport validate_arr_vals, get_arr_dtype_size, array_view

cdef void* null_ptr = NULL
cdef size_t NULL_PYTHON_PTR = 0

cdef inline bint check_null_ptr(Buffer* ptr_addr) noexcept:
    return True if ptr_addr == NULL and <size_t>ptr_addr > 0 else False
 
cdef inline Buffer* get_buffer_ptr(size_t ptr_addr) noexcept:
    if ptr_addr < 1:
        return <Buffer*>0
        
    return <Buffer*>ptr_addr

cdef inline void free_buffer(size_t ptr_addr) noexcept:
    cdef Buffer* buff = get_buffer_ptr(ptr_addr)
    if not check_null_ptr(buff):
        destroy_buffer(buff)

cdef inline int get_rows(Buffer* buff_ptr) noexcept:
    return buff_ptr.rows()
    
cdef inline int get_cols(Buffer* buff_ptr) noexcept:
    return buff_ptr.cols()

cdef inline int get_channels(Buffer* buff_ptr) noexcept:
    return buff_ptr.channels()

cdef inline size_t get_size(Buffer* buff_ptr) noexcept:
    return buff_ptr.size()

cdef inline void* get_data(Buffer* buff_ptr) noexcept:
    return buff_ptr.data()

cdef inline Buffer* _new_buffer(int rows, int cols, int channels, size_t dtype) noexcept:
    return new Buffer(rows, cols, channels, dtype)

cdef inline object get_array_view(size_t arr_ptr, const char* dtype_raw, int dims):
    cdef Buffer* new_array = get_buffer_ptr(arr_ptr)
    if check_null_ptr(new_array):
        return NULL_PYTHON_PTR

    cdef int rows = get_rows(new_array)
    cdef int cols = get_cols(new_array)
    cdef int channels = get_channels(new_array)

    cdef bint is_valid = validate_arr_vals(rows, cols, channels)
    if is_valid == 0:
        return NULL_PYTHON_PTR

    cdef bint tridimentional = 1

    if dims == 0 or dims == 2:
        tridimentional = 0
    
    cdef void* data = get_data(new_array)
    cdef object view = array_view(data, rows, cols, channels, dtype_raw, tridimentional)
    if not view:
        return NULL_PYTHON_PTR

    return view

cdef inline size_t create_new_buffer(int rows, int cols, int channels, const char* dtype_raw)  noexcept:
    if not validate_arr_vals(rows, cols, channels):
        return NULL_PYTHON_PTR

    cdef bad_dtype = get_arr_dtype_size(dtype_raw)
    if bad_dtype < 1:
        return NULL_PYTHON_PTR
    
    cdef size_t dtype = <size_t>bad_dtype
    cdef Buffer* new_array
    new_array = _new_buffer(rows, cols, channels, dtype)

    return <size_t>new_array