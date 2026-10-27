# bidings/container/container.pxd
from libc.stddef cimport size_t
from bidings.compiled_utils.compiled_utils cimport validate_arr_vals, get_arr_dtype_size, array_view

cdef extern from "buffer_container.hpp":
    cppclass Buffer:
        Buffer(int rows, int cols, int channels, size_t dtype) noexcept
        void* data()
        const void* read_only_data() const
        int rows() const
        int cols() const
        int channels() const
        size_t size() const

    void destroy_buffer(Buffer* buffer) noexcept

cdef inline bint check_null_ptr(Buffer* ptr_addr) noexcept:
    return ptr_addr == NULL or <size_t>ptr_addr == 0

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

cdef inline size_t create_new_buffer(int rows, int cols, int channels, const char* dtype_raw) noexcept:
    if not validate_arr_vals(rows, cols, channels):
        return 0

    cdef size_t bad_dtype = get_arr_dtype_size(dtype_raw)
    if bad_dtype < 1:
        return 0

    cdef Buffer* b = new Buffer(rows, cols, channels, bad_dtype)
    return <size_t>b

cdef inline object get_array_view(size_t arr_ptr, const char* dtype, int dims) noexcept:
    cdef Buffer* new_array = get_buffer_ptr(arr_ptr)
    if check_null_ptr(new_array):
        return 0

    cdef int rows = get_rows(new_array)
    cdef int cols = get_cols(new_array)
    cdef int channels = get_channels(new_array)

    if not validate_arr_vals(rows, cols, channels):
        return 0

    cdef bint tridimentional_view = 1

    if dims == 0 or dims == 2:
        tridimentional_view = 0

    cdef void* data = get_data(new_array)
    return array_view(data, rows, cols, channels, dtype, tridimentional_view)