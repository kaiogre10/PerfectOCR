# compiled_utils/compiled_utils.pxd
from libc.stddef cimport size_t
cimport numpy as cnp

cnp.import_array()
    
cdef object get_arr_dtype(const char* dtype_raw) noexcept

cdef size_t get_arr_dtype_size(const char* dtype) noexcept

cdef bint validate_arr_vals(int rows, int cols, int channels) noexcept

cdef object array_view(void* data_view, int rows_view, int cols_view, int channels_view, const char* dtype, bint tridimentional_view) noexcept