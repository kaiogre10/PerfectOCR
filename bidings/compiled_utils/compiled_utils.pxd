# compiled_utils.pxd
from libcpp cimport bool
from libc.stddef cimport size_t
cimport numpy as cnp

cnp.import_array()

cdef size_t get_arr_dtype_size(const char* dtype) noexcept

cdef object array_view(void* data_view, int rows_view, int cols_view, int channels_view, const char* dtype, bint tridimentional_view) noexcept

cdef bool validate_arr_vals(int rows, int cols, int channels) noexcept