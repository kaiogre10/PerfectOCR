# compiled_utils/compiled_utils.pxd
cimport numpy as cnp
cnp.import_array()

cdef inline object array_view(void* data_view, int rows_view, int cols_view, int channels_view, const char* dtype, bint tridimentional_view)
    
cdef inline object get_arr_dtype(const char* dtype_raw) 

cdef inline int get_arr_dtype_size(const char* dtype_raw)

cdef inline bint validate_arr_vals(int rows, int cols, int channels)