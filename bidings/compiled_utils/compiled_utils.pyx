# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, embedsignature=True
from libc.string cimport strcmp, strlen
from libc.stddef cimport size_t
from libcpp cimport bool
cimport numpy as cnp

cnp.import_array()

cdef int _max_size = 20000000

cdef inline int _get_dtype(const char* dtype) noexcept nogil:
    if strcmp(dtype, b"uint8") == 0: return cnp.NPY_UINT8
    elif strcmp(dtype, b"uint16") == 0: return cnp.NPY_UINT16
    elif strcmp(dtype, b"uint32") == 0: return cnp.NPY_UINT32
    elif strcmp(dtype, b"uint64") == 0: return cnp.NPY_UINT64
    elif strcmp(dtype, b"int8") == 0: return cnp.NPY_INT8
    elif strcmp(dtype, b"int16") == 0: return cnp.NPY_INT16
    elif strcmp(dtype, b"int32") == 0: return cnp.NPY_INT32
    elif strcmp(dtype, b"int64") == 0: return cnp.NPY_INT64
    elif strcmp(dtype, b"float32") == 0: return cnp.NPY_FLOAT32
    elif strcmp(dtype, b"float64") == 0: return cnp.NPY_FLOAT64
    return -1

cdef inline size_t  _get_dtype_size(char last) noexcept nogil:
    if last == b"8": # Termina en 8 (uint8, int8), ascci = 56
        return 1
    elif last == b"6": # Termina en 16 (uint16, int16), ascii=54
        return 2
    elif last == b"2": # Termina en 32, ascii=50
        return 4
    elif last == b"4": # Termina en 64, ascii=52
        return 8
    elif last == b"e":
        return sizeof(size_t)
    return 0

cdef size_t get_arr_dtype_size(const char* dtype) noexcept:
    cdef size_t len_type = strlen(dtype)

    # Validación mínima de seguridad para dtypes estándar (ej: "int8", "float32")
    if 3 >= len_type:
        return 0
    # Evaluamos el último carácter
    cdef char last = dtype[len_type - 1]
    return _get_dtype_size(last)
    
cdef object array_view(void* data_view, int rows_view, int cols_view, int channels_view, const char* dtype, bint tridimentional_view) noexcept:
    cdef cnp.intp_t[3] shape_dims
    cdef cnp.intp_t nd

    cdef int type_num = _get_dtype(dtype)
    if type_num < 0:
        return cnp.PyArray_SimpleNewFromData(1, &nd, cnp.NPY_UINT8, NULL)

    # Asignar la geometría del arreglo usando el flag'tridimentional_view'
    shape_dims[0] = rows_view
    shape_dims[1] = cols_view
    if tridimentional_view:
        shape_dims[2] = channels_view
        nd = 3
    else:
        nd = 2
        
    # 3. Retornamos la vista del ndarray generada directamente desde la API de C
    return cnp.PyArray_SimpleNewFromData(nd, shape_dims, type_num, data_view)

cdef bool validate_arr_vals(int rows, int cols, int channels) noexcept:
    return (rows > 0 and cols > 0 and channels > 0) and (rows * cols * channels < _max_size)