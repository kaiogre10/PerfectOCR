# bidings/image_utils/image_utils.pyx
from libc.string cimport strcmp, strlen
from libc.stddef cimport size_t
cimport numpy as cnp

cnp.import_array()

cdef int _get_dtype(const char* dtype) noexcept:
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

cdef object get_arr_dtype(const char* dtype_raw) noexcept:
    """
    Obtiene y devuelve el objeto descriptor de NumPy de forma segura.
    Devuelve None si el tipo no es válido.
    """
    cdef int intype = _get_dtype(dtype_raw)
    # En NumPy, NPY_BOOL es 0, por lo que validar 'intype < 0' es lo correcto
    if intype < 0:
        return -1
        
    return cnp.PyArray_DescrFromType(intype)

cdef size_t get_arr_dtype_size(const char* dtype) noexcept:
    cdef size_t len_type = strlen(dtype)
    
    # Validación mínima de seguridad para dtypes estándar (ej: "int8", "float32")
    if len_type < 4:
        return 999
        
    # Evaluamos el último carácter
    cdef char last = dtype[len_type - 1]
    
    if last == b'8':
        return 1
    elif last == b'6': # Termina en 16
        return 2
    elif last == b'2': # Termina en 32
        return 4
    elif last == b'4': # Termina en 64
        return 8
        
    return 999

cdef bint validate_arr_vals(int rows, int cols, int channels) noexcept:
    return rows > 0 and cols > 0 and channels > 0

cdef object array_view(void* data_view, int rows_view, int cols_view, int channels_view, const char* dtype, bint tridimentional_view) noexcept:
    cdef cnp.intp_t shape_dims[3]
    cdef int nd

    cdef int type_num = _get_dtype(dtype)
    if type_num < 0:
        return 0

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