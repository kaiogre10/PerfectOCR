# bidings/image_utils/image_utils.pyx
from libc.string cimport strcmp, strlen
cimport numpy as cnp

cnp.import_array()

cdef inline int get_dtype(const char* dtype):
    if strcmp(dtype, b"uint8") == 0: type_num = cnp.NPY_UINT8
    elif strcmp(dtype, b"uint16") == 0: type_num = cnp.NPY_UINT16
    elif strcmp(dtype, b"uint32") == 0: type_num = cnp.NPY_UINT32
    elif strcmp(dtype, b"uint64") == 0: type_num = cnp.NPY_UINT64
    elif strcmp(dtype, b"int8") == 0: type_num = cnp.NPY_INT8
    elif strcmp(dtype, b"int16") == 0: type_num = cnp.NPY_INT16
    elif strcmp(dtype, b"int32") == 0: type_num = cnp.NPY_INT32
    elif strcmp(dtype, b"int64") == 0: type_num = cnp.NPY_INT64
    elif strcmp(dtype, b"float32") == 0: type_num = cnp.NPY_FLOAT32
    elif strcmp(dtype, b"float64") == 0: type_num = cnp.NPY_FLOAT64
    else:
        return 0

cdef inline object get_arr_dtype(const char* dtype_raw):
    """
    Obtiene y devuelve el objeto descriptor de NumPy de forma segura.
    Devuelve None si el tipo no es válido.
    """
    cdef int intype = get_dtype(dtype_raw)
    # En NumPy, NPY_BOOL es 0, por lo que validar 'intype < 0' es lo correcto
    if intype < 1:
        return None
        
    return cnp.PyArray_DescrFromType(intype)

cdef inline int get_arr_dtype_size(const char* dtype):
    cdef size_t len = strlen(dtype)
    
    # Validación mínima de seguridad para dtypes estándar (ej: "int8", "float32")
    if len < 4:
        return 0
        
    # Evaluamos el último carácter
    cdef char last = dtype[len - 1]
    
    if last == b'8':
        return 1
    elif last == b'6': # Termina en 16
        return 2
    elif last == b'2': # Termina en 32
        return 4
    elif last == b'4': # Termina en 64
        return 8
        
    return 0

cdef inline bint validate_arr_vals(int rows, int cols, int channels):
    return rows > 0 and cols > 0 and channels > 0

cdef inline object array_view(void* data_view, int rows_view, int cols_view, int channels_view, const char* dtype, bint tridimentional_view):
    cdef cnp.intp_t shape_dims[3]
    cdef int nd
    cdef int type_num = get_dtype(dtype)
    if type_num == 0:
        return cnp.PyArray_EMPTY(0, shape_dims, 31, 0)

    # 2. Asignamos la geometría del arreglo usando tu bandera 'tridimentional'
    shape_dims[0] = rows_view
    shape_dims[1] = cols_view
    if tridimentional_view:
        shape_dims[2] = channels_view
        nd = 3
    else:
        nd = 2
        
    # 3. Retornamos la vista del ndarray generada directamente desde la API de C
    return cnp.PyArray_SimpleNewFromData(nd, shape_dims, type_num, data_view)

