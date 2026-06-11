# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cpython.unicode cimport PyUnicode_ReadChar

cdef inline bint _is_alpha_char(Py_UCS4 char_code) nogil:
    """Check if char is alpha (65-90: A-Z, 97-122: a-z)"""
    return (65 <= char_code <= 90) or (97 <= char_code <= 122)

cdef inline bint _is_decimal_char(Py_UCS4 char_code) nogil:
    """Check if char is numeric (48-57: 0-9)"""
    return (48 <= char_code <= 57)

cdef inline bint _is_alnum_char(Py_UCS4 char_code) nogil:
    """Check if char is alphanumeric (48-57: 0-9, 65-90: A-Z, 97-122: a-z)"""
    return (_is_alpha_char(char_code)) or (_is_decimal_char(char_code))

cdef inline bint _is_cuant_char(Py_UCS4 char_code) nogil:
    """Check if char is cuantitative (44: ,, 46: ., 36: $)"""
    return (_is_decimal_char(char_code)) or ((char_code == 44) or (char_code == 36) or (char_code == 46))

cpdef bint validate_quant_chars(str text):
    """Valida todos si todos los caracteres de un string son cuantitativos"""
    cdef Py_ssize_t text_len = len(text)
    cdef Py_ssize_t i
    cdef Py_UCS4 char_code
    cdef bint valid = False

    if not text:
        return False

    for i in range(text_len):
        char_code = PyUnicode_ReadChar(text, i)
        if not _is_cuant_char(char_code):
            return False
        if _is_decimal_char(char_code):
            valid = True

    return valid

cpdef int count_cuants(str text):
    """Cuenta caracteres cuantitativos (0-9, ',', '.', '$') y devuelve 0 si no existe ningún dígito."""
    cdef Py_ssize_t text_len
    cdef Py_ssize_t i
    cdef Py_UCS4 char_code
    cdef int total_cuants = 0
    cdef bint has_decimal = False

    if not text:
        return 0

    text_len = len(text)

    for i in range(text_len):
        char_code = PyUnicode_ReadChar(text, i)

        if 48 <= char_code <= 57:
            has_decimal = True
            total_cuants += 1

        elif char_code == 44 or char_code == 46 or char_code == 36:
            total_cuants += 1

    return total_cuants if has_decimal else 0

cpdef bint validate_text(str text):
    """Valida que un string contenga caracteres válidos y que no esté vacío"""
    cdef Py_ssize_t text_len = len(text)
    cdef Py_ssize_t i
    cdef Py_UCS4 char_code

    if not text:
        return False

    for i in range(text_len):
        char_code = PyUnicode_ReadChar(text, i)
        if _is_alnum_char(char_code):
            return True

    return False