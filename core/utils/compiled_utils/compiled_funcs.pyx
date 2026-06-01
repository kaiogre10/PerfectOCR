# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cpython.unicode cimport PyUnicode_ReadChar

cdef inline bint _is_alnum_char(Py_UCS4 char_code) nogil:
    """Check if char is alphanumeric (48-57: 0-9, 65-90: A-Z, 97-122: a-z)"""
    return (48 <= char_code <= 57) or (65 <= char_code <= 90) or (97 <= char_code <= 122)

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