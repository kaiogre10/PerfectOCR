# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport uint8_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.unicode cimport PyUnicode_GET_LENGTH, PyUnicode_1BYTE_DATA, PyUnicode_FromStringAndSize
from cpython.bytes cimport PyBytes_GET_SIZE, PyBytes_AS_STRING

cdef inline bint _is_space_char(uint8_t char_code) noexcept nogil:
    return (char_code == 32)

cdef inline bint _is_alpha_char(uint8_t char_code) noexcept nogil:
    """Check if char is alpha (65-90: A-Z, 97-122: a-z)"""
    return (65 <= char_code <= 90) or (97 <= char_code <= 122)

cdef inline bint _is_alnum_char(uint8_t char_code) noexcept nogil:
    """Check if char is alphanumeric"""
    return _is_alpha_char(char_code) or _is_decimal_char(char_code)

cdef inline bint _is_decimal_char(uint8_t char_code) noexcept nogil:
    """Check if char is numeric (48-57: 0-9)"""
    return (48 <= char_code <= 57)

cdef inline bint _is_cuant_spec(uint8_t char_code) noexcept nogil:
    """Check if char is cuantitative (44: ',', 46: '.', 36: '$')"""
    return (char_code == 46) or (char_code == 36) or (char_code == 44)

cdef inline bint _is_cuant_char(uint8_t char_code) noexcept nogil:
    """Check if char is cuantitative (44: ,, 46: ., 36: $, decimal)"""
    return _is_decimal_char(char_code) or _is_cuant_spec(char_code)

cdef inline bint _validate_quant_chars_c(const uint8_t* s, Py_ssize_t text_len) noexcept nogil:
    cdef Py_ssize_t i
    cdef uint8_t char_code
    cdef bint valid = 0

    for i in range(text_len):
        char_code = s[i]
        if not _is_cuant_char(char_code):
            valid = 0
            break
        if _is_decimal_char(char_code):
            valid = 1

    return valid

def validate_quant_chars(str text) -> bool:
    """Valida todos si todos los caracteres de un string son cuantitativos y hay por lo menos un decimal"""
    if not text:
        return False

    cdef Py_ssize_t text_len = PyUnicode_GET_LENGTH(text)
    cdef const uint8_t* s = <const uint8_t*>PyUnicode_1BYTE_DATA(text)

    return _validate_quant_chars_c(s, text_len)

def validate_quant_bytes(bytes text) -> bool:
    """Valida si todos los bytes son cuantitativos y hay por lo menos un decimal."""
    if not text:
        return False

    cdef Py_ssize_t text_len = PyBytes_GET_SIZE(text)
    cdef const uint8_t* s = <const uint8_t*>PyBytes_AS_STRING(text)

    return _validate_quant_chars_c(s, text_len)

cdef inline int _count_cuants_c(const uint8_t* s, Py_ssize_t text_len) noexcept nogil:
    cdef Py_ssize_t i
    cdef uint8_t char_code
    cdef int total_cuants = 0
    cdef bint has_decimal = 0

    for i in range(text_len):
        char_code = s[i]
        if _is_decimal_char(char_code):
            has_decimal = 1
            total_cuants += 1
        elif _is_cuant_spec(char_code):
            total_cuants += 1

    return total_cuants if has_decimal else 0

def count_cuants(str text) -> int:
    """Cuenta caracteres cuantitativos (0-9, ',', '.', '$') y devuelve 0 si no existe ningún dígito."""
    if not text:
        return 0

    cdef Py_ssize_t text_len = PyUnicode_GET_LENGTH(text)
    cdef const uint8_t* s = <const uint8_t*>PyUnicode_1BYTE_DATA(text)

    return _count_cuants_c(s, text_len)

def count_cuant_bytes(bytes text) -> int:
    """Cuenta bytes cuantitativos (0-9, ',', '.', '$') y devuelve 0 si no existe ningún dígito."""
    if not text:
        return 0

    cdef Py_ssize_t text_len = PyBytes_GET_SIZE(text)
    cdef const uint8_t* s = <const uint8_t*>PyBytes_AS_STRING(text)

    return _count_cuants_c(s, text_len)

cdef inline float _ngram_similarity(const unsigned char* a, const unsigned char* b, Py_ssize_t length) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t matches = 0

    for i in range(length):
        if a[i] == b[i]:
            matches += 1

    return <float>matches / <float>length

def ngram_similarity(bytes a, bytes b) -> float:
    """Calcula la similitud suave entre dos n-gramas."""
    cdef Py_ssize_t length = PyBytes_GET_SIZE(a)
    cdef const unsigned char* ptr_a = <const unsigned char*>PyBytes_AS_STRING(a)
    cdef const unsigned char* ptr_b = <const unsigned char*>PyBytes_AS_STRING(b)

    return _ngram_similarity(ptr_a, ptr_b, length)

cdef inline float _length_penalty_c(int a, int b) noexcept nogil:
    cdef int num_min = a if a < b else b
    cdef int num_max = b if a < b else a

    return <float>num_min / <float>num_max

def length_penalty(int a, int b) -> float:
    """Penalización simétrica por diferencia de longitud."""
    return _length_penalty_c(a, b)

cdef inline bint _validate_text_c(const uint8_t* s, Py_ssize_t text_len) noexcept nogil:
    cdef Py_ssize_t i
    cdef bint is_valid = 0

    for i in range(text_len):
        if _is_alnum_char(s[i]):
            is_valid = 1
            break

    return is_valid

def validate_text(str text) -> bool:
    """Valida que un string contenga caracteres válidos y que no esté vacío"""
    if not text:
        return False

    cdef Py_ssize_t text_len = PyUnicode_GET_LENGTH(text)
    cdef const uint8_t* s = <const uint8_t*>PyUnicode_1BYTE_DATA(text)

    return _validate_text_c(s, text_len)

def validate_bytes(bytes text) -> bool:
    """Valida que bytes ASCII contengan caracteres válidos y que no estén vacíos."""
    if not text:
        return False

    cdef Py_ssize_t text_len = PyBytes_GET_SIZE(text)
    cdef const uint8_t* s = <const uint8_t*>PyBytes_AS_STRING(text)

    return _validate_text_c(s, text_len)

cdef Py_ssize_t _space_removal_core(const uint8_t* s, Py_ssize_t n, char* out) nogil:
    """
    Función cdef pura. Recibe puntero de entrada + longitud, escribe en 'out'
    y retorna la longitud de salida. Asume ASCII (1 byte por carácter).
    """
    cdef Py_ssize_t orig_len = n

    if n == 1:
        if _is_space_char(s[0]):
            return 0
        out[0] = <char>s[0]
        return 1

    # Recortar espacios iniciales
    while n > 0 and _is_space_char(s[0]):
        s += 1
        n -= 1

    # Recortar espacios finales
    while n > 0 and s[n - 1] == 32:
        n -= 1

    if n == 0:
        return 0

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef uint8_t c

    while i < n:
        c = s[i]
        out[j] = <char>c
        j += 1
        if _is_space_char(c):
            while i + 1 < n and s[i + 1] == 32:
                i += 1
        i += 1

    return j

def space_removal_bytes(bytes text) -> bytes:
    """Wrapper para bytes (ASCII garantizado)."""
    if not text:
        return b""
    
    cdef Py_ssize_t n = len(text)
    if n == 1:
        return text if _is_space_char(text[0]) else b""
    
    cdef const uint8_t* s = <const uint8_t*>text
    cdef char* out = <char*>PyMem_Malloc(n + 1)
    if out == NULL:
        raise MemoryError()
    cdef Py_ssize_t j
    try:
        j = _space_removal_core(s, n, out)
        if j == n:
            return text
        return out[:j]
    finally:
        PyMem_Free(out)

def space_removal(str text) -> str:
    """Wrapper para str (ASCII garantizado, 1 byte por carácter)."""
    if not text:
        return ""
    cdef Py_ssize_t orig_len = PyUnicode_GET_LENGTH(text)
    if orig_len == 0:
        return ""

    if orig_len == 1:
        return text if text != " " else ""
    
    cdef const uint8_t* s = <const uint8_t*>PyUnicode_1BYTE_DATA(text)
    cdef char* out = <char*>PyMem_Malloc(orig_len + 1)
    
    if out == NULL:
        raise MemoryError()
    cdef Py_ssize_t j
    try:
        j = _space_removal_core(s, orig_len, out)
        if j == orig_len:
            return text
        return PyUnicode_FromStringAndSize(out, j)
    finally:
        PyMem_Free(out)