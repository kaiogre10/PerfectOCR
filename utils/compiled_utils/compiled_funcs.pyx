# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from cpython.version cimport PY_MAJOR_VERSION
from libc.stdlib cimport malloc, free
from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_FromStringAndSize

# Acceso directo a la estructura interna de CPython para strings ASCII/CompactBytes
cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object o) except NULL
    object PyUnicode_FromStringAndSize(const char *v, Py_ssize_t len)
    bytes PyBytes_FromString(char *v) except NULL

cdef inline bint _is_alpha_char(char char_code) nogil:
    """Check if char is alpha (65-90: A-Z, 97-122: a-z)"""
    return (65 <= char_code <= 90) or (97 <= char_code <= 122)

cdef inline bint _is_decimal_char(char char_code) nogil:
    """Check if char is numeric (48-57: 0-9)"""
    return (48 <= char_code <= 57)

cdef inline bint _is_alnum_char(char char_code) nogil:
    """Check if char is alphanumeric (48-57: 0-9, 65-90: A-Z, 97-122: a-z)"""
    return (_is_alpha_char(char_code)) or (_is_decimal_char(char_code))

cdef inline bint _is_cuant_char(char char_code) nogil:
    """Check if char is cuantitative (44: ,, 46: ., 36: $)"""
    return (_is_decimal_char(char_code)) or ((char_code == 44) or (char_code == 36) or (char_code == 46))

cpdef bint validate_quant_chars(str text):
    """Valida todos si todos los caracteres de un string son cuantitativos y hay por lo menos un decimal"""
    cdef Py_ssize_t text_len = len(text)
    cdef Py_ssize_t i
    cdef char* s
    cdef char char_code
    cdef bint valid = False

    if not text:
        return False

    s = PyUnicode_AsUTF8(text)
    for i in range(text_len):
        char_code = s[i]
        if not _is_cuant_char(char_code):
            return False
        if _is_decimal_char(char_code):
            valid = True

    return valid

cpdef int count_cuants(str text):
    """Cuenta caracteres cuantitativos (0-9, ',', '.', '$') y devuelve 0 si no existe ningún dígito."""
    cdef Py_ssize_t text_len
    cdef Py_ssize_t i
    cdef char* s
    cdef char char_code
    cdef int total_cuants = 0
    cdef bint has_decimal = False

    if not text:
        return 0

    text_len = len(text)
    s = PyUnicode_AsUTF8(text)

    for i in range(text_len):
        char_code = s[i]

        if 48 <= char_code <= 57:
            has_decimal = True
            total_cuants += 1

        elif char_code == 44 or char_code == 46 or char_code == 36:
            total_cuants += 1

    return total_cuants if has_decimal else 0

cdef inline float _ngram_similarity(const unsigned char* a, const unsigned char* b, int text_len) noexcept nogil:
    cdef Py_ssize_t i
    cdef int matches = 0

    for i in range(text_len):
        if a[i] == b[i]:
            matches += 1

    # Forzar división flotante para evitar truncamiento a 0
    return <float>matches / <float>text_len

def ngram_similarity(bytes texta, bytes textb):
    cdef int text_len = len(texta)
    if text_len == 0 or text_len != len(textb):
        raise ValueError("Los textos deben tener longitudes iguales y mayores a cero.")

    # Pasamos los punteros directos a la función inline
    return _ngram_similarity(texta, textb, text_len)

cdef inline float _length_penalty_c(int a, int b) nogil:
    return min(a, b) / max(a, b)

def length_penalty(a: int, b: int) -> float:
    return _length_penalty_c(a, b)

cpdef bint validate_text(str text):
    """Valida que un string contenga caracteres válidos y que no esté vacío"""
    cdef Py_ssize_t text_len = len(text)
    cdef Py_ssize_t i
    cdef char* s
    cdef char char_code

    if not text:
        return False

    s = PyUnicode_AsUTF8(text)
    for i in range(text_len):
        char_code = s[i]
        if _is_alnum_char(char_code):
            return True

    return False

cpdef str space_removal(str text):
    """
    Normaliza espacios asumiendo UTF-8/ASCII garantizado de 1 byte por carácter.
    Cero objetos intermedios. Máxima velocidad de ejecución en C.
    """
    if text is None:
        return ""

    cdef Py_ssize_t n = len(text)
    if n == 0:
        return ""

    # Obtener el puntero nativo de C directo del objeto Python (Sin copiar)
    cdef char* s = PyUnicode_AsUTF8(text)
    
    if n == 1:
        return "" if s[0] == 32 else text

    # --- Fast Path: Detección rápida ---
    cdef Py_ssize_t i = 0
    cdef bint has_changes = False
    cdef bint in_space = False

    if s[0] == 32:  # Leading space
        has_changes = True
    else:
        while i < n:
            if s[i] == 32:
                if in_space or (i == n - 1):  # Espacio duplicado o trailing space
                    has_changes = True
                    break
                in_space = True
            else:
                in_space = False
            i += 1

    # Si el string ya está limpio, retornamos la referencia original (0 alocaciones)
    if not has_changes:
        return text

    # --- Allocating Buffer Temporal ---
    cdef char* buffer = <char*>malloc(n * sizeof(char))
    if buffer == NULL:
        raise MemoryError()

    cdef Py_ssize_t j = 0
    in_space = False
    i = 0

    while i < n:
        if s[i] == 32:
            if j == 0 or in_space:
                i += 1
                continue
            buffer[j] = 32
            j += 1
            in_space = True
        else:
            buffer[j] = s[i]
            j += 1
            in_space = False
        i += 1

    # Remover trailing space residual si existe
    if j > 0 and buffer[j - 1] == 32:
        j -= 1

    if j == 0:
        free(buffer)
        return ""

    # Construir el nuevo objeto Python str directamente desde el buffer de C
    cdef str result = PyUnicode_FromStringAndSize(buffer, j)
    
    free(buffer)
    return result

cpdef bytes bspace_removal(bytes text):
    if text is None:
        return b""
    cdef Py_ssize_t n = len(text)
    if n == 0:
        return b""
    cdef char* s = text  # conversión directa e implícita, sin función extra
    
    if n == 1:
        return b"" if s[0] == 32 else text
    
    cdef Py_ssize_t i = 0
    cdef bint has_changes = False
    cdef bint in_space = False
    if s[0] == 32:
        has_changes = True
    else:
        while i < n:
            if s[i] == 32:
                if in_space or (i == n - 1):
                    has_changes = True
                    break
                in_space = True
            else:
                in_space = False
            i += 1
    
    if not has_changes:
        return text
    
    cdef char* buffer = <char*>malloc(n * sizeof(char))
    if buffer == NULL:
        raise MemoryError()
    cdef Py_ssize_t j = 0
    in_space = False
    i = 0
    while i < n:
        if s[i] == 32:
            if j == 0 or in_space:
                i += 1
                continue
            buffer[j] = 32
            j += 1
            in_space = True
        else:
            buffer[j] = s[i]
            j += 1
            in_space = False
        i += 1
    
    if j > 0 and buffer[j - 1] == 32:
        j -= 1
    if j == 0:
        free(buffer)
        return b""
    
    cdef bytes result = buffer[:j]  # construcción directa desde char*
    free(buffer)
    return result
    