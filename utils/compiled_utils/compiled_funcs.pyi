def validate_text(text: str) -> bool: """Valida que un string contenga caracteres válidos y que no esté vacío"""
def validate_quant_chars(text: str) -> bool: """Valida todos si todos los caracteres de un string son cuantitativos y hay por lo menos un decimal"""
def count_cuants(text: str) -> int: """Cuenta caracteres cuantitativos (0-9, ',', '.', '$') y devuelve 0 si no existe ningún dígito."""
def space_removal(text: str) -> str: """
Normaliza espacios asumiendo UTF-8/ASCII garantizado de 1 byte por carácter.
Cero objetos intermedios. Máxima velocidad de ejecución en C.
"""
def bspace_removal(text: bytes) -> bytes: """"""
def ngram_similarity(a: bytes, b: bytes) -> float: """Calcula la similitud suave entre dos n-gramas."""
def length_penalty(a: int, b: int) -> float: """Penalización simétrica por diferencia de longitud."""