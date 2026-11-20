from enum import IntEnum, StrEnum, Enum

class SemantiClass(IntEnum):
    NOISE = -1 # Ruido y caracteres especiales en general
    UNIQUE = 0 # Clasificación que se le da a ciertos strings con características especificas, sirve como difereciador, casi no los hay
    DESCRIPTIVE = 1 # Palabras en general
    UMD = 2 # Con str cortos que suelen informar el contenido de los productos
    CODE = 3 # Códigos generales tipo SKU, de identificación, contienen letras y números. Selen ser alfanuméricos
    QUANTITATIVE = 4 # Cantidades monetarias, sub clasificación de numericos
    NUMERIC = 5 # Strings numericos refernte a cantidades generales, no confudir con "isnumeric()"

class KeyField(IntEnum):
    total_doc = 1
    total_art = 2
    subtotal = 3
    folio_doc = 4
    # "nombrecliente": 5 se ajustará su eliminación ya que es innecesario
    header = 6
    rfc_prov = 7
    monto_iva = 8
    date_doc = 9
    telefonop = 10
    correop = 11
    direccionp = 12

class DataKeys(StrEnum):
    id_cliente = "id_cliente"
    nombre_cliente = "nombre_cliente"
    total_doc = "total_doc"
    total_art = "total_art"
    total_cal = "total_cal"
    art_calc = "art_calc" #
    id_proveedor = "id_proveedor"
    giro = "giro"
    proveedor_norm = "proveedor_norm"
    fecha_captura = "fecha_captura"
    rfc_prov = "rfc_prov"
    id_registro = "id_registro"
    folio_doc = "folio_doc"
    date_doc = "date_doc"
    monto_iva = "monto_iva"
    subtotal = 'subtotal'

    cantidad_art = "cantidad_art"
    precio_unitario = "precio_unitario"
    costo_tran = "costo_tran"
    producto_norm = "producto_norm"

class StageKeys(StrEnum):
    IMGPREP_KEY = "image_preparation_stager"
    PREPRO_KEY = "preprocessing_stager"
    OCR_KEY = "ocr_stager"
    VECT_KEY = "vectorization_stager"

class NumberStr(StrEnum):
    ZEROS_CUANT = ".00"
    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"

class GeoDict(StrEnum):
    POLYGON_ID = "polygon_id"
    XMIN = "xmin"
    XMAX = "xmax"
    CX = "cx"
    CY = "cy"
    OCR_TEXT = "ocr_text"
    SEMANTIC_CLASIFICATION = "semantic_clasification"
    LINEAL_ID = "lineal_id"
    POLYGON_IDS = "polygon_ids"
    WORDS = "words"

class DataMathDict(StrEnum):
    DF_COPY = "df_copy"
    CUT_POLYGONS = "cut_polygons"
    DEC_ROWS_IDS = "dec_rows_ids"
    DEC_COLS = "dec_cols"
    ARITH_COLS_IDS = "arith_cols_ids"
    TEXT_COL_TEMP = "text_col_temp"
    COMPLETE = "complete"
    ROWS_IDX = "rows_idx"
    COLS_IDX = "cols_idx"
    TEXT = "text"
    TEXT_COL = "text_col"
    SEMANTIC_CLASIFICATION = "semantic_clasification"
    MAX_IDX = "max_idx"

class TypeModels(StrEnum):
    ASCII = "ascii"
    UTF8 = "utf-8"
    UTF16_CSHARP = "utf16le"

class OSModels(Enum):
    """[0]Bibliotecas, [1] Librerias"""
    WINDOWS = {".dll": ".lib"}
    LINUX = {".so": ".a"}
    MACOS = {".dylib": ".a"}

class StringsModels(StrEnum):
    NO_MANGER = "NO_MANAGER"
    TIME_MASK = "Tiempo: "
    OTSU = "otsu"
    ADAPTATIVE_GAUSS = "adaptative_gauss"
    SAUVOLA = "sauvola"
    ADAPTATIVE_MEAN = "adaptative_mean"

class Metrics(StrEnum):
    code = 'code' 
    comment = 'comment' 
    blank = 'blank'
    total = 'total'
    functions = 'functions'
    files = 'files'
    python = 'python'
    yaml = 'yaml'
    cplusplus = 'cplusplus'
    cython = 'cython'
    sql = "sql"

class Params(StrEnum):
    CHAR_NGRAMS = "char_ngrams"
    WF_PATH = "wf_path"
    NGRAMS_NAME = "ngrams_name"
    MATRIX_NAME = "matrix_name"
    PKL_PATH = "pkl_path"
    MATRIX_PATH = "matrix_path"
    MATRIX_FOLDER = "matrix_folder"
    KF_PATH = "kf_path"
    KF_FOLDER = "kf_folder"
    KEY_FIELD = "key_field"
    KEY_WORD = "key_word"
    SIMILARITY = "similarity"
    TEXT = "text"
    NORM_OCR_TEXT = "norm_ocr_text"
    START = "start"
    END = "end"

class Workers(StrEnum):
    IMAGE_LOADER = "image_loader"
    ANGLE_CORRECTOR = "angle_corrector"
    INK_ENHANCEMENT = "ink_enhancement"
    GEOMETRY_DETECTOR = "geometry_detector"
    MOIRE = "moire"
    SP = "sp"
    GAUSS = "gauss"
    CLAHE = "clahe"
    SHARP = "sharp"
    PADDLE_WRAPPER = "paddle_wrapper"
    TEXT_REFINER = "text_refiner"
    DATA_FINDER = "data_finder"
    LINEAL = "lineal"
    COS_SIM = "cos_sim"
    TABLE_STRUCTURER = "table_structurer"
    MATH_MAX = "math_max"
    COLLECTOR = "collector"

# 1. Subclase mínima de str en C (__slots__ = ()): 0 bytes extra por atributo,
# class _MetaStr(str):
#     __slots__ = ()
#     def __bool__(self) -> NoReturn:
#         raise TypeError("No se permite evaluación booleana; usa 'is None' o 'is not None'.")

#     def __eq__(self, other: object) -> bool:
#         if isinstance(other, str):
#             return str.__eq__(self, other)
#         raise TypeError("KindofSearch solo admite comparaciones de igualdad contra str.")

#     def __ne__(self, other: object) -> bool:
#         if isinstance(other, str):
#             return str.__ne__(self, other)
#         raise TypeError("KindofSearch solo admite comparaciones de desigualdad contra str.")

#     __hash__ = str.__hash__

# class _KindofMeta(type):
#     def __bool__(cls) -> NoReturn:
#         raise TypeError("Evaluación booleana prohibida sobre KindofSearch.")

#     def __eq__(cls, other: object) -> bool:
#         if other is cls:
#             return True
#         if isinstance(other, str):
#             return False
#         raise TypeError("KindofSearch solo admite comparaciones contra sí mismo o str.")

#     def __ne__(cls, other: object) -> bool:
#         return not cls.__eq__(other)

#     def __hash__(cls) -> int:
#         return type.__hash__(cls)

#     def __setattr__(cls, name: str, value: object) -> NoReturn:
#         raise AttributeError("KindofSearch es inmutable.")

#     def __delattr__(cls, name: str) -> NoReturn:
#         raise AttributeError("KindofSearch es inmutable.")

#     @property
#     def OP_1(cls) -> _MetaStr:
#         return _POOL[0]

#     @property
#     def OP_2(cls) -> _MetaStr:
#         return _POOL[1]

#     @property
#     def OP_3(cls) -> _MetaStr:
#         return _POOL[2]

#     @property
#     def OP_4(cls) -> _MetaStr:
#         return _POOL[3]

#     @property
#     def OP_5(cls) -> _MetaStr:
#         return _POOL[4]

# class KindofSearch(metaclass=_KindofMeta):
#     """Dominio cerrado e inmutable de estados de búsqueda sin instanciación.

#     Expone 5 constantes preasignadas (`OP_1` a `OP_5`) respaldadas por una
#     subclase mínima de `str` en C, más la propia clase como centinela y `None`
#     como sexto estado base válido.

#     Reglas del contrato en tiempo de ejecución:
#         - **Cero alocaciones:** No es instanciable (`KindofSearch()` lanza `TypeError`).
#           Sus propiedades devuelven referencias preasignadas en memoria.
#         - **Inmutabilidad estricta:** Cualquier intento de asignación o borrado de
#           atributos lanza `AttributeError`.
#         - **Sin evaluación booleana:** Ejecutar `bool(x)` o `if x:` lanza `TypeError`.
#           La existencia del estado debe comprobarse exclusivamente por identidad
#           (`is None` o `is not None`).
#         - **Comparaciones acotadas:** Los operadores `==` y `!=` solo operan contra
#           cadenas (`str`) o la propia clase; cualquier otro tipo (`int`, `bool`,
#           `None`, etc.) lanza `TypeError`.
#     """
#     __slots__ = ()
#     def __new__(cls) -> NoReturn:
#         raise TypeError("KindofSearch no es instanciable; usa sus propiedades o None.")

# Preasignación única al cargar el módulo
# _POOL: tuple[_MetaStr, ...] = (
#     _MetaStr("filename"),
#     _MetaStr("no_extensions"),
#     _MetaStr("extensions"),
#     _MetaStr("prefix"),
#     _MetaStr("subfix"),
# )
# type SearchKind = Literal["filename", "no_extensions", "extensions", "prefix", "subfix"]