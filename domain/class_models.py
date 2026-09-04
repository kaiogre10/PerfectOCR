from enum import IntEnum, StrEnum

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