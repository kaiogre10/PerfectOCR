from typing import Dict, List, Any, FrozenSet
from types import MappingProxyType
import numpy as np

DENSITY_ENCODER: Dict[str, float] = {
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "7": 7.0,
    "8": 8.0,
    "9": 9.0,
    "$": 10.0,
    ",": 11.0,
    ".": 12.0,
    "¢": 13.0,
    "/": 14.0,
    "#": 15.0,
    "%": 16.0,
    "(": 17.0,
    ")": 18.0,
    "°": 19.0,
    "—": 20.0,
    "<": 21.0,
    ">": 22.0,
    "+": 23.0,
    "-": 24.0,
    "=": 25.0,
    "*": 26.0,
    "^": 27.0,
    "\"": 28.0,
    ";": 29.0,
    "\\": 30.0,
    "|": 31.0,
    "[": 32.0,
    "]": 33.0,
    "{": 34.0,
    "}": 35.0,
    "@": 36.0,
    "&": 37.0,
    "_": 38.0,
    "¿": 39.0,
    "?": 40.0,
    "¡": 41.0,
    "!": 42.0,
    "~": 43.0,
    "`": 44.0,
    "'": 45.0,
    "™": 46.0,
    "©": 47.0,
    "®": 48.0,
    ":": 49.0,
    "Ó": 50.0,
    "Á": 51.0,
    "Ú": 52.0,
    "Ü": 53.0,
    "Ñ": 54.0,
    "W": 55.0,
    "X": 56.0,
    "Z": 57.0,
    "Y": 58.0,
    "Q": 59.0,
    "U": 60.0,
    "K": 61.0,
    "H": 62.0,
    "O": 63.0,
    "G": 64.0,
    "F": 65.0,
    "L": 66.0,
    "J": 67.0,
    "E": 68.0,
    "V": 69.0,
    "T": 70.0,
    "I": 71.0,
    "R": 72.0,
    "M": 73.0,
    "N": 74.0,
    "S": 75.0,
    "B": 76.0,
    "D": 77.0,
    "P": 78.0,
    "C": 79.0,
    "A": 80.0,
    "w": 81.0,
    "k": 82.0,
    "ü": 83.0,
    "ú": 84.0,
    "y": 85.0,
    "x": 86.0,
    "ñ": 87.0,
    "ó": 88.0,
    "q": 89.0,
    "j": 90.0,
    "é": 91.0,
    "v": 92.0,
    "f": 93.0,
    "z": 94.0,
    "h": 95.0,
    "í": 96.0,
    "g": 97.0,
    "á": 98.0,
    "p": 99.0,
    "b": 100.0,
    "u": 101.0,
    "d": 102.0,
    "m": 103.0,
    "l": 104.0,
    "t": 105.0,
    "c": 106.0,
    "n": 107.0,
    "o": 108.0,
    "i": 109.0,
    "s": 110.0,
    "r": 111.0,
    "e": 112.0,
    "a": 113.0,
    " ": 114.0
}

CHAR_FRECUENCY = np.array(
  [[1100589.0, 1.6773916e+01], #'a'
    [799520.0, 1.2185368e+01], #'e'
    [631299.0, 9.6215363e+00], #'r'
    [574065.0, 8.7492409e+00], #'s'
    [521553.0, 7.9489136e+00], #'i':
    [426607.0, 6.5018549e+00], #'o':
    [414107.0, 6.3113441e+00], #'n':
    [284463.0, 4.3354588e+00], #'c':
    [264671.0, 4.0338120e+00], #'t':
    [235861.0, 3.5947230e+00], #'l':
    [235854.0, 3.5946164e+00], #'m':
    [209819.0, 3.1978209e+00], #'d':
    [173974.0, 2.6515124e+00], #'u':
    [144072.0, 2.1957803e+00], #'b':
    [141022.0, 2.1492958e+00], #'p':
    [90359.0,  1.3771484e+00], #'g':
    [62033.0,  9.4543594e-01], #'h':
    [61667.0,  9.3985778e-01], #'z':
    [57838.0,  8.8150054e-01], #'f':
    [50633.0,  7.7169013e-01], #'v':
    [38096.0,  5.8061558e-01], #'j':
    [24330.0,  3.7081000e-01], #'q':
    [9540.0,   1.4539775e-01], #'x':
    [8678.0,   1.3226013e-01], #'y':
    [578.0,    8.8092135e-03], #'k':
    [84.0,     1.2802318e-03]] #'w'
)

VECTOR_DUMMIE: np.ndarray[Any, np.dtype[np.float32]] = np.array([
    0.997321218,
    0.969818993,
    1.011409402,
    0.966547903,
    0.928644615,
    0.745721912,
    0.702291889,
    1.027324882,
    0.9558476,
    0.967390293,
    0.726808697,
    0.827808893,
    0.935039788,
    1.000396907,
    0.957225491,
    0.999715149,
    0.969329995,
    1.034419696,
    0.940832421,
    0.900620922,
    0.974235016,
    0.996399108,
    0.986301479,
    0.996924947,
    0.985073666,
    0.9994829,
    0.996825397,
    0.868421053,
    0.807954546,
    1.0,
    0.836353079,
    1.0,
    1.0
])

FEATURES_NAME: List[str] = [
    "line_id",
    "bbox_height_inv",
    "bbox_h_dif",
    "bbox_width_inv",
    "bbox_w_dif",
    "norm_wid",
    "width_rel",
    "area_norm",
    "area_inv",
    "area_dif",
    "center_align",
    "ratio_area_norm",
    "aspcrat_inv_norm",
    "perimeter_norm",
    "perimeter_inv",
    "perimeter_dif",
    "diag_inv",
    "diag_dif",
    "angle_inv",
    "diag_norm",
    "compact",
    "prev_xmin_align",
    "prev_xmax_align",
    "next_xmin_align",
    "next_xmax_align",
    "align_prev",
    "align_next",
    "dig_margin",
    "has_quant",
    "numeric_count_norm",
    "digit_above",
    "digit_char_frec",
    "has_digit",
    "has_kf"
]

SEMATIC_TYPES_MAP: Dict[str, int] = {
    "noise": -1, # Ruido y caracteres especiales en general
    "unique": 0, # Clasificación que se le da a ciertos strings con características especificas, sirve como difereciador, casi no los hay
    "descriptive": 1, # Palabras en general, siglas aquí
    "umd": 2, # Con str cortos que suelen informar el contenido de los productos
    "code": 3, # Códigos generales tipo SKU, de identificación, contienen letras y números. Selen ser alfanuméricos
    "quantitative": 4, # Cantidades monetarias, sub clasificación de numericos
    "numeric": 5, # Strings numericos refernte a cantidades generales, no confudir con "isnumeric()"
}

UMD_CORRECTIONS: Dict[str, str] = {
    "o": "0",
    "O": "0"
}

NUMERIC_CORRECTIONS: Dict[str, str] = {
    "Q": "0",
    "D": "0",
    "C": "6",
    "(": "6",
    "q": "9",
    "O": "0",
    "o": "0",
    "I": "1",
    "i": "1",
    "|": "1",
    "!": "1",
    "¡": "1",
    "l": "1",
    "S": "5",
    "s": "5",
    # "$": "5",
    "G": "6",
    "g": "9",
    "B": "8",
    "Z": "2",
    "z": "2",
    "j": "9",
    "/": "7",
    "?": "2"
}

DESCRIPTIVE_CORRECTIONS: Dict[str, str] = {
    "$": "S",
    "è": "é",
    "ý": "y",
    "\\": "/",
}

VALID_NUM_PUNT_CHARS: FrozenSet[str] = frozenset({
    ":", ",", ".",
    "-", "%", "$",
    "+", "@",
})

not_valid_chars: FrozenSet[str] = frozenset({
    "~", "©", "®", "™", "`",
    "¬", "¨", "<", ">",
    "*", "^", "°", "'",
    "-", "_", ";", "#", 
    '`', '"', "÷", "=",
})

PUNT_CHARS: FrozenSet[str] = frozenset({
    ")", "(", "]", "[", "{", "}", "|"
    "-", "_",
    "!", "¡", "?", "¿", "'", "\\", "/"
})

VOWELS: FrozenSet[str] = frozenset({"A", "E", "I", "O", "U", "a", "e", "i", "o", "u"})
NOT_VALID_CHARS = not_valid_chars.union(PUNT_CHARS)

ESPECIAL_CHARS = NOT_VALID_CHARS.union(VALID_NUM_PUNT_CHARS)

VALID_CUANT_CHARS: FrozenSet[str] = frozenset({".", ",", "$"})
CHAR_NUM: FrozenSet[str]= frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})

CUANT_CHAR: FrozenSet[str] = frozenset(CHAR_NUM.union(VALID_CUANT_CHARS))

alone_chars: FrozenSet[str] = frozenset({"a", "e", "y", "o", "A", "E", "Y", "O"})
VALID_ALONE_CHARS: FrozenSet[str] = frozenset(CHAR_NUM.union(alone_chars))

_conversion_kf: Dict[str, int] = {
        "total_doc": 1, # 'MontoTotalDocumento'
        'total_art': 2, # TotalProductos
        'subtotal': 3,
        'folio_doc': 4,
        # "NombreCliente": 5 Se ajustará su eliminación ya que es innecesario
        "HeaderWords": 6,
        "rfc_prov": 7,
        "monto_iva": 8,
        "date_doc": 9,
        "TelefonoP": 10,
        "CorreoP": 11,
        "DirecciónP": 12
    }
CONVERSION_KF = MappingProxyType(_conversion_kf)

IMPOSIBLE_BIGRAMS: FrozenSet[str] = frozenset({"bh",
    "bq",
    "bk",
    "bw",
    "bx",
    "bz",
    "cv",
    "cw",
    "cx",
    "dc",
    "df",
    "dx",
    "fb",
    "fc",
    "fd",
    "fj",
    "fm",
    "fp",
    "fq",
    "fv",
    "fw",
    "fx",
    "fy",
    "fz",
    "gj",
    "gq",
    "gv",
    "gw",
    "gx",
    "gy",
    "hc",
    "hd",
    "hf",
    "hg",
    "hh",
    "hj",
    "hq",
    "hv",
    "hw",
    "hx",
    "hz",
    "jb",
    "jd",
    "jf",
    "jg",
    "jh",
    "jj",
    "jk",
    "jm",
    "jn",
    "jp",
    "jq",
    "jr",
    "js",
    "jt",
    "jv",
    "jw",
    "jx",
    "jy",
    "kb",
    "kf",
    "kh",
    "kk",
    "kp",
    "kq",
    "kt",
    "kv",
    "kx",
    "kz",
    "lw",
    "lx",
    "mf",
    "mg",
    "mh",
    "mj",
    "mw",
    "mx",
    "my",
    "mz",
    "nw",
    "nx",
    "pb",
    "pg",
    "pj",
    "pk",
    "pm",
    "pq",
    "pw",
    "px",
    "pz",
    "qa",
    "qb",
    "qc",
    "qd",
    "qe",
    "qf",
    "qg",
    "qh",
    "qi",
    "qj",
    "qk",
    "ql",
    "qm",
    "qn",
    "qo",
    "qp",
    "qq",
    "qr",
    "qs",
    "qt",
    "qv",
    "qw",
    "qx",
    "qy",
    "qz",
    "sx",
    "tj",
    "tk",
    "tq",
    "tv",
    "tx",
    "vb",
    "vf",
    "vg",
    "vh",
    "vj",
    "vk",
    "vm",
    "vp",
    "vq",
    "vr",
    "vs",
    "vt",
    "vv",
    "vw",
    "vx",
    "vz",
    "wc",
    "wd",
    "wf",
    "wg",
    "wj",
    "wk",
    "wl",
    "wp",
    "wq",
    "wu",
    "wv",
    "wx",
    "wy",
    "wz",
    "xb",
    "xd",
    "xg",
    "xj",
    "xk",
    "xn",
    "xr",
    "xx",
    "xz",
    "yf",
    "yg",
    "yh",
    "yj",
    "yk",
    "yq",
    "yv",
    "yx",
    "yy",
    "yz",
    "zj",
    "zv",
    "zw",
    "zx",
    "zy"})