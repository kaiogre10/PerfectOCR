# Diccionario final.py
# Estructura detallada del diccionario antes de llegar al fragmentador
document_dict = {
    # --- METADATOS DEL DOCUMENTO (creados por ImageCleaner._resolutor y _quick_enhance) ---
    "metadata": {
        "doc_name": "ejemplo.pdf",          # Nombre del archivo original
        "formato": "PDF",                   # Formato del documento (JPEG, PNG, PDF, etc.)
        "img_dims": {                       # Dimensiones de la imagen
            "width": 1240,                  # Ancho en píxeles
            "height": 1754                  # Alto en píxeles
        },
        "dpi": 300,                         # Resolución del documento en DPI (None si no disponible)
        "fecha_creacion": "2023-04-15 10:30:45"  # Fecha de creación del archivo (None si no disponible)
    },
    
    # --- POLÍGONOS DETECTADOS (creados por Deskewer._detect_geometry) ---
    # Diccionario donde la clave es el polygon_id y el valor contiene toda la información del polígono
    "polygons": {
        "poly_0000": {
            "polygon_id": "poly_0000",      # Identificador único del polígono
            "geometry": {
                "polygon_coords": [         # Coordenadas de los puntos del polígono (de PaddleOCR)
                    [120.0, 230.0],         # Esquina superior izquierda
                    [420.0, 230.0],         # Esquina superior derecha
                    [420.0, 260.0],         # Esquina inferior derecha
                    [120.0, 260.0]          # Esquina inferior izquierda
                ],
                "bounding_box": [120.0, 230.0, 420.0, 260.0],  # [xmin, ymin, xmax, ymax]
                "centroid": [270.0, 245.0], # Centro del polígono [x, y]
                "width": 300.0,             # Ancho del polígono (xmax - xmin)
                "height": 30.0              # Alto del polígono (ymax - ymin)
            },
            "line_id": "line_0001",         # ID de línea asignado por LineReconstructor._reconstruct_lines
            "cropped_img": "imagen_recortada_0000",  # Imagen recortada por PolygonExtractor._extract_individual_polygons
            "padding_coords": [115, 225, 425, 265],  # Coordenadas después de aplicar padding [x1, y1, x2, y2]
            "was_fragmented": False,        # Añadido por PolygonFragmentator._intercept_polygons
            "perimeter": 580.0              # Perímetro de la palabra (añadido por PolygonFragmentator._intercept_polygons)
        },
        "poly_0001": {
            "polygon_id": "poly_0001",
            "geometry": {"..."
            }
        },
    },    
}
    # --- GEOMETRÍA DE LÍNEAS (creada por LineReconstructor._reconstruct_lines) ---
    
    
# --- GEOMETRÍA DE LÍNEAS (creada por LineReconstructor._reconstruct_lines) ---
# Este diccionario no está en el diccionario principal, sino que se obtiene por separado
lines_geometry = {
    "line_0001": {
        "bounding_box": [120.0, 230.0, 1080.0, 260.0],  # Bbox de toda la línea
        "centroid": [600.0, 245.0],                     # Centro de la línea
        "polygon_ids": ["poly_0000", "poly_0001", "poly_0002"]  # IDs de polígonos en esta línea
    },
    "line_0002": {
        "bounding_box": [120.0, 280.0, 800.0, 310.0],
        "centroid": [460.0, 295.0],
        "polygon_ids": ["poly_0003", "poly_0004", "poly_0005"]
    }
}
    # --- POLÍGONOS BINARIZADOS (creados por Binarizator._binarize_polygons) ---
    # No está en el diccionario principal sino en un diccionario separado
binarized_polygons = {  # Este es un diccionario separado polygon_id -> imagen binarizada (np.ndarray)
    "poly_0000": imagen_binarizada_0000,  # np.ndarray   # type: ignore
    "poly_0001": imagen_binarizada_0001,  # type: ignore 
    "poly_0002": imagen_binarizada_0002   # type: ignore
}

# --- OBSERVACIONES IMPORTANTES ---
# 1. El diccionario principal 'document_dict' se va enriqueciendo a lo largo del flujo
# 2. La clave 'polygons' ahora es un DICCIONARIO donde cada polygon_id es una clave
# 3. Cada componente añade su información al diccionario existente, no crea uno nuevo
# 4. Las claves "binarized_polygons" y "lines_geometry" son diccionarios separados 
#    que se obtienen con métodos específicos (_get_lines_geometry, _get_polygons_copy)
# 5. El fragmentador opera sobre este diccionario y añade dos campos nuevos:
#    - was_fragmented: bool (True si fue fragmentado, False si no)
#    - perimeter: float (perímetro de la palabra/palabras)
# 6. Los IDs se reasignan secuencialmente manteniendo el orden de lectura del texto
# 7. Cuando un polígono se fragmenta, los polígonos posteriores se desplazan automáticamente