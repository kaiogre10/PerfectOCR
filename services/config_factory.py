import os
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MatrizSimilitudMapeada:
    """
    Contenedor inmutable que encapsula los componentes binarios 
    de una matriz dispersa indexada en disco.
    """
    def __init__(self, ruta_directorio: str):
        self.data = np.load(os.path.join(ruta_directorio, "data.npy"), mmap_mode='r')
        self.indices = np.load(os.path.join(ruta_directorio, "indices.npy"), mmap_mode='r')
        self.indptr = np.load(os.path.join(ruta_directorio, "indptr.npy"), mmap_mode='r')
        self.shape = np.load(os.path.join(ruta_directorio, "mtx_shape.npy"))

class MotorMatricesControl:
    """
    Componente centralizado que gestiona y mantiene en memoria persistente
    las matrices de control segmentadas por longitud.
    """
    def __init__(self, project_root: str, config: Dict[str, Any]):
        # Extracción del subdirectorio destino desde el objeto de configuración
        # Ejemplo esperado de estructura: config = {"matrix_folder": "matrices_control"}
        self.matrix_folder = config.get("matrix_path", "")
        output_path = config["data_path"]
        # Resolución de la ruta absoluta del repositorio de almacenamiento
        self.matrix_path = os.path.join(project_root, *output_path)
        # logger.info(f"{self.matrix_path}")
        # Estructura de datos persistente para el almacenamiento indexado de las instancias
        self.registro_matrices: Dict[int, Any] = {}
        
        # Ejecución mandatoria del mapeo en el arranque de la instancia
        self._inicializar_carga_persistente()

    def _inicializar_carga_persistente(self):
        """
        Escanea el directorio físico resuelto y consolida los mapas de memoria 
        dentro del estado interno del objeto.
        """
        if not os.path.exists(self.matrix_path):
            raise FileNotFoundError(f"Ruta de almacenamiento no localizada: '{self.matrix_path}'")

        for item in os.listdir(self.matrix_path):
            ruta_completa = os.path.join(self.matrix_path, item)
            # logger.info(f"FULL: {ruta_completa}: {os.path.isdir(ruta_completa)}\n"f"item: {item}, {self.matrix_folder}")
            # Identificación de la nomenclatura jerárquica 'longitud_{key}'
            if os.path.isdir(ruta_completa) and item.endswith(f"{self.matrix_folder}"):
                llave_rango = int(item.replace(f"_{self.matrix_folder}", ""))
                
                try:
                    # Persistencia del mapeo virtual en el diccionario de control
                    self.registro_matrices[llave_rango] = MatrizSimilitudMapeada(ruta_completa)
                    # logger.info(f"{self._registro_matrices[llave_rango]}")
                except FileNotFoundError:
                    # Omisión de directorios con escrituras binarias corruptas o incompletas
                    continue
        self.registro_matrices