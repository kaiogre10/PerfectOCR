import os
import numpy as np
from typing import Dict, Any
import logging
import pickle

logger = logging.getLogger(__name__)

class MappedMatrix:
    """
    Contenedor inmutable que encapsula los componentes binarios 
    de una matriz dispersa indexada en disco.
    """
    def __init__(self, path_dir: str):
        # self.data = np.load(os.path.join(path_dir, "data.npy"), mmap_mode='r')
        # self.indices = np.load(os.path.join(path_dir, "indices.npy"), mmap_mode='r')
        # self.indptr = np.load(os.path.join(path_dir, "indptr.npy"), mmap_mode='r')
        # self.shape = np.load(os.path.join(path_dir, "mtx_shape.npy"), mmap_mode='r')
        self.matrix = np.load(os.path.join(path_dir, "matrix.npz"), mmap_mode='r')
        self.matrix_ngrams = np.load(os.path.join(path_dir, "ngrams.npy"), mmap_mode='r')

class MatrixManager:
    """
    Componente centralizado que gestiona y mantiene en memoria persistente
    las matrices de control segmentadas por longitud.
    """
    def __init__(self, project_root: str, config: Dict[str, Any]):
        # Extracción del subdirectorio destino desde el objeto de configuración
        self.project_root = project_root
        self.model_path = config["wf_model_path"]
        self.matrix_folder = config.get("matrix_path", "")
        output_path = config["data_path"]
        self.matrix_path = os.path.join(project_root, *output_path)

        self.matrix_registry: Dict[int, Any] = {}
        self._load_matrixes()

        self.model_pkl: Dict[str, Any] = {}
        self._load_model()

    def _load_matrixes(self):
        """
        Escanea el directorio físico resuelto y consolida los mapas de memoria 
        dentro del estado interno del objeto.
        """
        if not os.path.exists(self.matrix_path):
            raise FileNotFoundError(f"Ruta de almacenamiento no localizada: '{self.matrix_path}'")

        for item in os.listdir(self.matrix_path):
            full_path = os.path.join(self.matrix_path, item)
            # logger.info(f"FULL: {ruta_completa}: {os.path.isdir(ruta_completa)}\n"f"item: {item}, {self.matrix_folder}")
            # Identificación de la nomenclatura jerárquica 'longitud_{key}'
            if os.path.isdir(full_path) and item.endswith(f"{self.matrix_folder}"):
                key_len = int(item.replace(f"_{self.matrix_folder}", ""))
                
                try:
                    # Persistencia del mapeo virtual en el diccionario de control
                    self.matrix_registry[key_len] = MappedMatrix(full_path)
                    # logger.info(f"{self._registro_matrices[llave_rango]}")
                except FileNotFoundError:
                    # Omisión de directorios con escrituras binarias corruptas o incompletas
                    continue
        self.matrix_registry

    def _load_model(self):
        try:
            model_path = os.path.join(self.project_root, *self.model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
            with open(model_path, "rb") as f:
                self.model_pkl: Dict[str, Any] = pickle.load(f)
            if not isinstance(self.model_pkl, dict):  # type: ignore
                raise ValueError("El pickle no tiene el formato esperado (dict).")
            self.model_pkl
        except ExceptionGroup as e:
            logger.error(f"Error al cargar el modelo {e}", exc_info=True)
            raise
