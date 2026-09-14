import os
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from utils.file_handler import load_pickle

logger = logging.getLogger(__name__)

class MappedMatrix:
    """Contenedor inmutable que encapsula la matriz dispersa universal """
    __slots__ = ("matrix", "matrix_ngrams")
    def __init__(self, path_dir: str, file_type: List[str]):
        self.matrix = np.load(os.path.join(path_dir, file_type[0]), mmap_mode='r', allow_pickle=False)
        self.matrix_ngrams = np.load(os.path.join(path_dir, file_type[1]), mmap_mode='r', allow_pickle=False)

    @property
    def matrix_data(self) -> np.ndarray[Any, Any]:
        return self.matrix["data"]

    @property
    def matrix_indices(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        return self.matrix["indices"]

    @property
    def matrix_indptr(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        return self.matrix["indptr"]

    @property
    def matrix_shape(self) -> Tuple[int, int]:
        shape = self.matrix["shape"]
        return int(shape[0]), int(shape[1])

    def sum_cross_points(self, row_indices: np.ndarray[Any, Any], col_indices: np.ndarray[Any, Any]) -> Tuple[int, float]:
        if row_indices.size == 0 or col_indices.size == 0:
            return 0, 0.0

        indptr = self.matrix_indptr
        indices = self.matrix_indices
        data = self.matrix_data
        _, total_cols = self.matrix_shape

        rows = np.asarray(row_indices, dtype=np.intp)
        cols = np.asarray(col_indices, dtype=np.intp)

        if np.any(rows < 0) or np.any(rows + 1 >= indptr.shape[0]):
            raise IndexError("row_indices fuera de rango para la matriz dispersa.")
        if np.any(cols < 0) or np.any(cols >= total_cols):
            raise IndexError("col_indices fuera de rango para la matriz dispersa.")

        col_mask = np.zeros(total_cols, dtype=bool)
        col_mask[cols] = True

        sim_count = 0
        sim_sum = 0.0
        for row in rows:
            start = int(indptr[row])
            end = int(indptr[row + 1])
            if start == end:
                continue

            row_cols = indices[start:end]
            selected = col_mask[row_cols]
            if not np.any(selected):
                continue

            row_data = data[start:end][selected]
            sim_count += int(row_data.size)
            sim_sum += float(np.sum(row_data, dtype=np.float32))

        return sim_count, sim_sum

class KeyFields:
    """Carga los arrays de los KeyFields"""
    __slots__ = ("kf_matrix", "kf_ngrams")
    def __init__(self, kf_path: str, file_type: List[str]):
        self.kf_matrix = np.load(os.path.join(kf_path, file_type[0]), mmap_mode='r', allow_pickle=False)
        self.kf_ngrams = np.load(os.path.join(kf_path, file_type[1]), mmap_mode='r', allow_pickle=False)

class MatrixFactory:
    """Componente centralizado que gestiona y mantiene en memoria persistente las matrices de control segmentadas por longitud."""
    __slots__ = (
        "models_path",
        "pkl_path",
        "matrix_folder",
        "matrix_path",
        "kf_folder",
        "kf_path",
        "matrix_registry",
        "kf_registry",
        "model_pkl",
        "files_list"
    )
    def __init__(self, config: Dict[str, Any]):
        self.models_path: str = config.get("wf_path", "")
        
        ngrams_name = config.get("ngrams_name", "")
        matrix_name = config.get("matrix_name", "")
        
        self.files_list = [matrix_name, ngrams_name]
        
        self.pkl_path = config.get("pkl_path", "")
        
        self.matrix_path = config.get("matrix_path", "")
        self.matrix_folder: str = config.get("matrix_folder", "")
        
        self.kf_path = config.get("kf_path", "")
        self.kf_folder: str = config.get("kf_folder", "")
        
        self.matrix_registry: Dict[int, Any] = {}
        self.kf_registry: Dict[int, Any] = {}
        self._load_matrixes()
        
        self.model_pkl = {}
        self._load_model()
    
    def _load_matrixes(self):
        """
        Escanea el directorio físico resuelto y consolida los mapas de memoria
        dentro del estado interno del objeto.
        """
        if not os.path.exists(self.matrix_folder):
            raise FileNotFoundError(f"Ruta de almacenamiento no localizada: '{self.matrix_folder}'")
        
        if not os.path.exists(self.kf_folder):
            raise FileNotFoundError(f"Ruta de KeyFields no localizada: '{self.kf_folder}'")
        
        for _, dirnames in enumerate(os.listdir(self.models_path)):
            if self.matrix_path in dirnames:
                for item in os.listdir(self.matrix_folder):
                    full_path: str = os.path.join(self.matrix_folder, item)
                    # Identificación de la nomenclatura jerárquica 'longitud_{key}'
                    if os.path.isdir(full_path) and item.endswith(f"{self.matrix_path}"):
                        key_len = int(item.replace(f"_{self.matrix_path}", ""))
                        self.matrix_registry[key_len] = MappedMatrix(full_path, self.files_list)
                        continue
        
            elif self.kf_path in dirnames:
                for item in os.listdir(self.kf_folder):
                    full_path: str = os.path.join(self.kf_folder, item)
                    if os.path.isdir(full_path) and item.endswith(f"{self.kf_path}"):
                        key_len = int(item.replace(f"_{self.kf_path}", ""))
                        self.kf_registry[key_len] = KeyFields(full_path, self.files_list)
                        continue
            else:
                continue
                
        self.matrix_registry
        self.kf_registry
        
    def _load_model(self):
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Modelo no encontrado en {self.pkl_path}")
        self.model_pkl = load_pickle(self.pkl_path, 'rb')
        # with open(self.pkl_path, "rb") as f:
        #     self.model_pkl = pickle.load(f)
        if not self.model_pkl:
            raise ModuleNotFoundError("ERROR EN LA CARGA DEL PICKLE")
        if not isinstance(self.model_pkl, dict):
            raise ValueError("El pickle no tiene el formato esperado (dict).")

    @staticmethod
    def edit_pickle_vals(config: Dict[str, Any]):
        pkl_path = config.get("pkl_path", "")
        model_pkl: Dict[str, Any] = load_pickle(pkl_path, 'rb')
        if not isinstance(model_pkl, dict): # type: ignore
            raise ValueError("El pickle no tiene el formato esperado (dict).")
            
        all_ngrams: Dict[bytes, Tuple[int, Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]]] = model_pkl.get("all_ngrams", {})
        
        ball_ngrams: Dict[bytes, Tuple[int, Dict[int, List[bytes]]]] = {}
        for word, word_ngrams in all_ngrams.items():
            array_grams: Dict[int, List[bytes]] = {}
            
            for lens, ngrams in word_ngrams[1].items():
                total_ngrmas = ngrams.shape[0]
                list_ngrams: List[bytes] = []

                for i in range(total_ngrmas):
                    plain_ngrams = ngrams[i].tobytes()
                    list_ngrams.append(plain_ngrams)
                
                array_grams[lens] = list_ngrams
                # array_grams[lens] = np.frombuffer(plain_ngrams, dtype=np.uint8).reshape(len(ngrams), lens)
                
            ball_ngrams[word] = (word_ngrams[0], array_grams)

        # try:
        #     save_pickle(model_pkl, pkl_path, 'wb')
        # except Exception as e:
        #     logger.error(f"ERROR GUARDADNO PICKLE: {e}", exc_info=True)
        #     return False
        
        # return True