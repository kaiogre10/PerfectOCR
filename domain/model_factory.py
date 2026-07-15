import os
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from services.output_service import load_pickle, save_pickle

logger = logging.getLogger(__name__)

class MappedMatrix:
    """Contenedor inmutable que encapsula la matriz dispersa universal """
    __slots__ = ("matrix", "matrix_ngrams")
    def __init__(self, path_dir: str, file_type: List[str]):
        self.matrix = np.load(os.path.join(path_dir, file_type[0]), mmap_mode='r', allow_pickle=False)
        self.matrix_ngrams = np.load(os.path.join(path_dir, file_type[1]), mmap_mode='r', allow_pickle=False)
        
class KeyFields:
    """Carga los arrays de los KeyFields"""
    __slots__ = ("kf_matrix", "kf_ngrams")
    def __init__(self, kf_path: str, file_type: List[str]):
        self.kf_matrix = np.load(os.path.join(kf_path, file_type[0]), mmap_mode='r', allow_pickle=False)
        self.kf_ngrams = np.load(os.path.join(kf_path, file_type[1]), mmap_mode='r', allow_pickle=False)
    
class KFIndex:
    """kf[:, 0], kw[:, 1], offset[:, -1]"""
    __slots__ = ("idx_matrix")
    def __init__(self, idx_path: str):
        self.idx_matrix = np.load(idx_path, mmap_mode='r', allow_pickle=False)
        
class MatrixFactory:
    """Componente centralizado que gestiona y mantiene en memoria persistente las matrices de control segmentadas por longitud."""
    __slots__ = (
        "models_path",
        "pkl_path",
        "idx_path",
        "idx_folder",
        "matrix_folder",
        "matrix_path",
        "kf_folder",
        "kf_path",
        "matrix_registry",
        "kf_registry",
        "model_pkl",
        "index_matrix",
        "files_list",
        "index_dict_path",
        "index_dict"
    )
    def __init__(self, config: Dict[str, Any]):
        self.models_path = config.get("wf_path", "")
        
        ngrams_name = config.get("ngrams_name", "")
        matrix_name = config.get("matrix_name", "")
        
        self.files_list = [matrix_name, ngrams_name]
        
        self.pkl_path = config.get("pkl_path", "")
        self.idx_path = config.get("kf_idx", "")
        self.index_dict_path = config.get("index_dict", "")
        
        self.matrix_path = config.get("matrix_path", "")
        self.matrix_folder = config.get("matrix_folder", "")
        
        self.kf_path = config.get("kf_path", "")
        self.kf_folder = config.get("kf_folder", "")
        
        self.matrix_registry: Dict[int, Any] = {}
        self.kf_registry: Dict[int, Any] = {}
        self._load_matrixes()
        
        self.model_pkl = {}
        self._load_model()
        
        self.index_matrix = np.asarray([])
        self.index_dict: Dict[bytes, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        """kf[:, 0], kw[:, 1], offset[:, -1]"""
        self._load_index()
    
    def _load_matrixes(self):
        """
        Escanea el directorio físico resuelto y consolida los mapas de memoria
        dentro del estado interno del objeto.
        """
        if not os.path.exists(self.matrix_folder):
            raise FileNotFoundError(f"Ruta de almacenamiento no localizada: '{self.matrix_folder}'")
        
        if not os.path.exists(self.kf_folder):
            raise FileNotFoundError(f"Ruta de KeyFields no localizada: '{self.kf_folder}'")
        
        for dirnames in os.listdir(self.models_path):    
            if self.matrix_path in dirnames:
                for item in os.listdir(self.matrix_folder):
                    full_path = os.path.join(self.matrix_folder, item)
                    # Identificación de la nomenclatura jerárquica 'longitud_{key}'
                    if os.path.isdir(full_path) and item.endswith(f"{self.matrix_path}"):
                        key_len = int(item.replace(f"_{self.matrix_path}", ""))
                        self.matrix_registry[key_len] = MappedMatrix(full_path, self.files_list)
                        continue
        
            elif self.kf_path in dirnames:
                for item in os.listdir(self.kf_folder):
                    full_path = os.path.join(self.kf_folder, item)
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
        
    def _load_index(self):
        """kf[:, 0], kw[:, 1], offset[:, -1]"""
        if not os.path.isfile(self.idx_path):
            raise FileNotFoundError(f"Indices no encontrados en: '{self.idx_path}'")
        self.index_matrix = KFIndex(self.idx_path)
        
        if not os.path.exists(self.index_dict_path):
            raise FileNotFoundError(f"Modelo no encontrado en {self.index_dict_path}")
        self.index_dict = np.load(self.index_dict_path, mmap_mode='r', allow_pickle=False)
    
    @staticmethod
    def edit_pickle_vals(config: Dict[str, Any]):
        # idx_path = config.get("kf_idx", "")
        # index_dict = config.get("index_dict", "")
        # idx_matrix = np.load(idx_path)
        #
        # idx_word = idx_matrix[1:, :2]
        # key_words = idx_matrix[1:, 2:24]
        #
        # ends = key_words[:, -1]
        # mapped_words: Dict[bytes, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        # for i in range(key_words.shape[0]):
        #     bkw = key_words[i, :ends[i]].tobytes()
        #     byteword = bkw.decode('ascii')
        #     mapped_words[byteword] = idx_word[i]
        #
        # # np.savez(index_dict, **mapped_words)
        
        pkl_path = config.get("pkl_path", "")
        model_pkl: Dict[str, Any] = load_pickle(pkl_path, 'rb')
        if not isinstance(model_pkl, dict):
            raise ValueError("El pickle no tiene el formato esperado (dict).")
            
        all_ngrams: Dict[bytes, Tuple[int, Dict[int, np.ndarray[Any, np.dtype[np.uint8]]]]] = model_pkl.get("all_ngrams", {})
        
        ball_ngrams: Dict[bytes, Tuple[int, Dict[int, List[bytes]]]] = {}
        for word, word_ngrams in all_ngrams.items():
            array_grams: Dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            
            for lens, ngrams in word_ngrams[1].items():
                total_ngrmas = ngrams.shape[0]
                list_ngrams: List[bytes] = []
                for i in range(total_ngrmas):
                    plain_ngrams = ngrams[i].tobytes()
                    list_ngrams.append(plain_ngrams)
                    # logger.info(f"{plain_ngrams}")
                array_grams[lens] = list_ngrams
                # array_grams[lens] = np.frombuffer(plain_ngrams, dtype=np.uint8).reshape(len(ngrams), lens)
                
            ball_ngrams[word] = (word_ngrams[0], array_grams)
            
        # logger.info("\n"f"{all_ngrams}")
        # logger.info("\n"f"{ball_ngrams}")
        
        # model_pkl["ball_ngrams"] = ball_ngrams
        # del model_pkl["all_ngrams"]
        # index_dict = np.load(index_dict_path, mmap_mode='r')
        #
        # bindex_dict: Dict[bytes, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        # for name, matrix in index_dict.items():
        #     bname = name.encode('ascii')
        #     logger.info(f"{bname}: {matrix}")
        #     bindex_dict[bname] = matrix
        #
        # logger.info(f"\n"f"{bindex_dict}")
        # np.savez(index_dict_path, bindex_dict)
        # # model_pkl["ball_ngrams"] = ball_ngrams
        #
        # try:
        #     save_pickle(model_pkl, pkl_path, 'wb')
        # except Exception as e:
        #     logger.error(f"ERROR GUARDADNO PICKLE: {e}", exc_info=True)
        #     return False
        
        # return True