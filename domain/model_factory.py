import os
import numpy as np
from typing import Dict, Any, List
import logging
import pickle
from config.config_loader import load_pickle, save_pickle

logger = logging.getLogger(__name__)

class MappedMatrix:
    """Contenedor inmutable que encapsula la matriz dispersa universal """
    __slots__ = ("matrix", "matrix_ngrams")
    def __init__(self, path_dir: str):
        self.matrix = np.load(os.path.join(path_dir, "matrix.npz"), mmap_mode='r', allow_pickle=False)
        self.matrix_ngrams = np.load(os.path.join(path_dir, "ngrams.npy"), mmap_mode='r', allow_pickle=False)

class KeyFields:
    """Carga los arrays de los KeyFields"""
    __slots__ = ("kf_matrix", "kf_ngrams")
    def __init__(self, kf_path: str):
        self.kf_matrix = np.load(os.path.join(kf_path, "matrix.npz"), mmap_mode='r', allow_pickle=False)
        self.kf_ngrams = np.load(os.path.join(kf_path, "ngrams.npy"), mmap_mode='r', allow_pickle=False)

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
        "index_matrix"
    )
    def __init__(self, config: Dict[str, Any]):
        self.models_path = config.get("wf_path", "")
        
        self.pkl_path = config.get("pkl_path", "")
        self.idx_path = config.get("kf_idx", "")

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
                        self.matrix_registry[key_len] = MappedMatrix(full_path)
                        continue

            elif self.kf_path in dirnames:
                for item in os.listdir(self.kf_folder):
                    full_path = os.path.join(self.kf_folder, item)
                    if os.path.isdir(full_path) and item.endswith(f"{self.kf_path}"):
                        key_len = int(item.replace(f"_{self.kf_path}", ""))
                        self.kf_registry[key_len] = KeyFields(full_path)
                        continue
            else:
                continue
                
        self.matrix_registry
        self.kf_registry
        
    def _load_model(self):
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Modelo no encontrado en {self.pkl_path}")
        with open(self.pkl_path, "rb") as f:
            self.model_pkl = pickle.load(f)
            if not self.model_pkl:
                raise pickle.UnpicklingError("ERROR EN LA CARGA DEL PICKLE")
        if not isinstance(self.model_pkl, dict):
            raise ValueError("El pickle no tiene el formato esperado (dict).")
    
    def _load_index(self):
        """kf[:, 0], kw[:, 1], offset[:, -1]"""
        if not os.path.isfile(self.idx_path):
            raise FileNotFoundError(f"Indices no encontrados en: '{self.idx_path}'")
        self.index_matrix = KFIndex(self.idx_path)

    @staticmethod
    def edit_pickle_vals(pkl_path: str):
        model_pkl = load_pickle(pkl_path, 'rb')
        if not isinstance(model_pkl, dict):
            raise ValueError("El pickle no tiene el formato esperado (dict).")
            
        logger.info(f"MODEL KEYS: {model_pkl.keys()}")
        _noise_words = model_pkl["noise_words"]
        _noise_filter = model_pkl.get("noise_filter", {})
        _noise_grams: List[Dict[int, List[str]]] = _noise_filter["noise_grams"]
        model_pkl["noise_cands"] = [(word, _noise_grams[i]) for i, word in enumerate(_noise_words) if word and i < len(_noise_grams)]
        # noise_words: List[str] = model_pkl["noise_words"]
        # sorted_noise_words = sorted(noise_words, key=len, reverse=True)
        # logger.info(f"NOISE WIRDS: {noise_words}\n"f"SORTED: {sorted_noise_words}")
        # model_pkl["noise_words"] = sorted_noise_words
        # noise_filter = model_pkl.get("noise_filter", {})
        # noise_array: List[np.ndarray[Any, np.dtype[np.uint8]]] = noise_filter["noise_array"]
        # noise_grams: List[Dict[int, List[str]]] = noise_filter["noise_grams"]
        # all_ngrams: Dict[str, Tuple[int, Dict[int, List[str]]]] = model.get("all_ngrams", {})
        #
        # total_ngrams = len(all_ngrams.items())
        #
        # global_words: List[str] = model["global_words"]
        # max_len = max((len(s) for s in global_words)) + 3
        # logger.info(f"total keywords: {global_words}, {max_len}")
        #
        # kfw = 0
        # kf_indices = np.zeros((total_ngrams, max_len), dtype=np.uint8, order='C')
        # logger.info(f"SHAPE IDX = {kf_indices.shape}")
        # global_words: List[str] = []
        # last_seen = None
        # for i, key_words in enumerate(all_ngrams.items()):
        #     key_word = key_words[0]
        #     key_field = key_words[1][0]
        #     # all_word_ngrams = key_words[1][1]
        #     global_words.append(key_word)
        #     kf_indices[i, 0] = key_field
        #
        #     if key_field == last_seen:  # type: ignore
        #         kfw += 1
        #         kf_indices[i, 1] = kfw
        #
        #     else:
        #         kfw = 0
        #         last_seen = key_field
        #         kf_indices[i, 1] = kfw
        #
        #     word_int = key_word.encode('ascii', 'ignore')
        #     cand_vect = np.frombuffer(word_int, np.uint8)
        #     len_word = cand_vect.size
        #
        #     size = np.arange(2, (len_word + 2), dtype=np.uint8)
        #     kf_indices[i, size] = cand_vect
        #     kf_indices[i, -1] = len_word
            
            # for len_ngram, ngramas in all_word_ngrams.items():
            #     logger.info("\n"f"{len_ngram}_ngrams: {ngramas}")
                # new_all_ngrams[key_word] = (key_field, all_word_ngrams)
        
        # idx_array_path = os.path.join(pkl_path, kf_idx)
        #
        # np.save(idx_array_path, kf_indices)
        # model["global_words"] = global_words
        #
        try:
            save_pickle(model_pkl, pkl_path, 'wb')
        except Exception as e:
            logger.error(f"ERROR GUARDADNO PICKLE: {e}", exc_info=True)
            return False
        return True
