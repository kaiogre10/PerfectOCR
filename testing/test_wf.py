from testing.testing_data.wf_data import base_queries2
from typing import List, Dict, Optional, Any
import time
import os
import numpy as np
from services.log_service import log_simple, basic_exc_logger
from app.models_builder import ModelsBuilder

class TestingManger:
    def __init__(self, project_root: str, models_config: Dict[str, Any]):
        self.project_root = project_root
        self.models_config = models_config
        self.init_models()
        self.load_matrix()
        self._model = None

    def init_models(self):
        if self.models_config:
            models_builder = ModelsBuilder.get_instance()
            if not models_builder.initialize_models(self.models_config):
                log_simple("MODELOS NO SE PUDIERON INICIAR: ABORTANDO")
                return None
            log_simple(f"MODELOS INSTANCIADOS CORRECTAMENTE")

    @property
    def model(self) -> Optional[Any]:
        if self._model is None:
            models_builder = ModelsBuilder.get_instance()
            self._model = models_builder.word_finder
            if self._model is None:
                basic_exc_logger("WORD FINDER no disponible PARA TESTEO")
                return None
        return self._model
    
    
    def load_matrix(self):
        file_path = os.path.join(self.project_root, "core", "assets", "data.npy")
        matrix = np.load(file_path, allow_pickle=False)
        log_simple(f"MATRIZ: {matrix.shape}")
        
    def run_queries(self):
        if self.model is None:
            basic_exc_logger("NO HAY MODELO DISPONBLE")
            return None
        
        num_matches = 0
        num_no_matches = 0
        no_matches: List[str] = []
        results = []
        time0 = time.perf_counter()

        for q in base_queries2:
            valid_results: List[Dict[str, Any]] = self.model.find_keywords(q)
            if valid_results:
                num_matches += 1
                results.append(valid_results.get("key_field"))
                results.append(valid_results.get('norm_ocr_text'))
                results.append(valid_results.get("key_word"))
                results.append(valid_results.get("similarity"))
            else:
                num_no_matches += 1
                no_matches.append(q)
                log_simple(f"QUERIES1: No match para: '{q}'")
        
        basic_exc_logger(f"TIEMPO TOTAL DE TESTEO: {time.perf_counter() - time0:.6f}'s\n"f"RESULTS: {results}")
    