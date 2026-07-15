import logging
from services import log_service
from testing.testing_data.wf_data import base_queries2, base_queries
from typing import List, Dict, Optional, Any
import time
from app.models_builder import ModelsBuilder

logger = logging.getLogger(__name__)

class TestingManger:
    def __init__(self, project_root: str, models_config: Dict[str, Any]):
        self.project_root = project_root
        log_service.setup_logging(self.project_root)
        self.models_config = models_config
        self.init_models()
        self._model = None

    def init_models(self):
        if self.models_config:
            models_builder = ModelsBuilder.get_instance()
            if not models_builder.initialize_models(self.models_config):
                logger.info("MODELOS NO SE PUDIERON INICIAR: ABORTANDO")
                return None
            logger.info(f"MODELOS INSTANCIADOS CORRECTAMENTE")
        return None

    @property
    def model(self) -> Optional[Any]:
        if self._model is None:
            models_builder = ModelsBuilder.get_instance()
            self._model = models_builder.word_finder
            if self._model is None:
                logger.warning("WORD FINDER no disponible PARA TESTEO")
                return None
        return self._model
        
    def run_queries(self):
        if self.model is None:
            logger.info("NO HAY MODELO DISPONBLE")
            return None
        
        query = base_queries2
        num_matches = 0
        no_matches: List[bytes] = []
        results: List[bytes] = []
        time0 = time.perf_counter()
        for q in query:
            key_resuts = self.model.find_keywords(q)
            if not key_resuts:
                # logger.info(f"QUERIES1: No match para: '{q}'")
                no_matches.append(q)
                continue
                
            # logger.info(f"RESULTS: {key_resuts}")
            num_matches += 1
            results.append(q)
            continue
        
        tottime = time.perf_counter() - time0
        total_test = len(query)
        logger.warning(f"{num_matches}/{total_test} MATCHES, {total_test - num_matches} / {total_test} SIN MATCHES:\n"f"TIEMPO TOTAL DE TESTEO: {tottime:.6f}'s, Promedio: {tottime/total_test:.8f}'s")
        return None