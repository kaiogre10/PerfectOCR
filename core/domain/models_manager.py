# core/domain/ocr_motor_manager.py
import logging
import threading
import time
from typing import Dict, Any, Optional
from paddleocr import PaddleOCR # type: ignore
from core.utils.word_finder import WordFinder
# import sys

# try:
#     word_finder = r"C:\word_finder_model\src"
#     if word_finder not in sys.path:
#         sys.path.insert(0, word_finder)
#     from word_finder import WordFinder #type: ignore
    
# except Exception as e:
#     logging.error(f"No se pudo importar WORD_FINDER; {e}", exc_info=True)
    
logger = logging.getLogger(__name__)

class ModelsManager:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        try:
            if ModelsManager._instance is not None:
                raise Exception("ModelsManager es un singleton. Usa get_instance()")
            self._detection_engine = None
            self._recognition_engine = None
            self._shared_engine = None
            self._initialized = False
            self._word_finder = None
            self._active = False
        except Exception as e:
            logger.error(f"Error Manager: '{e}'", exc_info=True)
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def initialize_models(self, config: Dict[str, Any]) -> bool:
        init_time = time.perf_counter()
        try:
            if not config:
                logger.info("No se ejecutará paddle, se detine models manager")
                return False
            
            models_config=config.get("models_config", {})
            self._shared_engine = PaddleOCR(
                det=True, rec=True, cls=False,
                det_model_dir=models_config.get('det_model_dir'),
                rec_model_dir=models_config.get('rec_model_dir'),
                show_log=models_config.get('show_log'),
                use_gpu=models_config.get('use_gpu'),
                enable_mkldnn=models_config.get('enable_mkldnn'),
                lang=models_config.get("lang"),
                rec_batch_num = models_config.get('rec_batch_num')
                )
            # Compartir la MISMA instancia
            self._detection_engine = self._shared_engine
            self._recognition_engine = self._shared_engine
            self._initialized = True
            logger.debug(f"Paddle iniciado en {time.perf_counter() - init_time:.6f}s")
            logger.debug(f"PADDLE Engines inicializados - det: {self.detection_engine is not None}, rec: {self.recognition_engine is not None}")

        except Exception as e:
            logger.error(f"No se pudo iniciar Paddle se deiene el proceso completo: {e}", exc_info=True)
            return False

        try:
            ocr_stage = config["ocr_stage"]
            if ocr_stage and self._shared_engine or self._detection_engine or self._recognition_engine:
                if "data_finder" in ocr_stage:
                    model_path=models_config.get("wf_model_path")
                    self._word_finder: WordFinder = WordFinder(
                        model_path=model_path
                    )
                    self._active = True
                    logger.debug(f"Finder iniciado en: {time.perf_counter() - init_time:.6f}s, MODEL_PATH: {model_path}")
                    return True

                else:
                    self._word_finder = None
                    logger.warning(f"Word Finder no se cargó porque no se usará en el pipeline")
                    return True
            else:
                logger.critical(f"No se pudo iniciar Paddle, no se cargará WordFinder")
                return False

        except Exception as e:
            logger.warning(f"No se pudo iniciar WordFinder: {e}", exc_info=True)
            return True
            
    @property
    def detection_engine(self) -> Optional[PaddleOCR]:
        return self._detection_engine
    
    @property  
    def recognition_engine(self) -> Optional[PaddleOCR]:
        return self._recognition_engine
        
    @property    
    def word_finder(self) -> Optional[WordFinder]:
        return self._word_finder
