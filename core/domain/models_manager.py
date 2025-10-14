# core/domain/ocr_motor_manager.py
import logging
import threading
import time
from typing import Dict, Any, Optional
from paddleocr import PaddleOCR # type: ignore
from data.scripts.word_finder import WordFinder

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
            self._model = False
            self._word_finder = None
            self.project_root = None
            self._active = False
        except Exception as e:
            logger.error(f"Error Manager{e}", exc_info=True)
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def initialize_models(self, models_config: Dict[str, Any], project_root: str) -> bool:
        init_time = time.perf_counter()
        self.project_root = project_root
        try:
            models_section = models_config.get("models", {})
            ocr_stage = models_config.get("ocr_stage", [])
            if "data_fimder" in ocr_stage:
                model_path = models_section.get("wordfinder_model_path")
                self._word_finder = WordFinder(
                    model_path=model_path,
                    project_root=project_root
                )
                self._active = True
                logger.debug(f"Finder iniciado en {time.perf_counter() - init_time:.6f}s")
            else:
                self._word_finder = None
                logger.warning(f"Word Finder no se cargó porque no se usará en el pipeline")

        except Exception as e:
            logger.debug(f"No se pudo iniciar WordFinder{e}", exc_info=True)
            
        try:
            models_section = models_config.get("models", {})    
            self._shared_engine = PaddleOCR(
                det=True, rec=True, cls=False,
                det_model_dir=models_section.get('det_model_dir'),
                rec_model_dir=models_section.get('rec_model_dir'),
                use_angle_cls=models_section.get('use_angle_cls', False),
                show_log=models_section.get('show_log', False),
                use_gpu=models_section.get('use_gpu', False),
                enable_mkldnn=models_section.get('enable_mkldnn', True),
                lang=models_section.get('lang', 'es'),
                rec_batch_num = models_section.get('rec_batch_num', 64)
            )        
            # Compartir la MISMA instancia
            self._detection_engine = self._shared_engine
            self._recognition_engine = self._shared_engine
            self._initialized = True
            logger.debug(f"Paddle iniciado en {time.perf_counter() - init_time:.6f}s")
            logger.debug(f"ModelsManager: Engines inicializados - det: {self.detection_engine is not None}, rec: {self.recognition_engine is not None}")
                
        except Exception as e:
            logger.error(f"No se pudo iniciar Paddle{e}", exc_info=True)
            return False
            
    @property
    def detection_engine(self) -> Optional[PaddleOCR]:
        return self._detection_engine
    
    @property  
    def recognition_engine(self) -> Optional[PaddleOCR]:
        return self._recognition_engine
        
    @property    
    def word_finder(self) -> Optional[WordFinder]:
        return self._word_finder