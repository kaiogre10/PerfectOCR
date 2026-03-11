# core/domain/ocr_motor_manager.py
import logging
import threading
import time
from typing import Dict, Any, Optional
from paddleocr import PaddleOCR #type: ignore
from core.utils.word_finder import WordFinder

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
            models_config = config.get("models_config", {})
            activate_rec = config.get("activate_rec")
            activate_det = config.get("activate_det")
            activate_wf = config.get("activate_wf")

            # 1. Inicialización SELECTIVA de motores de Paddle
            if activate_det or activate_rec:
                # PaddleOCR solo cargará en RAM/VRAM los modelos marcados como True
                self._shared_engine = PaddleOCR(
                    det=activate_det, 
                    rec=activate_rec,
                    cls=models_config.get('use_angle_cls'),
                    det_model_dir=models_config.get('det_model_dir'),
                    rec_model_dir=models_config.get('rec_model_dir'),
                    show_log=models_config.get('show_log'),
                    use_gpu=models_config.get('use_gpu'),
                    enable_mkldnn=models_config.get('enable_mkldnn'),
                    lang=models_config.get("lang"),
                    table= models_config.get('table'),
                    rec_batch_num = models_config.get('rec_batch_num'),
                    cpu_threads = models_config.get('cpu_threads'),
                    max_batch_size= models_config.get('max_batch_size'),
                    det_limit_side_len= models_config.get('det_limit_side_len'),
                    det_db_score_mode= models_config.get('det_db_score_mode'),
                    use_mp= models_config.get('use_mp'),
                    max_text_length = models_config.get('max_text_length'),
                    rec_image_inverse = models_config.get('rec_image_inverse'),
                )
                # Asignamos al puntero solo si el modelo está realmente activo
                self._detection_engine = self._shared_engine if activate_det else None
                self._recognition_engine = self._shared_engine if activate_rec else None
                self._initialized = True
                logger.debug(f"Motores Paddle listos (det={activate_det}, rec={activate_rec})")
            else:
                logger.debug("Ningún modelo de Paddle requerido. Saltando inicialización.")
                self._shared_engine = None
                self._detection_engine = None
                self._recognition_engine = None

            # 2. Inicialización de WordFinder
            if activate_wf:
                self._word_finder = WordFinder(
                    model_path=models_config.get("wf_model_path"),
                    set_params=models_config.get("set_wf_params", False)
                )
                self._active = True
                logger.debug(f"WordFinder cargado en {time.perf_counter() - init_time:.4f}s")
            else:
                self._word_finder = None
                self._active = False

            return True

        except Exception as e:
            logger.error(f"Error crítico inicializando modelos: {e}", exc_info=True)
            self._initialized = False
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
