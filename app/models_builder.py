# app/models_builder.py
import logging
import os
import threading
import time
from typing import Dict, Any, Optional
from paddleocr import PaddleOCR # type: ignore
from core.utils.word_finder import WordFinder

logger = logging.getLogger(__name__)

class ModelsBuilder:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        try:
            if ModelsBuilder._instance is not None:
                raise Exception(f"{self.__class__.__name__} es un singleton. Usa get_instance()")
            self._detection_engine = None
            self._recognition_engine = None
            self._shared_engine = None
            self._word_finder = None
        except Exception as e:
            logger.error(f"Error Manager: '{e}'", exc_info=True)
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize_models(self, config: Dict[str, Any], project_root: str) -> bool:
        self.project_root = project_root # type: ignore
        init_time = time.perf_counter()
        try:
            # 1. Inicialización SELECTIVA de motores de Paddle
            if not self._activate_paddle(config):
                return False

            # 2. Inicialización de WordFinder
            elif config.get("activate_wf") and self._activate_wf(config):
                logger.debug(f"STACK COMPLETO DE MODELOS CARGADOS EN: {time.perf_counter() - init_time:.6f}'s")
                return True
            else:
                logger.debug("SOLO SE CARGÓ OCR")
                self._word_finder = None
                return True

        except Exception as e:
            logger.error(f"Error crítico inicializando modelos: {e}", exc_info=True)
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

    def _activate_paddle(self, config: Dict[str, Any]) -> bool:
        try:
            # PaddleOCR solo cargará en RAM/VRAM los modelos marcados como True
            activate_rec = config.get("activate_rec")
            activate_det = config.get("activate_det")
            models_config = config.get("models_config", {})
            
            activate_cls = models_config.get('use_angle_cls')
            if activate_cls:
                logger.warning(f"ADVERTENCIA SE ACTIVÓ EL MODELO DE DETECCIÓN DE ANGULO DE PADDLE: {activate_cls}")
            
            if activate_det or activate_rec:
                
                det_dir = models_config['det_model_dir']
                rec_dir = models_config['rec_model_dir']
                
                det_model_dir = os.path.join(self.project_root, *det_dir)
                rec_model_dir = os.path.join(self.project_root, *rec_dir)
            
                self._shared_engine = PaddleOCR(
                    det=activate_det, 
                    rec=activate_rec,
                    cls=activate_cls,
                    det_model_dir=det_model_dir,
                    rec_model_dir=rec_model_dir,
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
                logger.debug(f"Motores Paddle listos (det={activate_det}, rec={activate_rec})")
                return True
            else:
                logger.debug("Ningún modelo de Paddle requerido. Saltando inicialización.")
                self._shared_engine = None
                self._detection_engine = None
                self._recognition_engine = None
                return False

        except ImportError as e:
            logger.error(f"NO SE CARGO MODULO OCR: {e}", exc_info=True)
        return False

    def _activate_wf(self, config: Dict[str, Any]) -> bool:
        models_config = config.get("models_config", {})
        model_path = models_config["wf_model_path"]
        
        model_dir = os.path.join(self.project_root, *model_path)
        
        try:
            self._word_finder = WordFinder(
                model_path=model_dir,
                set_params=models_config.get("set_wf_params", False)
            )
            return True
        except ImportError as e:
            logger.error(f"NO SE CARGO MODULO OCR: {e}", exc_info=True)
        self._word_finder = None
        return False