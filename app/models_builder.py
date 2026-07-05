# app/models_builder.py
import os
import logging
import threading
import time
from typing import Dict, Any, Optional
from utils.word_finder import WordFinder
from paddleocr import PaddleOCR # type: ignore
from domain.matrix_factory import MatrixManager

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
        init_time = time.perf_counter()
        try:
            # 1. Inicialización SELECTIVA de motores de Paddle
            if not self._activate_paddle(config, project_root):
                return False

            # 2. Inicialización de WordFinder
            elif config.get("activate_wf") and self._activate_wf(config.get("models_config", {}), project_root):
                #logger.info(f"STACK COMPLETO DE MODELOS CARGADOS EN: {time.perf_counter() - init_time:.6f}'s")
                return True
            else:
                #logger.debug("SOLO SE CARGÓ OCR")
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

    def _activate_paddle(self, config: Dict[str, Any], project_root: str) -> bool:
        try:
            # PaddleOCR solo cargará en RAM/VRAM los modelos marcados como True
            activate_rec = config.get("activate_rec")
            activate_det = config.get("activate_det")
            models_config = config.get("models_config", {})
            paddle_config = models_config.get("paddle_config", {})

            show_log = paddle_config.get('show_log')
            if show_log:
                save_log_path = os.path.join(project_root)
            else:
                save_log_path = paddle_config["save_log_path"]

            activate_cls = paddle_config.get('use_angle_cls')
            if activate_cls:
                logger.warning(f"ADVERTENCIA SE ACTIVÓ EL MODELO DE DETECCIÓN DE ANGULO DE PADDLE: '{activate_cls}'")
            
            if activate_det or activate_rec:
                
                det_dir = paddle_config['det_model_dir']
                rec_dir = paddle_config['rec_model_dir']
                
                det_model_dir = os.path.join(project_root, *det_dir)
                rec_model_dir = os.path.join(project_root, *rec_dir)
            
                self._shared_engine = PaddleOCR(
                    det = activate_det, 
                    rec = activate_rec,
                    cls = activate_cls,
                    det_model_dir = det_model_dir,
                    rec_model_dir = rec_model_dir,
                    show_log = show_log,
                    use_gpu = paddle_config.get('use_gpu'),
                    enable_mkldnn = paddle_config.get('enable_mkldnn'),
                    table = paddle_config.get('table'),
                    lang = paddle_config.get("lang"),
                    rec_batch_num = paddle_config.get('rec_batch_num'),
                    cpu_threads = paddle_config.get('cpu_threads'),
                    max_batch_size = paddle_config.get('max_batch_size'),
                    det_limit_side_len = paddle_config.get('det_limit_side_len'),
                    det_db_score_mode = paddle_config.get('det_db_score_mode'),
                    use_mp = paddle_config.get('use_mp'),
                    max_text_length = paddle_config.get('max_text_length'),
                    return_word_box = paddle_config.get('return_word_box'),
                    save_log_path = save_log_path
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

    def _activate_wf(self, config: Dict[str, Any], project_root: str) -> bool:
        wf_config = config.get("wf_config", {})
        model = MatrixManager(project_root, wf_config)

        try:
            self._word_finder = WordFinder(
                config=wf_config,
                motor=model,
                project_root=project_root
            )
            return True
        except ImportError as e:
            logger.error(f"NO SE CARGO MODULO OCR: {e}", exc_info=True)
        self._word_finder = None
        return False
