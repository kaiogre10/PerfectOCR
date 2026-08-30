# app/models_builder.py
import logging
import threading
import time
from typing import Dict, Any, Optional
from utils.word_finder import WordFinder
from paddleocr import PaddleOCR # type: ignore
from domain.model_factory import MatrixFactory

logger = logging.getLogger(__name__)

class ModelsBuilder:
    _instance = None
    _lock = threading.Lock()
    __slots__ = ("_detection_engine", "_recognition_engine", "_shared_engine", "_word_finder")
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

    def initialize_models(self, config: Dict[str, Any]) -> bool:
        init_time = time.perf_counter()
        try:
            models_config = config.get("models_config", {})
            
            paddle_config = models_config.get("paddle_config", {})
            wf_config = models_config.get("wf_config", {})
            model_update = config.get("update_model")
            test_wf_model = wf_config.get("test_wf_model")
                        
            if test_wf_model:
                logger.warning(f"TESTEANDO WORD FINDER")
                if self._activate_wf(wf_config):
                    return True
            
            elif model_update:
                logger.warning("SE ACTUALIAZÁN PARAMETROS DE WORD FINDER")
                self.edit_model_params(wf_config)
                logger.warning("PARAMETROS ACTUALIZADOS EN INSTANCIA WORD FINDER")
                return True
                    
            # 1. Inicialización SELECTIVA de motores de Paddle
            elif not self._activate_paddle(paddle_config):
                return False
                
            # 2. Inicialización de WordFinder
            elif config.get("activate_wf"):
                if self._activate_wf(wf_config):
                    logger.debug(f"STACK COMPLETO DE MODELOS CARGADOS EN: {time.perf_counter() - init_time:.6f}'s")
                    return True
                else:
                    logger.error("NO SE PUDO INICIAR WORD FINDER")
                    return False
            else:
                logger.debug("SOLO SE CARGÓ OCR")
                self._word_finder = None
                return True

        except ModuleNotFoundError as e:
            logger.error(f"Error crítico inicializando modelos: {e}", exc_info=True)
        return False

    def _activate_paddle(self, paddle_config: Dict[str, Any]) -> bool:
        try:
            # PaddleOCR solo cargará en RAM/VRAM los modelos marcados como True
            activate_rec = paddle_config.get("activate_rec")
            activate_det = paddle_config.get("activate_det")
            activate_cls = paddle_config.get('use_angle_cls')
            if activate_cls:
                logger.warning(f"ADVERTENCIA SE ACTIVÓ EL MODELO DE DETECCIÓN DE ANGULO DE PADDLE: '{activate_cls}'")
            
            if activate_det or activate_rec:
                self._shared_engine = PaddleOCR(
                    det = activate_det, 
                    rec = activate_rec,
                    cls = activate_cls,
                    det_model_dir = paddle_config.get('det_model_dir'),
                    rec_model_dir = paddle_config.get('rec_model_dir'),
                    show_log = paddle_config.get('show_log'),
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
                )
                
                self._detection_engine = self._shared_engine if activate_det else None
                self._recognition_engine = self._shared_engine if activate_rec else None
                logger.debug(f"Motores Paddle listos (det={activate_det}, rec={activate_rec})")
                return True
            else:
                logger.warning("Ningún modelo de Paddle requerido. Saltando inicialización.")
                self._shared_engine = None
                self._detection_engine = None
                self._recognition_engine = None
                return False

        except ImportError as e:
            logger.error(f"NO SE CARGO MODULO OCR: {e}", exc_info=True)
        return False

    def _activate_wf(self, config: Dict[str, Any]) -> bool:
        try:
            factory = self.get_factory(config)
            if factory is None:
                self._word_finder = None
                return False
            else:
                self._word_finder = WordFinder(config=config, motor=factory)
                return True
        except ModuleNotFoundError as e:
            logger.error(f"NO SE PUDO INCIAR WORD FINDER: {e}", exc_info=True)
        self._word_finder = None
        return False
        
    def edit_model_params(self, config: Dict[str, Any]):
        if MatrixFactory.edit_pickle_vals(config):
            logger.info(f"PICLKLE EDITADO CON ÉXITO")
            
        if config.get("test_wf_model"):
            self._activate_wf(config)
        
    @property
    def detection_engine(self) -> Optional[PaddleOCR]:
        return self._detection_engine
    
    @property
    def recognition_engine(self) -> Optional[PaddleOCR]:
        return self._recognition_engine
        
    @property
    def word_finder(self) -> Optional[WordFinder]:
        return self._word_finder
    
    def get_factory(self, config: Dict[str, Any]) -> Optional[MatrixFactory]:
        try:
            return MatrixFactory(config)
        except Exception as e:
            logger.error(f"NO SE PUDO INICIAR LA FACTORY DE MODELOS: {e}", exc_info=True)
        return None