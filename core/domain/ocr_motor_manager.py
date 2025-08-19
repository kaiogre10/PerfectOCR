# core/domain/ocr_motor_manager.py
import logging
import time
from typing import Dict, Any, Optional
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class PaddleManager:
    _instance = None
    
    def __init__(self):
        if PaddleManager._instance is not None:
            raise Exception("PaddleManager es un singleton. Usa get_instance()")
        self._detection_engine = None
        self._recognition_engine = None
        self._shared_engine = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize_engines(self, paddle_config: Dict[str, Any]):
        init_time = time.perf_counter()
        if not self._initialized:
            self._shared_engine = PaddleOCR(
                det=True, rec=True, cls=False,
                det_model_dir=paddle_config['models']['det_model_dir'],
                rec_model_dir=paddle_config['models']['rec_model_dir'],
                use_angle_cls=paddle_config.get('use_angle_cls', False),
                show_log=paddle_config.get('show_log', False),
                use_gpu=paddle_config.get('use_gpu', False),
                enable_mkldnn=paddle_config.get('enable_mkldnn', True),
                lang=paddle_config.get('lang', 'es'),
                rec_batch_num = paddle_config.get('rec_batch_num', 64)
            )
            logging.debug(f"Paddle iniciado en {time.perf_counter() - init_time:.6f}s")
            # Compartir la MISMA instancia
            self._detection_engine = self._shared_engine
            self._recognition_engine = self._shared_engine
            self._initialized = True
            # logger.info(f"PaddleManager: Engines inicializados - det: {self.detection_engine is not None}, rec: {self.recognition_engine is not None}")
    
    @property
    def detection_engine(self) -> Optional[PaddleOCR]:
        return self._detection_engine
    
    @property  
    def recognition_engine(self) -> Optional[PaddleOCR]:
        return self._recognition_engine