# En el nuevo archivo: services/logging_service.py
import logging
import sys
from typing import Dict, Any

class LoggingService:
    def __int__(self, logging_config: Dict[str, Any]):
        
    def setup(self):
        """Configura el sistema de logging centralizado."""
        logger_root = logging.getLogger()
        logger_root.setLevel(logging.DEBUG)
        if logger_root.hasHandlers():
            logger_root.handlers.clear()
            
        # Leemos los formatos desde el diccionario de config
        formatters = {
            'file': logging.Formatter(
                fmt=self.config.get('file_format'),
                datefmt=self.config.get('date_format')
            ),
            'console': logging.Formatter(self.config.get('console_format'))
        }
        
         # Leemos la ruta del archivo y los niveles del config
        log_file_path = self.config.get('log_file', 'perfectocr.log') # Un default por si acaso
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatters['file'])
        file_handler.setLevel(self.config.get('file_level', 'DEBUG').upper())
        logger_root.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatters['console'])
        console_handler.setLevel(self.config.get('console_level', 'INFO').upper())
        logger_root.addHandler(console_handler)
        
        logging.info("LoggingService: Sistema de logging configurado.")
        
        formatters = {
            'file': logging.Formatter(
                fmt='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(module)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'console': logging.Formatter('%(levelname)s:%(name)s:%(lineno)d - %(message)s')
        }

        file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatters['file'])
        file_handler.setLevel(logging.DEBUG)
        logger_root.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatters['console'])
        console_handler.setLevel(logging.INFO)
        logger_root.addHandler(console_handler)
        return logging.getLogger(__name__)

    logger = setup_logging()
