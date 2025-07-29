# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import os
import cv2
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class PaddleOCRWrapper:
    """
    Una instancia de PaddleOCR especializada únicamente en el RECONOCIMIENTO
    de texto en imágenes pre-recortadas (polígonos).
    """
    def __init__(self, config_dict: Dict, project_root: str):
        start_time = time.perf_counter()
        self.paddle_config = config_dict
        self.project_root = project_root
        logger.info("=== INICIALIZACIÓN PADDLEOCR ===")
        logger.info(f"Initializing PaddleOCR for RECOGNITION (text transcription)...")
        logger.debug(f"Config for PaddleOCR (rec-only): {self.paddle_config}")

        try:
            # --- PARÁMETROS PARA RECONOCIMIENTO PURO Y POR LOTES ---
            init_params = {
                'use_angle_cls': False,
                'det': False,  # <- ¡CLAVE! No carga el modelo de detección.
                'lang': self.paddle_config.get('lang', 'es'),
                'show_log': self.paddle_config.get('show_log', False),
                'use_gpu': self.paddle_config.get('use_gpu', False),
                'enable_mkldnn': self.paddle_config.get('enable_mkldnn', True),
                'rec_batch_num': 64,  # Procesar hasta 64 imágenes en un lote
            }
            
            logger.info(f"Parámetros de inicialización: {init_params}")
        
            model_load_start = time.perf_counter()
            logger.info("Cargando modelo de reconocimiento...")
            
            rec_model_path = self.paddle_config.get('rec_model_dir')
            if rec_model_path and os.path.exists(rec_model_path):
                init_params['rec_model_dir'] = rec_model_path
                logger.info(f"Recognition-only model loaded from local path: {rec_model_path}")
            
            self.engine = PaddleOCR(**init_params)
            
            model_load_time = time.perf_counter() - model_load_start
            total_init_time = time.perf_counter() - start_time
            
            logger.info(f"Modelo cargado en: {model_load_time:.3f}s")
            logger.info(f"Total inicialización: {total_init_time:.3f}s")
            logger.info("PaddleOCR instance for RECOGNITION initialized successfully.")
            logger.info("=== FIN INICIALIZACIÓN PADDLEOCR ===")

        except Exception as e:
            logger.error(f"Critical error initializing PaddleOCR for recognition: {e}", exc_info=True)
            self.engine = None

    def recognize_text_from_image(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        start_time = time.perf_counter()
        
        if self.engine is None:
            logger.error("PaddleOCR recognition engine not initialized. Cannot recognize text.")
            return None
        if image is None or image.size == 0:
            logger.warning("Se recibió una imagen vacía para el reconocimiento.")
            return None

        logger.debug(f"Procesando imagen de tamaño: {image.shape}")
        
        try:
            ocr_start = time.perf_counter()
            result = self.engine.ocr(image, cls=False)
            ocr_time = time.perf_counter() - ocr_start
            
            logger.debug(f"OCR ejecutado en: {ocr_time:.4f}s")
            
            # Validar el resultado
            if not result or not result[0] or not result[0][0]:
                logger.debug("OCR no devolvió resultados para un polígono.")
                return None

            # Extraer texto y confianza
            text, confidence = result[0][0]
            
            total_time = time.perf_counter() - start_time
            logger.debug(f"Total tiempo polígono: {total_time:.4f}s - Texto: '{text}'")
            
            return {
                "text": str(text).strip(),
                "confidence": round(float(confidence) * 100.0, 2) if isinstance(confidence, (float, int)) else 0.0
            }

        except Exception as e:
            logger.error(f"Error durante el reconocimiento de texto en un polígono: {e}", exc_info=True)
            return None

    def recognize_text_from_batch(self, images: List[np.ndarray]) -> List[Optional[Dict[str, Any]]]:
        """
        Ejecuta OCR en un lote (batch) de imágenes pre-recortadas.
        Espera que el motor se haya inicializado con det=False.

        Args:
            images: Una lista de imágenes numpy.

        Returns:
            Una lista de diccionarios con 'text' y 'confidence', uno por cada imagen.
            El orden de la lista de resultados corresponde al orden de las imágenes de entrada.
        """
        if self.engine is None:
            logger.error("PaddleOCR recognition engine not initialized. Cannot recognize text.")
            return [None] * len(images)
        if not images:
            logger.warning("Se recibió una lista vacía de imágenes para el reconocimiento por lotes.")
            return []

        try:
            start_time = time.perf_counter()
            # PaddleOCR procesará la lista de imágenes en un solo lote.
            batch_results = self.engine.ocr(images, cls=False, det=False)
            total_time = time.perf_counter() - start_time
            logger.info(f"Batch OCR para {len(images)} polígonos completado en: {total_time:.3f}s")
            
            # DEBUG: Información detallada sobre lo que devolvió PaddleOCR
            logger.info(f"PaddleOCR devolvió {len(batch_results)} resultados (esperados: {len(images)})")
            logger.info(f"Tipo de batch_results: {type(batch_results)}")
            logger.info(f"Primer resultado: {batch_results[0] if batch_results else 'None'}")
            logger.info(f"Estructura del primer resultado: {type(batch_results[0]) if batch_results else 'None'}")

            # Procesar cada resultado en el lote
            final_results = []
            for i, result_for_image in enumerate(batch_results):
                logger.debug(f"Procesando resultado {i}: {result_for_image}")
                # El resultado para una imagen de línea es como: [[('texto reconocido', 0.99)]]
                if not result_for_image or not result_for_image[0] or not result_for_image[0][0]:
                    logger.debug(f"Resultado {i} vacío o inválido: {result_for_image}")
                    final_results.append(None)
                    continue
                
                # Para una línea pre-recortada, se asume un único resultado principal.
                text, confidence = result_for_image[0][0]
                
                processed_result = {
                    "text": str(text).strip(),
                    "confidence": round(float(confidence) * 100.0, 2) if isinstance(confidence, (float, int)) else 0.0
                }
                final_results.append(processed_result)
                logger.debug(f"Resultado {i} procesado: {processed_result}")
                
            logger.info(f"Total de resultados finales procesados: {len(final_results)}")
            return final_results

        except Exception as e:
            logger.error(f"Error durante el reconocimiento de texto en lote: {e}", exc_info=True)
            return [None] * len(images)