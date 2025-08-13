# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import os
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from paddleocr import PaddleOCR # type: ignore

logger = logging.getLogger(__name__)

class PaddleOCRWrapper:
    """
    Una instancia de PaddleOCR especializada únicamente en el RECONOCIMIENTO
    de texto en imágenes pre-recortadas (polígonos).
    Utiliza carga perezosa para el motor de PaddleOCR.
    """
    def __init__(self, config_dict: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config_dict = config_dict
        self.init_paramss = config_dict.get("paddleocr", {})
        self._engine = None

    @property
    def engine(self) -> Optional[PaddleOCR]:
        if self._engine is None:
            start_time = time.perf_counter()
            try:
                # Configurar parámetros de inicialización
                init_params = {
                    "use_angle_cls": False,
                    "det": False,
                    "lang": self.config_dict.get("paddle_config", {}).get("lang", "es"),
                    "show_log": self.config_dict.get("paddle_config", {}).get("show_log", False),
                    "use_gpu": self.config_dict.get("paddle_config", {}).get("use_gpu", False),
                    "enable_mkldnn": self.config_dict.get("paddle_config", {}).get("enable_mkldnn", True),
                    "rec_model_dir": "C:/PerfectOCR/data/models/paddle/rec/es",
                    'rec_batch_num': 64,
                }
                
                rec_model_path = self.config_dict.get('rec_model_dir')
                if rec_model_path:
                    if os.path.exists(rec_model_path):
                        init_params["rec_model_dir"] = rec_model_path
                        logger.info(f"Usando modelo de detección en: {rec_model_path}")
                    else:
                        logger.warning(f"Ruta del modelo de detección no válida: {rec_model_path}")
                else:
                    logger.warning("No se especificó 'det_model_dir'; PaddleOCR intentará descargar el modelo.")


                model_load_start = time.perf_counter()
                self._engine = PaddleOCR(**init_params)
                model_load_time = time.perf_counter() - model_load_start
                total_init_time = time.perf_counter() - start_time
                logger.info(f"Total inicialización PaddleOCRWrapper: {total_init_time:.3f}s (carga de modelo: {model_load_time:.3f}s)")

            except Exception as e:
                logger.error(f"Critical error initializing PaddleOCR for recognition: {e}", exc_info=True)
                self._engine = None
        return self._engine

    def recognize_text_from_image(self, image: Optional[np.ndarray[Any, Any]]) -> Optional[Dict[str, Any]]:
        start_time = time.perf_counter()
        
        if self.engine is None:
            logger.error("PaddleOCR recognition engine not initialized. Cannot recognize text.")
            return None
        if image is None or image.size == 0:
            logger.warning("Se recibió una imagen vacía para el reconocimiento.")
            return None
                
        try:
            ocr_start = time.perf_counter()
            result: List[Any] = self.engine.ocr(image, det= False, cls=False)
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

    def recognize_text_from_batch(self, image_list: List[np.ndarray[Any, Any]]) -> List[Optional[Dict[str, Any]]]:
        """
        Ejecuta OCR en un lote (batch) de imágenes pre-recortadas.
        Está adaptado para manejar el caso en que PaddleOCR devuelve una única
        lista consolidada de resultados.
        """
        if self.engine is None:
            logger.error("PaddleOCR recognition engine not initialized. Cannot recognize text.")
            return [None] * len(image_list)
        if not image_list:
            logger.warning("Se recibió una lista vacía de imágenes para el reconocimiento por lotes.")
            return []

        try:
            start_time = time.perf_counter()
            valid_images = []
            for idx, img in enumerate(image_list):
                if img is None or not hasattr(img, "shape") or len(img.shape) < 2 or img.size == 0:
                    logger.warning(f"Imagen inválida en el batch (índice {idx}): {type(img)} - shape: {getattr(img, 'shape', None)}")
                    continue
                valid_images.append(img)
            if not valid_images:
                logger.error("No hay imágenes válidas para el reconocimiento por lotes.")
                return []
            batch_results: List[Any] = self.engine.ocr(image_list, cls=False, det=False)  # type: ignore
            total_time = time.perf_counter() - start_time
            logger.info(f"Batch OCR para {len(image_list)} polígonos completado en: {total_time:.3f}s")
            
            if len(batch_results) == 1 and isinstance(batch_results[0], list):
                consolidated_results = batch_results[0]
                
                if len(consolidated_results) == len(image_list):
                    logger.info(f"Resultado consolidado detectado. Mapeando {len(consolidated_results)} textos a {len(image_list)} imágenes por orden.")
                    final_results = []
                    for text, confidence in consolidated_results:
                        processed_result = {
                            "text": str(text).strip(),
                            "confidence": round(float(confidence) * 100.0, 2) if isinstance(confidence, (float, int)) else 0.0
                        }
                        final_results.append(processed_result)
                    
                    logger.info(f"Total de resultados finales procesados: {len(final_results)}")
                    return final_results
                else:
                    logger.error(f"Error de mapeo: El lote devolvió {len(consolidated_results)} textos para {len(image_list)} imágenes. No se puede garantizar la correspondencia.")
                    return [None] * len(image_list)
            
            # Escenario ideal (si PaddleOCR se comportara como se espera en el futuro)
            logger.info("Procesando resultados con la estructura esperada (un resultado por imagen).")
            final_results = []
            for i, result_for_image in enumerate(batch_results):
                logger.debug(f"Procesando resultado {i}: {result_for_image}")
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
            logger.error(f"Error crítico durante el reconocimiento de texto en lote: {e}", exc_info=True)
            return [None] * len(image_list)