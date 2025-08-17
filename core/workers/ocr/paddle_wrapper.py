# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import os
import cv2
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from paddleocr import PaddleOCR # type: ignore
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class PaddleOCRWrapper(OCRAbstractWorker):
    """
    Una instancia de PaddleOCR especializada únicamente en el RECONOCIMIENTO
    de texto en imágenes pre-recortadas (polígonos).
    Utiliza carga perezosa para el motor de PaddleOCR.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config_dict = config
        self.init_params = config.get("paddle_rec_config",  {})
        self._engine = None
        
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        
        start_time = time.perf_counter()
        polygons = manager.get_polygons_with_cropped_img()
        logger.debug(f"[PaddleWrapper] Polígonos obtenidos: {len(polygons)}")
        for poly_id, poly_data in list(polygons.items())[:3]:
            cropped_img = poly_data.get("cropped_img", {})
            logger.debug(f"[PaddleWrapper] {poly_id}: cropped_img type={type(cropped_img)}, shape={getattr(cropped_img, 'shape', 'N/A')}")
        
        # Preparar batch (igual que preprocessing pero optimizado para OCR)
        image_list: List[np.ndarray[Any, Any]] = []
        polygon_ids: List[str] = []
        
        for poly_id, poly_data in polygons.items():
            cropped_img = poly_data.get("cropped_img", {})
            cropped_img: np.ndarray[Any, Any]
            if cropped_img is not None:
                # Convertir a np.ndarray si es necesario
                if isinstance(cropped_img, list):
                    cropped_img = np.array(cropped_img)
                
                # Validar que la imagen sea procesable
                if hasattr(cropped_img, 'shape') and len(cropped_img.shape) >= 2:
                    if min(cropped_img.shape[:2]) > 0:  # Dimensiones válidas
                        # CONVERTIR A 3 CANALES para PaddleOCR
                        if len(cropped_img.shape) == 2:  # Escala de grises
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                        elif cropped_img.shape[2] == 1:  # 1 canal
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                        
                        image_list.append(cropped_img)
                        polygon_ids.append(poly_id)
        
        if not image_list:
            logger.warning("[PaddleWrapper] No se encontraron imágenes válidas para OCR.")
            return False
            
        # image_list_copy = image_list.copy()
        # Procesar BATCH (mantener rendimiento)
        batch_results: List[Optional[Dict[str, Any]]] = self.recognize_text_from_batch(image_list)
        # cleaned_batch_results, fragmentation_candidates = self._text_cleanner.clean_and_analyze_batch(polygon_ids, batch_result)
        # polygons = manager.get_polygons()
        # batch_results = self._interceptor.intercept_polygons(polygon_ids, cleaned_batch_results, fragmentation_candidates, image_list_copy, polygons)
        # Actualizar resultados usando el método centralizado
        processed_count = 0
        if batch_results:
            success = manager.update_ocr_results(batch_results, polygon_ids)
            processed_count = len(batch_results) if success else 0

            for poly_id in polygon_ids:
                if poly_id in manager.get_polygons():
                    manager.workflow_dict["polygons"][poly_id]["cropped_img"] = None
                    #logger.debug("Cropped_img liberadas, texto generado")
        
        image_name = manager.get_metadata().get("image_name", "unknown_image")
        self._save_complete_ocr_results(manager, image_name)
        total_time = time.perf_counter() - start_time
        logger.debug(f"[OCREngineManager] Batch OCR completado. {processed_count}/{len(image_list)} polígonos procesados en {total_time:.4f}s.")
        return True 

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
                    "rec_model_dir": self.config_dict.get("C:/PerfectOCR/data/models/paddle/rec/es"),
                    'rec_batch_num': 64,
                }
                
                rec_model_path = self.config_dict.get('rec_model_dir')
                if rec_model_path:
                    if os.path.exists(rec_model_path):
                        init_params["rec_model_dir"] = rec_model_path
                        logger.debug(f"Usando modelo de reconocimiento en: {rec_model_path}")
                    else:
                        logger.warning(f"Ruta del modelo de reconocimiento no válida: {rec_model_path}")
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
                logger.info("OCR no devolvió resultados para un polígono.")
                return None

            # Extraer texto y confianza
            text, confidence = result[0][0]
            
            total_time = time.perf_counter() - start_time
            logger.info(f"Total tiempo polígono: {total_time:.4f}s - Texto: '{text}'")
            
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
            valid_images: List[np.ndarray[Any, Any]] = []
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
                consolidated_results: List[List[Any]] = batch_results[0]
                
                if len(consolidated_results) == len(image_list):
                    logger.debug(f"Resultado consolidado detectado. Mapeando {len(consolidated_results)} textos a {len(image_list)} imágenes por orden.")
                    final_results: List[Dict[str, str|float]] = []
                    for text, confidence in consolidated_results:
                        processed_result = {
                            "text": str(text).strip(),
                            "confidence": round(float(confidence) * 100.0, 2) if isinstance(confidence, (float, int)) else 0.0
                        }
                        final_results.append(processed_result)
                    
                    logger.debug(f"Total de resultados finales procesados: {len(final_results)}")
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
            
            
    def _save_complete_ocr_results(self, manager: DataFormatter, image_name: str):
        """
        Ordena al OutputService que guarde los resultados del OCR.
        """
        if not self.config.get('output_flag', {}).get('ocr_raw', False):
            return

        output_folder = self.config.get('output_folder')
        if not output_folder:
            logger.warning("[OCREngineManager] No se puede guardar resultados OCR porque 'output_folder' no está definido.")
            return

        try:
            from services.output_service import save_ocr_json
            save_ocr_json(manager, output_folder, image_name)
        except Exception as e:
            logger.error(f"[OCREngineManager] Fallo al invocar save_ocr_json: {e}")        
        pass
