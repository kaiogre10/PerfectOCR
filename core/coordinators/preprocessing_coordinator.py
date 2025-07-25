# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from core.workflow.preprocessing.moire import MoireDenoiser
from core.workflow.preprocessing.sp import DoctorSaltPepper
from core.workflow.preprocessing.gauss import GaussianDenoiser
from core.workflow.preprocessing.clahe import ClaherEnhancer
from core.workflow.preprocessing.sharp import SharpeningEnhancer

from core.workspace.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PreprocessingCoordinator:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo
    a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        self.workflow_config = config.get('workflow', {})
        self.output_config = config.get('output_config', {})
        quality_rules = config.get('corrections', {})
        
        # Instanciar workers con sus configuraciones específicas
        self._moire = MoireDenoiser(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._sp = DoctorSaltPepper(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._gauss = GaussianDenoiser(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._claher = ClaherEnhancer(config=quality_rules.get('contrast', {}), project_root=self.project_root)
        self._sharp = SharpeningEnhancer(config=quality_rules.get('sharpening', {}), project_root=self.project_root)
        self.image_saver = ImageOutputHandler()
        
    def _apply_preprocessing_pipelines(
        self,
        image_array: np.ndarray,
        input_path: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Procesa la imagen de forma secuencial:
        1. Moiré
        2. Ruido sal y pimienta
        3. Ruido general
        4. Contraste
        5. Nitidez
        """
        try:
            cropped_line = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
            pipeline_start = time.time()
            logger.info("=== INICIANDO PIPELINE DE PREPROCESAMIENTO ===")
            
            # 1. Remoción de moiré
            step_start = time.time()
            logger.info("Iniciando detección y corrección de moiré...")
            moire_img = self._moire._detect_moire_patterns(cropped_line)
            step_duration = time.time() - step_start
            logger.info(f"Corrección de moiré completada en {step_duration:.4f}s")

            # 2. Filtro de ruido sal y pimienta
            step_start = time.time()
            logger.info("Iniciando filtrado de ruido sal y pimienta")
            sp_img = self._sp._estimate_salt_pepper_noise(moire_img)
            step_duration = time.time() - step_start
            logger.info(f"Filtrado de ruido sal y pimienta completado en {step_duration:.4f}s")

            # 3. Filtro de ruido general
            step_start = time.time()
            logger.info("Iniciando filtrado de ruido general")
            gauss_img = self._gauss._estimate_gaussian_noise(sp_img)
            step_duration = time.time() - step_start
            logger.info(f"Filtrado de ruido general completado en {step_duration:.4f}s")

            # . Corrección de sobreiluminación (comentado)
            #if self._detect_overexposure(image):
            #   image = self._apply_overexposure_correction(image)

            # 4. Mejora de contraste
            step_start = time.time()
            logger.info("Iniciando mejora de contraste")
            clahed_img = self._claher._estimate_contrast(gauss_img)
            step_duration = time.time() - step_start
            logger.info(f"Mejora de contraste completada en {step_duration:.4f}s")

            # 5. Mejora de nitidez
            step_start = time.time()
            logger.info("Iniciando mejora de nitidez...")
            corrected_image = self._sharp._estimate_sharpness(clahed_img)
            step_duration = time.time() - step_start
            logger.info(f"Mejora de nitidez completada en {step_duration:.4f}s")

            # Guardar la imagen final del polígono procesado
            # processed_ocr_images[f"polygon_{i}"] = corrected_image

            total_duration = time.time() - pipeline_start
            logger.info(f"PIPELINE COMPLETADO - Tiempo total: {total_duration:.3f}s")
            logger.info("=== FINALIZADO PIPELINE DE PREPROCESAMIENTO ===")
        
            # Guardar imagen si está habilitado (NOTA: Esto guardará la última imagen procesada, se podría ajustar)
            if self.output_config.get('enabled_outputs', {}).get('preprocessed_image', False):
                if not input_path:
                    logger.warning("No se proporcionó 'input_path', no se puede guardar la imagen preprocesada con un nombre de archivo único.")
                else:
                    output_folder = self.workflow_config.get('output_folder')
                    if not output_folder:
                        logger.error("La 'output_folder' no está definida en la configuración del workflow.")
                    else:
                        base_name = os.path.basename(input_path)
                        file_name, _ = os.path.splitext(base_name)
                        # Guardamos la última imagen como referencia
                        output_filename = f"{file_name}_preprocessed_last_poly.png"
                        self.image_saver.save(corrected_image, output_folder, output_filename)

            # Empaquetar el resultado para el siguiente coordinador (OCR).
            # Ahora es un diccionario de imágenes de polígonos
            logger.info("Preprocessing: Pipeline completado exitosamente por el worker.")

            return {"processed_image": corrected_image}, total_duration

        except Exception as e:
            logger.error(f"Preprocessing: Falló el pipeline del worker: {e}", exc_info=True)
            return {"processed_image": None}, 0.0