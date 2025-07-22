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
from core.workflow.preprocessing.binarization import Binarizator
from core.workflow.geometry.multifeaturer import MultiFeacturer

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

        # Acceso correcto a quality_assessment_rules
        quality_rules = config.get('quality_assessment_rules', {})
        
        # Instanciar workers con sus configuraciones específicas
        self._moire = MoireDenoiser(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._sp = DoctorSaltPepper(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._gauss = GaussianDenoiser(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._claher = ClaherEnhancer(config=quality_rules.get('contrast', {}), project_root=self.project_root)
        self._sharp = SharpeningEnhancer(config=quality_rules.get('sharpening', {}), project_root=self.project_root)
        self._bin = Binarizator(config=quality_rules.get('binarize', {}), project_root=self.project_root)
        self._features = MultiFeacturer()

        self.image_saver = ImageOutputHandler()
        
    def apply_preprocessing_pipelines(
        self,
        image_array: np.ndarray,
        input_path: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Procesa la imagen de forma secuencial:
        2. Moiré
        3. Ruido sal y pimienta
        4. Ruido general
        5. Contraste
        6. Nitidez
        7. Sobreiluminación
        8. Binarización
        9. Generación de Features
        """
        try:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
            pipeline_start = time.time()
            logger.info("=== INICIANDO PIPELINE DE PREPROCESAMIENTO ===")
            
            # Extracción de polígonos
            step_start = time.time()
            logger.info("Iniciando extracción de polígonos imagen")
            list_of_polygon_images = self._poly.extract_all_polygons(deskewed_img, polygons_coords)
            
            if not list_of_polygon_images:
                logger.warning("No se extrajeron imágenes de polígonos. Finalizando preprocesamiento.")
                return {"ocr_images": {}}, 0.0

            step_duration = time.time() - step_start
            logger.info(f"Extracción de {len(list_of_polygon_images)} polígonos completada en: {step_duration:.4f}s")
    
            processed_ocr_images = {}
            # Iterar sobre cada imagen de polígono recortada
            for i, polygon_img in enumerate(list_of_polygon_images):
                logger.info(f"Procesando Polígono {i+1}/{len(list_of_polygon_images)}")
            
            # 2. Extracción de polígonos
            step_start = time.time()
            logger.info("Iniciando extracción de polígonos")
            list_of_polygon_images = self._poly.extract_all_polygons(deskewed_img, polygons_coords)
            
            if not list_of_polygon_images:
                logger.warning("No se extrajeron imágenes de polígonos. Finalizando preprocesamiento.")
                return {"ocr_images": {}}, 0.0

            step_duration = time.time() - step_start
            logger.info(f"Extracción de {len(list_of_polygon_images)} polígonos completada en: {step_duration:.4f}s")
    
            processed_ocr_images = {}
            # Iterar sobre cada imagen de polígono recortada
            for i, polygon_img in enumerate(list_of_polygon_images):
                logger.info(f"Procesando Polígono {i+1}/{len(list_of_polygon_images)}")

                # 3. Remoción de moiré
                step_start = time.time()
                logger.info("[2/8] Iniciando detección y corrección de moiré...")
                # La firma de _detect_moire_patterns necesita ser ajustada
                moire_img = self._moire._detect_moire_patterns(polygon_img)
                step_duration = time.time() - step_start
                logger.info(f"[2/8] Corrección de moiré completada en {step_duration:.4f}s")

                # 4. Filtro de ruido sal y pimienta
                step_start = time.time()
                logger.info("[3/8] Iniciando filtrado de ruido sal y pimienta...")
                sp_img = self._sp._estimate_salt_pepper_noise(moire_img)
                step_duration = time.time() - step_start
                logger.info(f"[3/8] Filtrado de ruido sal y pimienta completado en {step_duration:.4f}s")

                # 5. Filtro de ruido general
                step_start = time.time()
                logger.info("[4/8] Iniciando filtrado de ruido general...")
                gauss_img = self._gauss._estimate_gaussian_noise(sp_img)
                step_duration = time.time() - step_start
                logger.info(f"[4/8] Filtrado de ruido general completado en {step_duration:.4f}s")

                # 5. Corrección de sobreiluminación (comentado)
                #if self._detect_overexposure(image):
                #   image = self._apply_overexposure_correction(image)

                # 6. Mejora de contraste
                step_start = time.time()
                logger.info("[5/8] Iniciando mejora de contraste...")
                clahed_img = self._claher._estimate_contrast(gauss_img)
                step_duration = time.time() - step_start
                logger.info(f"[5/8] Mejora de contraste completada en {step_duration:.4f}s")

                # 7. Mejora de nitidez
                step_start = time.time()
                logger.info("[6/8] Iniciando mejora de nitidez...")
                corrected_image = self._sharp._estimate_sharpness(clahed_img)
                step_duration = time.time() - step_start
                logger.info(f"[6/8] Mejora de nitidez completada en {step_duration:.4f}s")

                image_to_binarize = corrected_image.copy()
                # 8. Binarización por separado (solo para las features)
                step_start = time.time()
                logger.info("[7/8] Iniciando binarización...")
                binarized_img = self._bin._estimate_binarization(image_to_binarize)
                step_duration = time.time() - step_start
                logger.info(f"[7/8] Binarización completada en {step_duration:.4f}s")

                # 9. Generación de Features
                step_start = time.time()
                logger.info("[8/8] Generando features...")
                features = self._features._extract_region_features(binarized_img)
                step_duration = time.time() - step_start
                logger.info(f"[8/8] Features generadas en {step_duration:.4f}s")

                # Guardar la imagen final del polígono procesado
                processed_ocr_images[f"polygon_{i}"] = corrected_image

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

            return {"ocr_images": processed_ocr_images}, total_duration

        except Exception as e:
            logger.error(f"Preprocessing: Falló el pipeline del worker: {e}", exc_info=True)
            ocr_images = {}
            return {"ocr_images": ocr_images}, 0.0