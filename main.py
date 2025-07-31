# PerfectOCR/main.py
import os
import sys
import logging
import cv2
import numpy as np
import time
from typing import Dict, Optional, Any, Tuple, List
from managment.polygon_manager import PolygonManager
from managment.preprocessing_manager import PreprocessingManager
from managment.ocr_manager import OCREngineManager
#managment.tensor_manager import TensorManager
# managment.text_manager import TextualManager
from core.utils.output_handlers import OutputHandler
from managment.cache_manager import CacheManager
from core.utils.encoders import NumpyEncoder
from managment.config_manager import ConfigManager
#from domain.workflow_job import WorkflowJob
from core.utils.batch_tools import get_optimal_workers
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.update({
    'OMP_NUM_THREADS': '1',        # Conservador para evitar contención
    'MKL_NUM_THREADS': '2',        # Conservador
    'FLAGS_use_mkldnn': '1',       # Mantener (es estable en main thread)
})

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MASTER_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "perfectocr.txt")
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def setup_logging():
    """Configura el sistema de logging centralizado."""
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
    if logger_root.hasHandlers():
        logger_root.handlers.clear()

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

class PerfectOCRWorkflow:
    def __init__(self, master_config_path: str):
        self.config_loader = ConfigManager(master_config_path)
        self.config = self.config_loader.config
        self.project_root = PROJECT_ROOT
        self._poly_coordinator: Optional[PolygonManager] = None
        self._preprocessing_coordinator: Optional[PreprocessingManager] = None
        self._ocr_coordinator: Optional[OCREngineManager] = None
 #       self._tensor_coordinator: Optional[TensorCoordinator] = None
        #self._text_cleaning_coordinator: Optional[TextCleaningCoordinator] = None
        output_config = self.config.get('output_config', {})
        self.output_handler = OutputHandler(config=output_config)
        self.workflow_config = self.config_loader.get_workflow_config()
        self.output_flags = self.config.get('output_config', {}).get('enabled_outputs', {})
        
    @property
    def polygon_coordinator(self) -> PolygonManager:
        if self._poly_coordinator is None:
            self._poly_coordinator = PolygonManager(
                config=self.config_loader.get_polygonal_config(), 
                project_root=self.project_root,
            )
        return self._poly_coordinator

    @property
    def preprocessing_coordinator(self) -> PreprocessingManager:
        if self._preprocessing_coordinator is None:
            self._preprocessing_coordinator = PreprocessingManager(
                config=self.config_loader.get_preprocessing_coordinator_config(),
                project_root=self.project_root
            )
        return self._preprocessing_coordinator

    @property
    def ocr_coordinator(self) -> OCREngineManager:
        """Inicializa perezosamente y devuelve el coordinador de OCR."""
        if self._ocr_coordinator is None:
            logger.info("Inicializando OCREngineCoordinator bajo demanda para transcripción...")
            self._ocr_coordinator = OCREngineManager(
                config=self.config_loader.get_ocr_config(),
                project_root=self.project_root,
                output_flags=self.output_flags,
                workflow_config=self.workflow_config
            )
        return self._ocr_coordinator

  #  @property
   # def tensor_coordinator(self) -> TensorCoordinator:
    #    if self._tensor_coordinator is None:
     #       self._tensor_coordinator = TensorCoordinator(
      #         config=self.config_loader.get_tensor_coordinator_config(), 
       #         project_root=self.project_root,
        #    )
        #return self._tensor_coordinator
    
#    @property
#    def text_cleaning_coordinator(self) -> TextCleaningCoordinator:
#        if self._text_cleaning_coordinator is None:
#            text_cleaning_config = self.config_loader.get_text_cleaning_config()
#            self._text_cleaning_coordinator = TextCleaningCoordinator(
#                config=text_cleaning_config['text_cleaning'],
#                output_flags=text_cleaning_config['output_flags']
#            )
#        return self._text_cleaning_coordinator

    def process_document(self, input_path: str, output_dir_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        workflow_start = time.perf_counter()
        processing_times_summary: Dict[str, float] = {}
        original_file_name = os.path.basename(input_path)
        base_name = os.path.splitext(original_file_name)[0]

        # CREAR ProcessingJob
        #job = WorkflowJob(job_id=input_path)

        # --- Cargar imagen ---
        image_array = cv2.imread(input_path)
        if image_array is None:
            return self._build_error_response("error_loading_image", original_file_name, "No se pudo cargar la imagen", "load")
        #job.full_img = image_array

        # FASE: Obtención de polígonos
        phase1_start = time.perf_counter()
        refined_polygons, time_poly = self.polygon_coordinator._generate_polygons(
            image_array,
            input_path
        )
        phase1_time = time.perf_counter() - phase1_start
        processing_times_summary["1_polygons"] = round(time_poly, 4)
        logger.info(f"Polygonos: {phase1_time:.3f}s")
        
        # Validar que se obtuvieron polígonos válidos
        if refined_polygons is None:
            logger.critical("No se pudieron generar polígonos válidos. Abortando procesamiento.")
            return self._build_error_response("error_polygon_generation", original_file_name,
                                              "No se pudieron generar polígonos", "polygon_generation")
        
        # FASE: PREPROCESAMIENTO (ya incluye evaluación interna)
        phase2_start = time.perf_counter()
        
        polygons_list, total_duration = self.preprocessing_coordinator._apply_preprocessing_pipelines(
            refined_polygons, input_path=input_path
        )

        phase2_time = time.perf_counter() - phase2_start
        processing_times_summary["2_preprocesamiento"] = round(total_duration, 4)
        logger.info(f"Preprocesamiento: {phase2_time:.3f}s")

        if not polygons_list.get("polygons"):
            logger.critical("No hay polígonos pre-procesados para OCR. Abortando.")
            return self._build_error_response("error_preprocessing", original_file_name,
                                                "No hay polígonos para OCR", "preprocessing")

        # FASE 3: OCR
        phase3_start = time.perf_counter()
        ocr_results_payload, time_ocr = self.ocr_coordinator._run_ocr_on_polygons(
            polygons=preprocessed_data["polygons"],
            image_file_name=original_file_name
        )
        phase3_time = time.perf_counter() - phase3_start
        processing_times_summary["3_ocr"] = round(time_ocr, 4)
        
        logger.info(f"OCR: {time_ocr:.3f}s")
        
        if not self.ocr_coordinator.validate_ocr_results(ocr_results_payload):
            return self._build_error_response("error_ocr", original_file_name, "OCR sin resultados válidos", "ocr_validation")

        # Guardar resultados JSON crudos del OCR si está habilitado
        ocr_results_json_path = self._save_raw_ocr_json(ocr_results_payload, base_name)

        # FASE 4: Vectorización (actualmente desactivada, se usa el payload de OCR)
        final_payload = ocr_results_payload

        # RESPUESTA FINAL
        final_response = self._build_final_response(
            original_file_name,
            ocr_results_json_path,
            final_payload
        )
        
        total_workflow_time = time.perf_counter() - workflow_start
        processing_times_summary["total_workflow"] = round(total_workflow_time, 4)
        
        if 'metadata' not in final_response: 
            final_response['metadata'] = {}
        final_response['metadata']['processing_times_seconds'] = processing_times_summary
        
        logger.info(f"Total: {total_workflow_time:.3f}s")
        return final_response
    
    def _save_raw_ocr_json(self, ocr_payload: Dict, base_filename: str) -> Optional[str]:
        """Guarda los resultados crudos del OCR en un archivo JSON si está habilitado."""
        if self.output_flags.get('ocr_raw', False):
            output_dir = self.workflow_config.get('output_folder')
            if not output_dir:
                logger.error("No se puede guardar el JSON de OCR porque 'output_folder' no está definido.")
                return None
            
            # Asegurarse de que el directorio de salida exista
            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(output_dir, f"{base_filename}_ocr_raw_results.json")
            try:
                self.output_handler.save(
                    data=ocr_payload,
                    output_dir=output_dir,
                    file_name_with_extension=os.path.basename(file_path)
                )
                logger.info(f"Resultados crudos de OCR guardados en: {file_path}")
                return file_path
            except Exception as e:
                logger.error(f"Fallo al guardar el JSON de OCR en {file_path}: {e}")
        return None

    def _build_error_response(self, status: str, filename: str, message: str, stage: Optional[str] = None) -> dict:
        error_details = {"message": message}
        if stage: error_details["stage"] = stage
        return {"document_id": filename, "status_overall_workflow": status, "error_details": error_details }

    def _build_final_response(self, filename: str, ocr_path: Optional[str], processing_payload: dict) -> dict:
        status = processing_payload.get("status", "error_unknown")
        final_status = "success" if status.startswith("success") else status
        output_config = self.config_loader.config.get('output_config', {})
        output_flags = output_config.get('enabled_outputs', {})
        
        if output_flags.get('line_grouping_results', False):
            output_dir = output_config.get('output_folder', './output')
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self.output_handler.save( # Changed from json_output_handler to output_handler
                data=processing_payload,
                output_dir=output_dir,
                file_name_with_extension=f"{base_name}_line_grouping_results.json"
            )

        summary_payload = processing_payload.copy()
        if 'lines' in summary_payload:
            del summary_payload['lines']
            
        outputs = {
            "ocr_raw_json": ocr_path
        }
        summary = {"processing_status": status, "details": summary_payload}
        return {"document_id": filename, "status_overall_workflow": final_status, "outputs": outputs, "summary": summary}

    def run_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Método wrapper para compatibilidad con WorkflowManager.
        Procesa una sola imagen y retorna el resultado.
        """
        try:
            result = self.process_document(image_path)
            return result if result else {"error": "No result returned", "image": image_path}
        except Exception as e:
            logger.error(f"Error procesando imagen {image_path}: {e}")
            return {"error": str(e), "image": image_path}
        finally:
            logger.info("Proceso terminado. Iniciando limpieza de caché.")
            cache_manager.cleanup_project_cache()

__all__ = ["PerfectOCRWorkflow"]

if __name__ == "__main__":
    cache_manager = CacheManager(MASTER_CONFIG_FILE)
    cache_manager.clear_output_folders()
    try:
        config_loader = ConfigManager(MASTER_CONFIG_FILE)
        workflow_config = config_loader.get_workflow_config()
        
        # Asegurarse de que las rutas sean strings
        input_folder = str(workflow_config.get('input_folder', './input'))
        output_folder = str(workflow_config.get('output_folder', './output'))
        
        batch_mode = workflow_config.get('batch_mode', True)
        workflow = PerfectOCRWorkflow(MASTER_CONFIG_FILE)

        if not os.path.isdir(input_folder):
            logger.critical(f"La carpeta de entrada especificada no existe o no es un directorio: '{input_folder}'")
            sys.exit(1)

        archivos = [f for f in os.listdir(input_folder) if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
        if not archivos:
            logger.critical("No se encontraron imágenes válidas en la carpeta de entrada.")
            sys.exit(1)

        if batch_mode:
            for f in archivos:
                workflow.process_document(os.path.join(input_folder, f), output_folder)
        else:
            # Solo procesa el primer archivo válido
            workflow.process_document(os.path.join(input_folder, archivos[0]), output_folder)
    finally:
        logger.info("Procesamiento finalizado, iniciando limpieza de caché.")
        cache_manager.cleanup_project_cache()
