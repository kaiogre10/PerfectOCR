# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
import os
from typing import List, Dict, Any, Tuple, Optional
#from core.workflow.vectorization.density_scanner import DensityScanner
from core.workflow.vectorial_transformation.tensorizer import VectorTensorizer
from core.workspace.utils.output_handlers import OutputHandler, TextOutputHandler

logger = logging.getLogger(__name__)

class TensorCoordinator:
    """
    PRINCIPIO: Orquestador puro. No ejecuta lógica de negocio.
    Su única responsabilidad es coordinar a los workers especializados para ejecutar
    el flujo de vectorización y detección de tablas de manera eficiente.
    """

    def __init__(self, config: Dict, project_root: str, output_flags: Dict[str, bool]):
        """
        Inicializa el coordinador y sus workers.

        Args:
            config: La sección de configuración para este coordinador.
            project_root: La ruta raíz del proyecto.
            output_flags: Diccionario con las banderas de salida activas.
        """
        self.config = config
        self.project_root = project_root
        self.output_flags = output_flags

        self.workflow_config = config.get('workflow', {})
        self.tensor_config = config.get('vectorization_process', {})
        self.density_map = config.get('density_map_path', {})

        self._vector_tensorizer = VectorTensorizer(config=self.config, project_root=self.project_root, density_map=self.density_map)
        
        
        self.json_handler = OutputHandler(config={'enabled_outputs': self.output_flags})
        self.text_handler = TextOutputHandler()
        logger.debug("Tensor inicializado con sus workers.")

    def orchestrate_vectorization_and_detection(
        self, 
        ocr_results_payload: Dict[str, Any], 
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Orquesta el flujo completo de vectorización siguiendo una estrategia por fases
        para máxima eficiencia de memoria.

        Fase 1: Agrupación de polígonos en líneas.
        Fase 2: Generación de secuencias de densidad LIGERAS para todas las líneas.
        Fase 3: Detección de la tabla usando las secuencias ligeras (DTW).
        Fase 4 (Condicional): Si se detecta una tabla, se generan los vectores
                 PESADOS (Elementales y Atómicos Enriquecidos) solo para
                 las líneas tabulares, se usan y se descartan."""
        
        start_time = time.perf_counter()

        # --- Fase 1: Agrupación de Líneas
        line_grouping_result = self._delegate_to_line_grouper(ocr_results_payload)
        if line_grouping_result.get("status", "").startswith("error"):
            return line_grouping_result
        
        grouped_lines = line_grouping_result["grouped_lines"]
        
        # Guardar las líneas agrupadas en formato de texto si está habilitado
        if self.output_flags.get("debug_grouped_lines_text", False):
            self._save_grouped_lines_text(grouped_lines, doc_id)
        
        # TEMPORAL: Terminar después del agrupamiento de líneas
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        logger.info(f"PROCESO TERMINADO TEMPORALMENTE después del agrupamiento de líneas.")
        logger.info(f"Se agruparon {len(grouped_lines)} líneas en {processing_time:.3f} segundos.")
        
        return {
            "status": "success_line_grouping_only",
            "message": "Proceso terminado después del agrupamiento de líneas (temporal).",
            "processing_time_seconds": processing_time,
            "total_lines": len(grouped_lines),
            "grouped_lines": grouped_lines
        }
    
    def _delegate_to_line_grouper(self, ocr_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Delega la tarea de agrupar polígonos en líneas al worker correspondiente."""
        poligonos = self._extract_polygons_from_payload(ocr_payload)
        if not poligonos:
            return {"status": "error_line_grouping", "message": "No se encontraron polígonos en el payload de OCR."}
        
        logger.info(f"Fase 1: Delegando agrupación de {len(poligonos)} polígonos a SubsetGrouper.")
        return self._subset_grouper.agrupar_poligonos(poligonos)

    def _delegate_to_table_detector(self, sequences: List[List[int]]) -> Dict[str, Any]:
        """Delega la detección de la tabla al worker TableDetector."""
        try:
            params = self.config.get('table_detection_params', {})
            min_cluster = params.get('min_cluster_size', 3)
            window = params.get('window_size', 4)
            threshold = params.get('coherence_threshold', 0.38) # Usar el nuevo default si no está en el YAML
            
            logger.debug(f"Delegando a TableDetector con params: min_cluster={min_cluster}, window={window}, threshold={threshold}")

            indices, non_indices, intervalo = self._table_detector.detectar_lineas_tabulares(
                sequences,
                min_cluster_size=min_cluster,
                window_size=window,
                coherence_threshold=threshold
            )
            return {
                "status": "success_table_detection",
                "tabular_indices": indices,
                "non_tabular_indices": non_indices,
                "intervalo": intervalo
            }
        except Exception as e:
            logger.error(f"Error crítico durante la delegación a TableDetector: {e}", exc_info=True)
            return {"status": "error_table_detection", "message": str(e)}

    def _process_heavy_vectors_for_table(self, all_lines: List[List[Dict]], interval: Tuple[int, int]):
        """
        Genera y usa vectores pesados para las líneas tabulares.
        """
        tabular_lines = all_lines[interval[0]:interval[1] + 1]
        
        # 1. GENERACIÓN JUST-IN-TIME de vectores pesados
        for i, line in enumerate(tabular_lines):
            k = interval[0] + i  # Índice de línea real
            
            # Generar vectores elementales completos
            elemental_vectors = self._vector_tensorizer.generate_elemental_vectors_for_line(line, k)
            
            # Generar vectores atómicos
            atomic_vectors = self._vector_tensorizer.generate_atomic_vectors_for_line(line)
            
            # Generar perfiles morfológicos  
            morphological_profiles = self._vector_tensorizer.generate_morphological_profiles_for_line(line)
            
            # Generar vector diferenciador (con la siguiente línea si existe)
            next_line = tabular_lines[i + 1] if i + 1 < len(tabular_lines) else None
            differential_vectors = self._vector_tensorizer.generate_differential_vector_for_line(line, next_line)
            
            # 2. USO de los vectores (ej. para la división de columnas)
            # Aquí irían los algoritmos que usan estos vectores
            # Por ejemplo: column_divider.split(atomic_vectors, elemental_vectors)
            
            logger.debug(f"Línea {k}: {len(elemental_vectors)} vectores elementales, {len(atomic_vectors)} vectores atómicos generados.")
        
        # 3. LIBERACIÓN explícita de la memoria
        # Los vectores salen de scope automáticamente al final del método
        logger.info("Vectores pesados procesados y liberados de memoria.")

    def _extract_polygons_from_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae la lista de polígonos de la fuente OCR prioritaria."""
        raw_results = payload.get("ocr_raw_results", {})
        if "paddleocr" in raw_results and raw_results["paddleocr"].get("words"):
            return raw_results["paddleocr"]["words"]
        if "tesseract" in raw_results and raw_results["tesseract"].get("words"):
            return raw_results["tesseract"]["words"]
        return []

    def _build_final_payload(self, doc_id, proc_time, lines, detection_result) -> Dict[str, Any]:
        """Construye el diccionario de salida final para ser guardado en disco."""
        num_lines = len(lines)
        num_tabular = len(detection_result["tabular_indices"])
        
        return {
            "doc_id": doc_id,
            "status": "success_orchestration",
            "processing_time_seconds": proc_time,
            "lines_summary": {
                "total_lines": num_lines,
                "tabular_lines_count": num_tabular,
                "non_tabular_lines_count": num_lines - num_tabular,
            },
            "table_interval": detection_result["intervalo"],
            # Incluir los datos de las líneas para depuración si es necesario.
            # "lines_data": lines 
        }

    def _save_results(self, payload: Dict[str, Any], doc_id: str):
        """Guarda los resultados de la orquestación en disco."""
        if not self.output_flags.get("vectorization_results", False):
            return

        try:
            output_dir = self.workflow_config.get('output_folder', os.path.join(self.project_root, "output"))
            os.makedirs(output_dir, exist_ok=True)
            
            # Guardar el JSON principal de resultados
            self.json_handler.save(
                data=payload,
                output_dir=output_dir,
                file_name_with_extension=f"{doc_id}_vectorization_results.json"
            )

            # Guardar un resumen de texto para inspección rápida
            table_info = f"Tabla detectada: filas {payload['table_interval'][0]}-{payload['table_interval'][1]}" if payload.get("table_interval") else "No se detectó tabla."
            summary_text = (
                f"=== Resumen de Vectorización para {doc_id} ===\n"
                f"Total líneas: {payload['lines_summary']['total_lines']}\n"
                f"Líneas tabulares: {payload['lines_summary']['tabular_lines_count']}\n"
                f"{table_info}\n"
                f"Tiempo de orquestación: {payload['processing_time_seconds']:.3f}s"
            )
            self.text_handler.save(
                text_content=summary_text.strip(),
                output_dir=output_dir,
                file_name_with_extension=f"{doc_id}_vectorization_summary.txt"
            )
        except Exception as e:
            logger.error(f"Error al guardar resultados de vectorización: {e}", exc_info=True)

    def _save_grouped_lines_text(self, grouped_lines: List[List[Dict[str, Any]]], doc_id: str):
        """
        Guarda las líneas agrupadas en formato de texto legible para depuración.
        """
        try:
            output_dir = self.workflow_config.get('output_folder', os.path.join(self.project_root, "output"))
            os.makedirs(output_dir, exist_ok=True)
            
            # Crear contenido de texto con formato mejorado
            text_content_lines = []
            text_content_lines.append(f"Total de líneas detectadas: {len(grouped_lines)}")
            text_content_lines.append("")
            
            for i, line in enumerate(grouped_lines):
                # Extraer y limpiar el texto de cada polígono en la línea
                line_texts = []
                for polygon in line:
                    text = polygon.get('text', '').strip()
                    if text:  # Solo agregar si hay texto
                        line_texts.append(text)
                
                # Unir todos los textos de la línea con espacios
                full_line_text = ' '.join(line_texts)
                
                # Solo mostrar líneas que tengan contenido
                if full_line_text.strip():
                    text_content_lines.append(full_line_text)
            
            text_content_lines.append("")
            
            # Guardar el archivo de texto
            text_content = '\n'.join(text_content_lines)
            self.text_handler.save(
                text_content=text_content,
                output_dir=output_dir,
                file_name_with_extension=f"{doc_id}_grouped_lines.txt"
            )
            
            logger.info(f"Líneas agrupadas guardadas en formato legible para {doc_id}")
            
        except Exception as e:
            logger.error(f"Error al guardar líneas agrupadas en texto: {e}", exc_info=True)