# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, List, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import VectorizationAbstractWorker

logger = logging.getLogger(__name__)

class VectorizationStager:
    """Inicializa el coordinador y sus workers. """
    def __init__(self, workers: List[VectorizationAbstractWorker], stage_config: Dict[str, Any], output_paths: Optional[List[str]], project_root: str):
        self.project_root = project_root
        self.workers = workers
        self.stage_config = stage_config
        self.output_paths = output_paths
    
    def vectorize_results(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """
        Orquesta el flujo completo de vectorización siguiendo una estrategia por fases
        para máxima eficiencia de memoria.
        """
        
        start_time = time.time()
        logger.info("[VectorStager] Iniciando pipeline de vectorización")
        metadata = manager.get_metadata()
        polygons = manager.get_polygons()
        
        
        # Para cada worker, procesar todos los polígonos
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.debug(f"[VectorStager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
                                
            # Contexto con TODOS los polígonos para el worker
            context: Dict[str, Any] = {
                "polygons": polygons,  # TODOS los polígonos, no solo uno
                "metadata": metadata,
                "output_paths": self.output_paths,
                "project_root": self.project_root,
            }
                
            result = worker.vectorize(context, manager)
            if result is None or (hasattr(result, "empty") and result.empty):
                # El resultado es None o un DataFrame vacío
                # Maneja el caso de error o datos insuficientes
                logger.error(f"Worker {worker_name} falló o devolvió resultados vacíos")
                return None, 0.0
            else:
                # El resultado es válido
                continue
        
        elapsed = time.time() - start_time
        logger.info(f"[VectorStager] Pipeline completado en: {elapsed:.3f}s")
        return manager, elapsed
            

    #     indices, non_indices, intervalo = self._table_detector.detectar_lineas_tabulares(sequences,
    #         min_cluster_size=min_cluster,
    #         window_size=window,
    #         coherence_threshold=threshold
    #     )
    #     return {
    #         "status": success_table_detection,
    #         "tabular_indices": indices,
    #         "non_tabular_indices": non_indices,
    #         "intervalo": intervalo,
    #         }
    #     except Exception as e:
    #         logger.error(f"Error crítico durante la delegación a TableDetector: {e}", exc_info=True)
    #         return {"status": "error_table_detection", "message": str(e)}

    # def _process_heavy_vectors_for_table(self, all_lines: List[List[Dict]], interval: Tuple[int, int]):
    #     """
    #     Genera y usa vectores pesados para las líneas tabulares.
    #     """
    #     tabular_lines = all_lines[interval[0]:interval[1] + 1]
        
    #     # 1. GENERACIÓN JUST-IN-TIME de vectores pesados
    #     for i, line in enumerate(tabular_lines):
    #         k = interval[0] + i  # Índice de línea real
            
    #         # Generar vectores elementales completos
    #         elemental_vectors = self._vector_tensorizer.generate_elemental_vectors_for_line(line, k)
            
    #         # Generar vectores atómicos
    #         atomic_vectors = self._vector_tensorizer.generate_atomic_vectors_for_line(line)
            
    #         # Generar perfiles morfológicos  
    #         morphological_profiles = self._vector_tensorizer.generate_morphological_profiles_for_line(line)
            
    #         # Generar vector diferenciador (con la siguiente línea si existe)
    #         next_line = tabular_lines[i + 1] if i + 1 < len(tabular_lines) else None
    #         differential_vectors = self._vector_tensorizer.generate_differential_vector_for_line(line, next_line)
            
    #         # 2. USO de los vectores (ej. para la división de columnas)
    #         # Aquí irían los algoritmos que usan estos vectores
    #         # Por ejemplo: column_divider.split(atomic_vectors, elemental_vectors)
            
    #         logger.debug(f"Línea {k}: {len(elemental_vectors)} vectores elementales, {len(atomic_vectors)} vectores atómicos generados.")
        
    #     # 3. LIBERACIÓN explícita de la memoria
    #     # Los vectores salen de scope automáticamente al final del método
    #     logger.info("Vectores pesados procesados y liberados de memoria.")

    # def _extract_polygons_from_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    #     """Extrae la lista de polígonos de la fuente OCR prioritaria."""
    #     raw_results = payload.get("ocr_raw_results", {})
    #     if "paddleocr" in raw_results and raw_results["paddleocr"].get("words"):
    #         return raw_results["paddleocr"]["words"]
    #     if "tesseract" in raw_results and raw_results["tesseract"].get("words"):
    #         return raw_results["tesseract"]["words"]
    #     return []

    # def _build_final_payload(self, doc_id, proc_time, lines, detection_result) -> Dict[str, Any]:
    #     """Construye el diccionario de salida final para ser guardado en disco."""
    #     num_lines = len(lines)
    #     num_tabular = len(detection_result["tabular_indices"])
        
    #     return {
    #         "doc_id": doc_id,
    #         "status": "success_orchestration",
    #         "processing_time_seconds": proc_time,
    #         "lines_summary": {
    #             "total_lines": num_lines,
    #             "tabular_lines_count": num_tabular,
    #             "non_tabular_lines_count": num_lines - num_tabular,
    #         },
    #         "table_interval": detection_result["intervalo"],
    #         # Incluir los datos de las líneas para depuración si es necesario.
    #         # "lines_data": lines 
    #     }

    # def _save_results(self, payload: Dict[str, Any], doc_id: str):
    #     """Guarda los resultados de la orquestación en disco."""
    #     if not self.output_flags.get("vectorization_results", False):
    #         return

    #     try:
    #         output_dir = self.workflow_config.get('output_folder', os.path.join(self.project_root, "output"))
    #         os.makedirs(output_dir, exist_ok=True)
            
    #         # Guardar el JSON principal de resultados
    #         self.json_handler.save(
    #             data=payload,
    #             output_dir=output_dir,
    #             file_name_with_extension=f"{doc_id}_vectorization_results.json"
    #         )

    #         # Guardar un resumen de texto para inspección rápida
    #         table_info = f"Tabla detectada: filas {payload['table_interval'][0]}-{payload['table_interval'][1]}" if payload.get("table_interval") else "No se detectó tabla."
    #         summary_text = (
    #             f"=== Resumen de Vectorización para {doc_id} ===\n"
    #             f"Total líneas: {payload['lines_summary']['total_lines']}\n"
    #             f"Líneas tabulares: {payload['lines_summary']['tabular_lines_count']}\n"
    #             f"{table_info}\n"
    #             f"Tiempo de orquestación: {payload['processing_time_seconds']:.3f}s"
    #         )
    #         self.text_handler.save(
    #             text_content=summary_text.strip(),
    #             output_dir=output_dir,
    #             file_name_with_extension=f"{doc_id}_vectorization_summary.txt"
    #         )
    #     except Exception as e:
    #         logger.error(f"Error al guardar resultados de vectorización: {e}", exc_info=True)

    # def _save_grouped_lines_text(self, grouped_lines: List[List[Dict[str, Any]]], doc_id: str):
    #     """
    #     Guarda las líneas agrupadas en formato de texto legible para depuración.
    #     """
    #     try:
    #         output_dir = self.workflow_config.get('output_folder', os.path.join(self.project_root, "output"))
    #         os.makedirs(output_dir, exist_ok=True)
            
    #         # Crear contenido de texto con formato mejorado
    #         text_content_lines = []
    #         text_content_lines.append(f"Total de líneas detectadas: {len(grouped_lines)}")
    #         text_content_lines.append("")
            
    #         for i, line in enumerate(grouped_lines):
    #             # Extraer y limpiar el texto de cada polígono en la línea
    #             line_texts = []
    #             for polygon in line:
    #                 text = polygon.get('text', '').strip()
    #                 if text:  # Solo agregar si hay texto
    #                     line_texts.append(text)
                
    #             # Unir todos los textos de la línea con espacios
    #             full_line_text = ' '.join(line_texts)
                
    #             # Solo mostrar líneas que tengan contenido
    #             if full_line_text.strip():
    #                 text_content_lines.append(full_line_text)
            
    #         text_content_lines.append("")
            
    #         # Guardar el archivo de texto
    #         text_content = '\n'.join(text_content_lines)
    #         self.text_handler.save(
    #             text_content=text_content,
    #             output_dir=output_dir,
    #             file_name_with_extension=f"{doc_id}_grouped_lines.txt"
    #         )
            
    #         logger.info(f"Líneas agrupadas guardadas en formato legible para {doc_id}")
            
    #     except Exception as e:
    #         logger.error(f"Error al guardar líneas agrupadas en texto: {e}", exc_info=True)