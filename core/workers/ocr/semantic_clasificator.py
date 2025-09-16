# semantic_clasificator
import logging
from typing import Dict, Any, List, Optional
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class SemanticClasificator(OCRAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("semantic_clasificator", {})
        self.enabled_outputs = config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("semantic_words", False)  
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para preocesar")
                
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            
            final_results: Dict[str, str] = self._clasify_words(polygons)
            
            file_name: str = manager.workflow.metadata.image_name
            
            if self.output:
                self._save_ocr_raw(context, final_results, file_name)
            return True
            
        except Exception as e:
            logger.info(f"Error en el clasiicador{e}", exc_info=True)
            
    def _clasify_words(self, polygons: Dict[str, Polygons]) -> Dict[str, str]:
    
        """""Mide la proporción del tipo de carácter para clasificar semánticamente una palabra"""
        numeric_range = self.worker_config.get("numeric", [70.0, 100.0])
        code_range = self.worker_config.get("code", [31.0, 69.9])
        descriptive_range = self.worker_config.get("descriptive", [0.0, 30.9])
        
        def norm(r):
            return(min(r[0], r[1]), max(r[0], r[1]))
            
        n_min, n_max = norm(numeric_range)
        c_min, c_max = norm(code_range)
        d_min, d_max = norm(descriptive_range) # type: ignore
            
        texts: Dict[str, str] = {poly_id: (polygon.ocr_text or "") for poly_id, polygon in polygons.items()}

        try:
            final_results: Dict[str, str] = {}
            for pid, s in texts.items():
                s = s or ""
                chars = [ch for ch in s if not ch.isspace()]
                total = len(chars)
                if total == 0:
                    pct = 0.0
                else:
                    digits = sum(1 for ch in chars if ch.isdigit())
                    pct = (digits / total) * 100.0

                if n_min <= pct <= n_max:
                    semantic = "numeric"
                elif c_min <= pct <= c_max:
                    semantic = "code"
                else:
                    semantic = "descriptive"

                final_results[pid] = semantic
                            
            return final_results
        except Exception as e:
            logger.info(f"Fallo en el mapeo de las letras semánticas {e}", exc_info=True )
                
    def _save_ocr_raw(self, context: Dict[str, Any], final_results: Dict[str, str], file_name: str):
        from services.output_service import save_json
        import os

        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir: str = os.path.join(path, "semantic_words")
            json_file_name = f"{os.path.splitext(file_name)[0]}.json"
            save_json(final_results, output_dir, json_file_name)
        
        if output_paths:
            logger.info(f"OCR Raw results para '{file_name}' guardado en {len(output_paths)} ubicaciones.")