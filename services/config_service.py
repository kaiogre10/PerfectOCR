# services/config_service.py
import yaml
from typing import Dict, Any, cast, List, Set, Tuple
from config.config_models import MasterConfig
import logging

logger = logging.getLogger(__name__)

class ConfigService:
    """Gestor de los parametros de configuración"""
    def __init__(self, config_path: str, TEST_MODE: bool):
        elemental_worker = "image_loader"
        self.ocr_workers: Set[str] = {"geometry_detector", "paddle_wrapper", "polygon_extractor"}
        self.min_workers: Set[str] = self.ocr_workers.union(elemental_worker)

        validated_config = self._load_and_validate_yaml(config_path)
        self.config = validated_config.model_dump()
        elemental_params = elemental_worker in self.create_stager[0][1]
        
        if not elemental_params:
            logger.error(f"ERROR CRÍTICO, NO HAY IMAGE LOADER")
            self.config = {}

        elif TEST_MODE and elemental_params:
            logger.warning(f"TEST MODE ACTIVADO, verificaciones robustas desactivadas. Stages activas: '{self.log_active_areas()}'")
            self.config = self.config

        elif not TEST_MODE and self._validate_min_workers():
            logger.warning(f"MODO PRODUCCIÓN ACTIVADO, se realizarán validaciones robustas. Stages activas: '{self.log_active_areas()}'")
            self.config = self.config

        else:
            logger.error(f"Error de configuración")
            self.config = {}
                
    def _load_and_validate_yaml(self, config_path: str) -> MasterConfig:
        """Carga YAML y valida con Pydantic - ROBUSTEZ."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                raise TypeError(f"Config raíz debe ser dict, recibido: {type(raw).__name__}")
            typed_raw = cast(Dict[str, Any], raw)
            return MasterConfig.model_validate(typed_raw)
            
        except Exception as e:
            logger.error(f"Error validando configuración desde {config_path}: {e}")
            raise
    
    @property
    def processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento."""
        processing = dict(self.config.get("processing", {}))
        valid_ext = self.utils_config.get("valid_image_extensions", ())
        if isinstance(valid_ext, str):
            valid_ext = (valid_ext,)
        else:
            valid_ext = tuple(valid_ext)

        processing["valid_image_extensions"] = valid_ext
        return processing

    @property
    def enabled_outputs(self) -> Dict[str, Any]:
        return self.config.get("enabled_outputs", {})
    
    @property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})

    @property
    def models_config(self) -> Dict[str, Any]:
        ocr_active = not self.ocr_workers.isdisjoint(self.all_workers)

        if not self.all_workers:
            logger.debug("Sin all workers")
            return {}

        # 1. Definir banderas lógicas corregidas:
        # Detección: Lo necesita GeometryDetector para encontrar las cajas.
        activate_det = "geometry_detector" in self.all_workers
        
        # Reconocimiento: Solo si está la trinidad que procesará texto (o si está paddle_wrapper)
        required_for_rec = {"geometry_detector", "polygon_extractor", "paddle_wrapper"}
        activate_rec = required_for_rec.issubset(self.all_workers)

        # 2. Evaluar estados combinados con Word Finder (data_finder)
        
        # Caso A: Word Finder activo (Requiere OCR completo + data_finder)
        if self.ocr_workers.issubset(self.all_workers) and "data_finder" in self.all_workers:
            logger.debug("Configuración: OCR completo + Word Finder")
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": True,
                "activate_rec": activate_rec,
                "activate_det": activate_det
            }

        # Caso B: Algún componente de OCR activo pero sin condiciones para Word Finder
        elif ocr_active:
            logger.debug(f"Configuración: OCR Parcial (Rec:{activate_rec}, Det:{activate_det})")
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": False,
                "activate_rec": activate_rec,
                "activate_det": activate_det
            }

        # Caso C: Solo data_finder (sin lógica de Paddle)
        elif "data_finder" in self.all_workers:
            logger.debug("Configuración: Solo data_finder")
            return {
                "models_config": {},
                "activate_wf": False,
                "activate_rec": False,
                "activate_det": False
            }
        
        else:
            logger.debug("Configuración: No se requiere carga de modelos")
            return {}
                
    @property
    def modules_config(self) -> Dict[str, Any]:
        return self.config.get("modules", {})
    
    @property
    def utils_config(self) -> Dict[str, Any]:
        return self.config.get("utils", {})
      
    @property
    def img_prep_config(self) -> Dict[str, Any]:
        return {
            **self.modules_config.get("image_preparation", {}),
            **self.enabled_outputs.get("image_load_outputs", {}),
            **self.utils_config,
            "imagepre_stage": self.workers_order["imagepre_stage"]
        }
       
    @property
    def preprocessing_config(self)-> Dict[str, Any]:
        if not self.create_stager[1][1]:
            return {}
        else:
            return {
                **self.modules_config.get("preprocessing", {}),
                **self.enabled_outputs.get("preprocessing_outputs", {}),
                **self.utils_config,
                "preprocessing_stage": self.workers_order["preprocessing_stage"]
            }

    @property
    def ocr_config(self) -> Dict[str, Any]:
        if not self.create_stager[2][1] or not self.ocr_workers.issubset(self.all_workers):
            return {}
        else:
            create_refiners = self.modules_config.get("ocr", {}).get("text_refiner", {}).get("num_passes", 0)
            return {
                **self.modules_config.get("ocr", {}),
                **self.enabled_outputs.get("ocr_outputs", {}),
                **self.utils_config,
                "ocr_stage": self.workers_order["ocr_stage"],
                "create_refiners": create_refiners > 0
            }
       
    @property
    def vectorization_config(self) -> Dict[str, Any]:
        vect_stage = self.create_stager[3][1]
        if not vect_stage or not "lineal" in vect_stage or not self.ocr_workers.issubset(self.all_workers):
            return {}
        else:
            return {
                **self.modules_config.get("vectorization", {}),
                **self.enabled_outputs.get("vectorization_outputs", {}),
                **self.utils_config,
                "vector_stage": self.workers_order["vector_stage"]
            }
        
    @property
    def manager_config(self) -> Dict[str, Dict[str, Any]]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return {
            "image_preparation": self.img_prep_config,
            "preprocessing": self.preprocessing_config,
            "ocr": self.ocr_config,
            "vectorization": self.vectorization_config
        }

    def _validate_min_workers(self) -> bool:
        try:
            if not self.workers_order:
                logger.error("No hay configuración de workers disponible")
                return False
            
            if not self.min_workers.issubset(self.all_workers):
                workers_missing: Set[str] = self.min_workers - self.all_workers
                logger.warning(f"Faltan: {workers_missing} de los '{len(self.min_workers)}' workers mínimos para el pipeline")
                return False
            else:
                for stage, stage_workers in self.workers_order.items():
                    if isinstance(stage_workers, (list, tuple, set)): #type: ignore
                        count = len(stage_workers)
                        workers_set = set(w for w in stage_workers if isinstance(w, str)) #type: ignore
                    elif isinstance(stage_workers, str): #type: ignore
                        count = 1
                        workers_set = {stage_workers}
                    else:
                        count = 0
                        workers_set: Set[str] = set()
                    logger.debug(f"Activos '({count}, {workers_set})' workers para '{stage}'")
                return True

        except Exception as e:
            logger.error(f"Error crítico en la revisión de parámetros mínimos: {e}", exc_info=True)
        return False
    
    @property
    def create_stager(self) -> List[Tuple[str, List[str]]]:
        full_stage: List[Tuple[str, List[str]]] = []
        for stage_workers in self.workers_order.items():
            full_stage.append(stage_workers)
        return full_stage
    
    @property
    def all_workers(self) -> Set[str]:
        all_workers = set(self.create_stager[0][1])

        if self.create_stager[1][1]:
            prep = set(self.create_stager[1][1])
            all_workers.update(prep)

        if self.create_stager[2][1]:
            ocr = set(self.create_stager[2][1])
            all_workers.update(ocr)

        if self.create_stager[3][1]:
            vect = set(self.create_stager[3][1])
            all_workers.update(vect)

        return all_workers
        
    def log_active_areas(self) -> str:
        stages_list: List[str] = []
        
        for stage, stager in self.manager_config.items():
            if not stager:
                continue
            
            workers: List[str] = []
            for order_key in self.workers_order.keys():
                if order_key in stager:
                    workers = stager[order_key]
                    break
            
            stage_title = stage.replace("_", " ", 1).title()
            workers_str = ", ".join(workers)
            
            stages_list.append(f"{stage_title}: [{workers_str}]")
            
        return " | ".join(stages_list)
