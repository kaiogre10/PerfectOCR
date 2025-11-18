# core/pipeline/stagers_factory.py
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from core.factory.main_factory import MainFactory
from typing import Dict, Any, List, Optional

class StagersFactory:
    """
    Reutiliza una única instancia de MainFactory.
    Crea workers y ensambla stagers de forma uniforme.
    """
    def __init__(self, manager_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.manager_config = manager_config
        self.main_factory = MainFactory(manager_config, project_root)
        self.modules_config = self.manager_config.get("modules_config", {})
        self.workers_order = self.manager_config.get("stage_secuence", {})
        self.image_workers = self.workers_order["imagepre_stage"]
        self.preprocessing_workers = self.workers_order.get("preprocessing_stage", [])
        self.ocr_workers = self.workers_order.get("ocr_stage", [])
        self.vectorizing_workers = self.workers_order.get("vector_stage", [])

    def create_image_prep_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> ImagePreparationStager:
        """Crea stager de preparación de imagen con configuraciones específicas del master config."""
        factory = self.main_factory.get_image_preparation_factory()

        # Agregar configuraciones específicas de image_preparation al contexto
        context_with_config: Dict[str, Any] = {
            **context,
            "image_preparation_config": self.modules_config.get("image_preparation", {}),
            "manager_config": self.manager_config
        }
        
        image_workers = factory.create_workers(self.image_workers, context_with_config)
        
        return ImagePreparationStager(
            workers=image_workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_preprocessing_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> Optional[PreprocessingStager]:
        """Crea stager de preprocessing con configuraciones específicas del master config."""
        if not self.preprocessing_workers:
            return None

        factory = self.main_factory.get_preprocessing_factory()
        
        # Agregar configuraciones específicas de preprocessing al contexto
        context_with_config: Dict[str, Any] = {
            **context,
            "preprocessing_config": self.modules_config.get("preprocessing", {}),
            "manager_config": self.manager_config
        }
        
        preprocessing_workers = factory.create_workers(self.preprocessing_workers, context_with_config)
        
        return PreprocessingStager(
            workers=preprocessing_workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_ocr_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> Optional[OCRStager]:
        """Crea stager de OCR con configuraciones específicas del master config."""
        if not self.ocr_workers:
            return None
        
        factory = self.main_factory.get_ocr_factory()
        
        # Agregar configuraciones específicas de OCR al contexto
        context_with_config: Dict[str, Any] = {
            **context,
            "ocr_config": self.modules_config.get("ocr", {}),
            "manager_config": self.manager_config
        }

        ocr_workers = factory.create_workers(self.ocr_workers, context_with_config)
        
        return OCRStager(
            workers=ocr_workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_vectorization_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> Optional[VectorizationStager]:
        """Crea stager de vectorización con configuraciones específicas del master config."""
        if not self.vectorizing_workers:
            return None

        factory = self.main_factory.get_vectorizing_factory()
        
        # Agregar configuraciones específicas de vectorización al contexto
        context_with_config: Dict[str, Any] = {
            **context,
            "vectorization_config": self.modules_config.get("vectorization", {}),
            "manager_config": self.manager_config
        }

        vectorizing_workers = factory.create_workers(self.vectorizing_workers, context_with_config)
        
        return VectorizationStager(
            workers=vectorizing_workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )
