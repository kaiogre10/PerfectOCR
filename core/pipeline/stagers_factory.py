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
        self.modules_config = manager_config
        self.image_workers: List[str] = self.modules_config.get("image_preparation", {}).get("imagepre_stage", [])
        self.preprocessing_workers = self.modules_config.get("preprocessing", {}).get("preprocessing_stage", [])
        self.ocr_workers: List[str] = self.modules_config.get("ocr", {}).get("ocr_stage", [])
        self.vectorizing_workers = self.modules_config.get("vectorization", {}).get("vector_stage", [])
        self.main_factory = MainFactory(self.modules_config, project_root)

    def create_image_prep_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> ImagePreparationStager:
        """Crea stager de preparación de imagen con configuraciones específicas del master config."""
        factory = self.main_factory.get_image_preparation_factory()
        if "polygon_extractor" in self.image_workers:
            if not "geometry_detector" in self.image_workers:
                self.image_workers.remove("polygon_extractor")

        image_workers = factory.create_workers(self.image_workers, context)
        
        return ImagePreparationStager(
            workers=image_workers,
            stage_config=self.modules_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_preprocessing_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> Optional[PreprocessingStager]:
        """Crea stager de preprocessing con configuraciones específicas del master config."""
        if not self.preprocessing_workers:
            return None
        
        factory = self.main_factory.get_preprocessing_factory()
        if factory is None:
            return None
        
        preprocessing_workers = factory.create_workers(self.preprocessing_workers, context)
        
        return PreprocessingStager(
            workers=preprocessing_workers,
            stage_config=self.modules_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_ocr_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> Optional[OCRStager]:
        """Crea stager de OCR con configuraciones específicas del master config."""
        if not self.ocr_workers:
            return None
        
        factory = self.main_factory.get_ocr_factory()
        if factory is None:
            return None
        
        ocr_workers = factory.create_workers(self.ocr_workers, context)
        
        return OCRStager(
            workers=ocr_workers,
            stage_config=self.modules_config,
            output_paths=output_paths,
            project_root=self.project_root
        )
    
    def create_vectorization_stager(self, context: Dict[str, Any], output_paths: List[str] | str) -> Optional[VectorizationStager]:
        """Crea stager de vectorización con configuraciones específicas del master config."""
        if not self.vectorizing_workers:
            return None

        factory = self.main_factory.get_vectorizing_factory()
        if factory is None:
            return None
            
        vectorizing_workers = factory.create_workers(self.vectorizing_workers, context)
        
        return VectorizationStager(
            workers=vectorizing_workers,
            stage_config=self.modules_config,
            output_paths=output_paths,
            project_root=self.project_root
        )
