from typing import Dict, Any, Optional, List
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from core.factory.main_factory import MainFactory

class StagersFactory:
    """
    Fábrica centralizada de stagers.
    - Reutiliza una única instancia de MainFactory.
    - Crea workers y ensambla stagers de forma uniforme.
    """

    def __init__(self, modules_config: Dict[str, Any], manager_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.manager_config = manager_config
        self.main_factory = MainFactory(modules_config, project_root)

        # Orden de workers por etapa (editable sin tocar main)
        self.workers_order = {
            "image_preparation": ["image_loader", "cleaner", "angle_corrector", "geometry_detector", "polygon_extractor"],
            "preprocessing": ["moire", "sp", "gauss", "clahe", "sharp"],
            "ocr": ["paddle_wrapper", "text_cleaner"],
            "vectorization": ["lineal", "dbscan", "table_structurer", "math_max"]
        }

    def create_image_prep_stager(self, context: Dict[str, Any], output_paths: Optional[List[str]]) -> ImagePreparationStager:
        """Crea stager de preparación de imagen con ImageLoader incluido."""
        factory = self.main_factory.get_image_preparation_factory()
        workers = factory.create_workers(self.workers_order["image_preparation"], context)
        
        return ImagePreparationStager(
            workers=workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_preprocessing_stager(self, context: Dict[str, Any], output_paths: Optional[List[str]]) -> PreprocessingStager:
        factory = self.main_factory.get_preprocessing_factory()
        workers = factory.create_workers(self.workers_order["preprocessing"], context)
        
        return PreprocessingStager(
            workers=workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_ocr_stager(self, context: Dict[str, Any], output_paths: Optional[List[str]]) -> OCRStager:
        factory = self.main_factory.get_ocr_factory()
        workers = factory.create_workers(self.workers_order["ocr"], context)
        
        return OCRStager(
            workers=workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    def create_vectorization_stager(self, context: Dict[str, Any], output_paths: Optional[List[str]]) -> VectorizationStager:
        factory = self.main_factory.get_vectorizing_factory()
        workers = factory.create_workers(self.workers_order["vectorization"], context)
        
        return VectorizationStager(
            workers=workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )