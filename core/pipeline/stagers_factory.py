# core/pipeline/stagers_factory.py
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory
from core.workers.ocr.ocr_factory import OCRFactory
from core.workers.vectorial_transformation.vectorizing_factory import VectorizingFactory
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from typing import Dict, Any, Optional

class StagersFactory:
    """Crea workers y ensambla stagers de forma uniforme."""
    def __init__(self, manager_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        modules_config = manager_config
        self.image_preparation_config = modules_config.get("image_preparation", {})
        self.preprocessing_config = modules_config.get('preprocessing', {})
        self.ocr_config = modules_config.get('ocr', {})
        self.vectorizing_config = modules_config.get('vectorization', {})

    def create_image_prep_stager(self, context: Optional[Dict[str, Any]] = None) -> Optional[ImagePreparationStager]:
        """Crea stager de preparación de imagen con configuraciones específicas del master config."""
        image_preparation_config = self.image_preparation_config
        if not image_preparation_config:
            return None

        image_workers = image_preparation_config["imagepre_stage"]
        if not image_workers:
            return None

        worker_config = image_preparation_config.get("worker_config")
        factory = self.get_image_preparation_factory(worker_config)
        if factory is None:
            return None

        if "polygon_extractor" in image_workers:
            if "geometry_detector" not in image_workers:
                image_workers.remove("polygon_extractor")

        image_workers = factory.create_workers(image_workers, context)

        return ImagePreparationStager(
            workers=image_workers,
            stage_config=image_preparation_config,
            project_root=self.project_root
        )

    def create_preprocessing_stager(self, context: Optional[Dict[str, Any]] = None) -> Optional[PreprocessingStager]:
        """Crea stager de preprocessing con configuraciones específicas del master config."""
        preprocessing_config = self.preprocessing_config
        if not preprocessing_config:
            return None

        preprocessing_workers = preprocessing_config["preprocessing_stage"]

        if not preprocessing_workers:
            return None

        factory = self.get_preprocessing_factory(preprocessing_config)
        if factory is None:
            return None

        preprocessing_workers = factory.create_workers(preprocessing_workers, context)

        return PreprocessingStager(
            workers=preprocessing_workers,
            stage_config=preprocessing_config,
            project_root=self.project_root,
        )

    def create_ocr_stager(self, context: Optional[Dict[str, Any]] = None) -> Optional[OCRStager]:
        """Crea stager de OCR con configuraciones específicas del master config."""
        ocr_config = self.ocr_config
        if not ocr_config:
            return None

        ocr_workers = self.ocr_config["ocr_stage"]
        if not ocr_workers:
            return None

        factory = self.get_ocr_factory(ocr_config)
        if factory is None:
            return None

        ocr_workers = factory.create_workers(ocr_workers, context)

        return OCRStager(
            workers=ocr_workers,
            stage_config=ocr_config,
            project_root=self.project_root
        )

    def create_vectorization_stager(self, context: Optional[Dict[str, Any]] = None) -> Optional[VectorizationStager]:
        """Crea stager de vectorización con configuraciones específicas del master config."""
        vectorizing_config = self.vectorizing_config
        if not vectorizing_config:
            return None

        vectorizing_workers = self.vectorizing_config["vector_stage"]
        if not vectorizing_workers:
            return None

        factory = self.get_vectorizing_factory(vectorizing_config)
        if factory is None:
            return None

        vectorizing_workers = factory.create_workers(vectorizing_workers, context)

        return VectorizationStager(
            workers=vectorizing_workers,
            stage_config=vectorizing_config,
            project_root=self.project_root
        )

    def get_image_preparation_factory(self, image_preparation_config: Dict[str, Any]) -> Optional[ImagePreparationFactory]:
        return None if not image_preparation_config else ImagePreparationFactory(image_preparation_config, self.project_root)

    def get_preprocessing_factory(self, preprocessing_config: Dict[str, Any]) -> Optional[PreprocessingFactory]:
        return None if not preprocessing_config else PreprocessingFactory(preprocessing_config, self.project_root)

    def get_ocr_factory(self, ocr_config: Dict[str, Any]) -> Optional[OCRFactory]:
        return None if not ocr_config else OCRFactory(ocr_config, self.project_root)

    def get_vectorizing_factory(self, vectorizing_config: Dict[str, Any]) -> Optional[VectorizingFactory]:
        return None if not vectorizing_config else VectorizingFactory(vectorizing_config, self.project_root)