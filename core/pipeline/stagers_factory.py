# core/pipeline/stagers_factory.py
from typing import Dict, Any, Optional, List
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from core.factory.main_factory import MainFactory
from core.workers.image_preparation.image_loader import ImageLoader

class StagersFactory:
    """
    Fábrica de stagers (solo creación). No procesa.
    - Reutiliza una única instancia de MainFactory.
    - Crea los workers necesarios y arma los stagers con sus firmas actuales.
    - No modifica la lógica interna de los stagers.
    """

    def __init__(self, modules_config: Dict[str, Any], manager_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.manager_config = manager_config
        # Instancia única de MainFactory para todo el ciclo
        self.main_factory = MainFactory(modules_config, project_root)

        # Atajos a las factories de cada módulo
        self.image_load_factory = self.main_factory.get_image_preparation_factory()
        self.preprocessing_factory = self.main_factory.get_preprocessing_factory()
        self.ocr_factory = self.main_factory.get_ocr_factory()
        self.vectorizing_factory = self.main_factory.get_vectorizing_factory()

        # Listas de workers por etapa (editable desde aquí sin tocar el main)
        self.image_prep_workers_order: List[str] = [
            "cleaner", "angle_corrector", "geometry_detector", "polygon_extractor"
        ]
        self.preprocessing_workers_order: List[str] = [
            "moire", "sp", "gauss", "clahe", "sharp"
        ]
        self.ocr_workers_order: List[str] = [
            "paddle_wrapper", "text_cleaner"
        ]
        self.vectorization_workers_order: List[str] = [
            "lineal", "dbscan", "table_structurer", "math_max"
        ]

    # Stager 1: Preparación de imagen (tu clase actual recibe factory + image_loader)
    def create_image_prep_stager(self, image_loader: ImageLoader, context: Dict[str, Any]) -> ImagePreparationStager:
        workers = self.image_load_factory.create_workers(self.preprocessing_workers_order, context)
        return ImagePreparationStager(
            workers=workers,
            image_loader = image_loader,
            project_root=self.project_root
        )

    # Stager 2: Preprocesamiento (tu clase actual recibe lista de workers)
    def create_preprocessing_stager(self, context: Dict[str, Any], output_paths: Optional[List[str]]) -> PreprocessingStager:
        workers = self.preprocessing_factory.create_workers(self.preprocessing_workers_order, context)
        return PreprocessingStager(
            workers=workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    # Stager 3: OCR (tu clase actual recibe lista de workers)
    def create_ocr_stager(self, context: Dict[str, Any], output_paths: Optional[List[str]]) -> OCRStager:
        workers = self.ocr_factory.create_workers(self.ocr_workers_order, context)
        return OCRStager(
            workers=workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )

    # Stager 4: Vectorización (tu clase actual recibe lista de workers)
    def create_vectorization_stager(self, context: Dict[str, Any], output_paths: Optional[List[str]]) -> VectorizationStager:
        workers = self.vectorizing_factory.create_workers(self.vectorization_workers_order, context)
        return VectorizationStager(
            workers=workers,
            stage_config=self.manager_config,
            output_paths=output_paths,
            project_root=self.project_root
        )