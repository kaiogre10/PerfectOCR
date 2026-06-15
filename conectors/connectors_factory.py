# core/workers/image_preparation_factory.py
# from typing import Dict, Callable, Any
# from core.factory.abstract_worker import ImagePrepAbstractWorker
# from core.factory.abstract_factory import AbstractBaseFactory
# from core.workers.image_preparation.image_loader import ImageLoader

# class ImagePreparationFactory(AbstractBaseFactory[ImagePrepAbstractWorker]):
#     def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], ImagePrepAbstractWorker]]:
#         return {
#             'postgre_local': self._create_postgre_conector,
#             # "office_local"
#             # "google_workspace"
#             # "remote"
#             # "azure"
#             # "csv":
#         }

#     def _create_postgre_conector(self,) -> ImageLoader:
#         return ImageLoader(config=self.module_config, project_root=self.project_root)