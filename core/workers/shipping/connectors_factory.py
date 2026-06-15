# core/workers/image_preparation_factory.py
from typing import Dict, Callable, Any
from core.factory.abstract_worker import ConnectorAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.shipping.postgre_local_conector import PostgreLocalConector

class ConnectorWorkersFactory(AbstractBaseFactory[ConnectorAbstractWorker]):
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], ConnectorAbstractWorker]]:
        return {
            'postgre_local': self._create_postgre_conector,
            # "office_local"
            # "google_workspace"
            # "remote"
            # "azure"
            # "csv":
        }

    def _create_postgre_conector(self, context: Dict[str, Any]) -> PostgreLocalConector:
        return PostgreLocalConector(project_root=self.project_root)