# perfectocr/management/cache_manager.py
import shutil
import os
import logging
from services.config_service import ConfigService

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        self.workflow_config = config_service.workflow_config
        self.roots_config = config_service.roots_config

    def _delete_item(self, path: str):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                logger.debug(f"Eliminada carpeta de caché: {path}")
            else:
                os.remove(path)
        except OSError as e:
            logger.error(f"Error al eliminar {path}: {e}")

    def clear_output_folders(self):
        folders_to_empty = self.roots_config.get('folders_to_empty', [])
        logger.info("--- Limpieza Inicial: Vaciando carpetas de salida ---")
        for folder_path in folders_to_empty:
            if not os.path.isdir(folder_path):
                continue
            for item_name in os.listdir(folder_path):
                self._delete_item(os.path.join(folder_path, item_name))

    def cleanup_project_cache(self):
        """Limpia el caché del proyecto usando la configuración."""
        project_root = self.workflow_config.get('project_root')
        if not project_root:
            logger.warning("No se encontró project_root en la configuración")
            return

        folder_names = self.roots_config.get('folder_names_to_delete', [])
        logger.info("--- Limpieza Final: Eliminando caché del proyecto ---")

        for dirpath, dirnames, _ in os.walk(project_root):
            for d in list(dirnames):
                if d in folder_names:
                    self._delete_item(os.path.join(dirpath, d))
                    dirnames.remove(d)