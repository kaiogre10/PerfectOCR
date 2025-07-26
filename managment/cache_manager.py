# perfectocr/management/cache_manager.py
import shutil
import yaml
import os
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, config_path: str):
        self.config = self._load_yaml_config(config_path)
        self.workflow_config = self.config.get('workflow', {})
        self.roots_config = self.config.get('roots_paths', {})
    

    def _load_yaml_config(self, path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error cargando config: {e}")
            return {}

    def _delete_item(self, path: str):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                logger.info(f"Eliminada carpeta de caché: {path}")
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
        project_root = self.workflow_config.get('project_root')
        if not project_root: return

        folder_names = self.roots_config.get('folder_names_to_delete', [])
        logger.info("--- Limpieza Final: Eliminando caché del proyecto ---")
        for dirpath, dirnames, _ in os.walk(project_root):
            for d in list(dirnames):
                if d in folder_names:
                    self._delete_item(os.path.join(dirpath, d))
                    dirnames.remove(d)