# main.py
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_MODE = True
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")

from services.config_service import ConfigService
config_service = ConfigService(DEFAULT_CONFIG_FILE, TEST_MODE)

from app.main_builder import MainBuilder
main_builder = MainBuilder(config_service, PROJECT_ROOT)

import services.system_service as system_service
system_service.set_project_root(PROJECT_ROOT)
system_config = config_service.system_config
system_service.set_system_config(system_config)

import services.output_service as output_service

import services.storage_service as storage_service
storage_service.set_project_root(PROJECT_ROOT)
storage_service.set_config(system_config)

import services.logs_service as log_service
log_service.setup_logging(PROJECT_ROOT)


import logging

logger = logging.getLogger(__name__)

os.environ.update({
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4',
    'FLAGS_use_mkldnn': '1',
})

OUTPUT_PATHS = system_config["output_paths"]
output_paths = [os.path.join(PROJECT_ROOT, folder) for folder in OUTPUT_PATHS]

output_service.set_output_paths(output_paths)
storage_service.set_output_paths(output_paths)

def main():
    system_service.clear_output_folders()
    result = main_builder.activate_main()
    if not result:
        system_service.cleanup_project_cache()
if __name__ == "__main__":
    main()
