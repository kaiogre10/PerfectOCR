# main.py
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app.main_builder as main_builder
main_builder.set_project_root(PROJECT_ROOT)

import services.system_service as cache_service
cache_service.set_project_root(PROJECT_ROOT)

import services.output_service as output_service

import services.storage_service as storage_service
storage_service.set_project_root(PROJECT_ROOT)

import services.logs_service as log_service
log_service.setup_logging(PROJECT_ROOT)

import logging

logger = logging.getLogger(__name__)

os.environ.update({
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4',
    'FLAGS_use_mkldnn': '1',
})

TEST_MODE = True

DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")

DEFAULT_OUTPUT_PATH = ["output"]

output_paths = [os.path.join(PROJECT_ROOT, folder) for folder in DEFAULT_OUTPUT_PATH]

output_service.set_output_paths(output_paths)

storage_service.set_output_paths(output_paths)

def main():
    default_output_paths = [os.path.join(PROJECT_ROOT, folder) for folder in DEFAULT_OUTPUT_PATH]
    cache_service.clear_output_folders(default_output_paths)
    result = main_builder.activate_main(DEFAULT_CONFIG_FILE, TEST_MODE)
    if not result:
        cache_service.cleanup_project_cache()
if __name__ == "__main__":
    main()
