from config.config_validator import ConfigValidator
from config.config_loader import load_config
from typing import List
import os
import logging

logger = logging.getLogger(__name__)

class ConfigBuilder:
    def __init__(self, config_path: List[str]):
    config = load_config(config_path)