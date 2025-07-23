# PerfectOCR/core/coordinators/featuring_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from core.workflow.manufeacturing.binarization import Binarizator
from core.workflow.manufeacturing.word_finder import WordSeparator
from core.workflow.manufeacturing.multifeaturer import MultiFeacturer
from core.workflow.manufeacturing.feature_analyzer import ImageFeatureAnalyzer
from core.workflow.manufeacturing.density_scanner import DensityScanner

logger = logging.getLogger(__name__)

class ManufeactureCoordinator:
    
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        self.workflow_config = config.get('workflow', {})
        self.output_config = config.get('output_config', {})
