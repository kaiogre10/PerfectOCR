# PerfectOCR/database_builder.py
from services.postgres_service import PostgresService
from typing import Dict, Any

class DataBaseBuilder:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        