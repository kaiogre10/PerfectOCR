import pandas as pd
from typing import Optional, Dict, Tuple, List, Any
import logging

PROJECT_ROOT = ""

def set_project_root(project_root: str):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root # type: ignore

logger = logging.getLogger(__name__)

class GatewayService:
    def __init__(self, config: Dict[str, Any], output_paths: List[str]):
        all_cols_name: List[str] = config["cols_name"]
        self.cant_name, self.pu_name, self.mtl_name, self.product_name = all_cols_name[0], all_cols_name[1], all_cols_name[2], all_cols_name[3]
        self.output_paths = output_paths

    def transform_df(self, df: pd.DataFrame):
        rows_idx = list(df.index)
        cols_idx = list(df.columns)

        plain_df: bytes = bytes()
        for r in rows_idx:
            for c in cols_idx:
                val = bytes(df.iat[r, c])
                plain_df: bytes