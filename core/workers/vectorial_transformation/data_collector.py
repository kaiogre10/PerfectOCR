# PerfectOCR/core/workers/vectorial_transformation/data_collector.py
import pandas as pd # type: ignore
import logging
import numpy as np
from decimal import Decimal
from typing import Dict, Any, Tuple, List, Optional
from core.utils.text_utils import format_cuant, get_rfc, get_ids, noramalice_df, its_similar, fast_classfier
from core.utils.patterns import umd_patterns
from core.utils.compiled_utils import validate_text
from core.utils.data_utils import CONVERSION_KF
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, AllLines

logger = logging.getLogger(__name__)

_umd_patterns = umd_patterns
conversion_kf = CONVERSION_KF

class FinalStructurer(VectorizationAbstractWorker):
    """"Recolecta los datos importantes y formatea el df dejando todo listo para ingresar a la db."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('math_max', {})
        all_cols_name: List[str] = worker_config["cols_name"]
        self.cant_name, self.pu_name, self.mtl_name, self.product_name, self.id_registro = all_cols_name[0], all_cols_name[1], all_cols_name[2], all_cols_name[3], all_cols_name[4]
        self.separator = worker_config.get("separator", "")

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter):
        try:
            df, global_data = self.collect_data(manager)

            if manager.save_final_output(df, global_data):
                context = context
                context = {}
                return True
        except Exception as e:
            logger.error(f"Error recolectando datos: '{e}'", exc_info=True)
        return False

    def collect_data(self, manager: DataFormatter) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        structured_data = manager.workflow.table_data if manager.workflow else None
        if structured_data is None:
            return (pd.DataFrame(), {})
        
        df: Optional[pd.DataFrame] = structured_data.df_table
        metadata = manager.workflow.metadata if manager.workflow else None
        if df is None or df.empty or metadata is None:
            return (pd.DataFrame(), {})

        df, totals = self.standarice_df(df, manager)
        kf_map_inv = {v: k for k, v in conversion_kf.items()}
        polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
        db_values: Dict[str, Any] = {}
        for poly_data in polygons.values():
            kf_list = poly_data.key_field
            value = poly_data.ocr_text or ""
            
            if kf_list is not None and value:
                kf = kf_list[0]
                if kf not in (5, 6):  # Excluir KeyFields innecesarios
                    if kf == 7:  # RFCProveedor
                        value = get_rfc(value)

                    elif kf in (1, 2):
                        value = format_cuant(value)
                    # Mapear el código numérico al nombre del campo
                    field_name: str = kf_map_inv.get(kf)
                    if field_name is not None:
                        db_values[field_name] = value  # 'MontoTotalDocumento': '1024.12'
        
        image_name = metadata.image_name if metadata else ""
        id_prov = get_ids(image_name, "prov")
        date_creation = metadata.date_creation if metadata else ""

        db_values.update(totals)
        db_values["image_name"] = image_name
        db_values["id_proveedor"] = id_prov
        db_values["id_cliente"] = 1
        db_values["nombre_cliente"] = "cliente_demo"
        db_values["giro"] = "giro_demo"
        db_values["proveedor_norm"] = f"proveedor_demo_{id_prov}"
        db_values["fecha_captura"] = date_creation

        # logger.info(f"{db_values}")
        return (df, db_values)
    
    def standarice_df(self, df: pd.DataFrame, manager: DataFormatter) -> Tuple[pd.DataFrame, Dict[str, str]]:

        mtl_col = df[self.mtl_name]
        c_col = df[self.cant_name]
        pu_col = df[self.pu_name]
        product_col = df[self.product_name]
        df = pd.concat([c_col, product_col, pu_col, mtl_col], axis=1)
        df = self.clean_df(df, manager)

        mtl_col = df[self.mtl_name]
        c_col = df[self.cant_name]
        
        mtl_col_dec = mtl_col.map(lambda x: Decimal(x[0:-1])) # type: ignore
        c_col_dec = c_col.map(lambda x: Decimal(x[0:-1])) # type: ignore
        
        total_total = Decimal(str(sum(mtl_col_dec)))
        total_prod = Decimal(str(sum(c_col_dec)))

        idx: str = manager.workflow.IDRegistro if manager.workflow else ""
        totals = {"art_cal": str(total_prod), "total_cal": str(total_total), self.id_registro: idx}
        
        # df.insert(loc=0, column=self.id_registro, value=idx, allow_duplicates=True)
        df = df.reset_index(drop=True)
        return df, totals
    
    def clean_df(self, df: pd.DataFrame, manager: DataFormatter) -> pd.DataFrame:
        pro_idx = df.columns.get_loc(self.product_name) if self.product_name in df.columns else None # type: ignore
        c_idx = df.columns.get_loc(self.cant_name) if self.cant_name in df.columns else None # type: ignore

        all_lines_dict: Dict[str, AllLines] = manager.workflow.all_lines if manager.workflow else {}
        line_ids = sorted(all_lines_dict.keys())
        sorted_lines = [all_lines_dict[k] for k in line_ids]

        tabular_lines = np.array([line.line_index for line in sorted_lines if line.lineal_id in line_ids and line.tabular_line], np.uint8)
        df_rows_ids = np.asarray(df.index, np.uint8)
        lineal_ids_df = tabular_lines[df_rows_ids]
        list_text_df = [line.text for line in all_lines_dict.values() if line.line_index in tabular_lines]
        # logger.info(f"\n"f"TABULAR: '{tabular_lines} SIZE: {tabular_lines.size}'\n"f"ROWS DF'{df_rows_ids} SIZE: {df_rows_ids.size}'\n"f"linealiddf: {lineal_ids_df} SIZE: {lineal_ids_df.size}")

        for i, r in enumerate(df_rows_ids):
            p_values = str(df.iat[i, pro_idx]) # type: ignore
            cant_values = str(df.iat[i, c_idx]) # type: ignore
            cant_split = cant_values.split(" ")
            if its_similar(cant_split[-1], p_values):
                p_values = p_values[len(cant_split[-1]):]
                p_split = p_values.split(" ")
                if not validate_text(p_split[0]):
                    p_split.remove(p_split[0])
                    df.iat[i, pro_idx] = " ".join(p_split).strip()

                df.iat[i, pro_idx] = " ".join(p_split).strip()

        if tabular_lines.size != lineal_ids_df.size:
            for i, r in enumerate(df_rows_ids):
                if tabular_lines[r] == lineal_ids_df[i]:
                    p_value = str(df.iat[i, pro_idx])
                    concat_val = list_text_df[i + 1]
                    if p_value.endswith(concat_val):
                        orig_p_value = p_value[:-len(concat_val)].strip()
                        orig_p_value_list: List[str] = orig_p_value.split(" ")

                        con_split_values: List[str] = concat_val.split(" ")

                        p_end = orig_p_value_list[-1].strip()
                        con_beg = con_split_values[0]
                        concat_p_text = (p_end + " " + con_beg)
                        sc, _ = fast_classfier(concat_p_text)
                        po = sc[0]
                        pm = sc[1]   # Texto solitario
                        if po == 4 or pm == 4:
                            continue
                        if po == pm or _umd_patterns.fullmatch(concat_p_text) or (po == 2 and pm == 5):
                            df.iat[i, pro_idx] = (orig_p_value +  concat_val)

        df = df.map(lambda x: noramalice_df(x, self.separator)) # type: ignore
        #logger.info(f"DF NORMALIZADO:\n{df.to_string(index=False)}")
        return df