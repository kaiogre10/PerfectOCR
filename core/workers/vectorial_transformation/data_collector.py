# PerfectOCR/core/workers/vectorial_transformation/output_structurer.py
import pandas as pd
import logging
from decimal import Decimal
from typing import Dict, Any, Tuple
from core.utils.text_utils import format_cuant, get_rfc, get_ids
from core.utils.data_utils import CONVERSION_KF
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

conversion_kf = CONVERSION_KF

class FinalStructurer(VectorizationAbstractWorker):
    """"
    Recolecta los datos importantes y formatea el df dejando todo listo para ingresar a la db.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter):
        try:
            df, global_data = self.collect_data(manager)

            if manager.save_final_output(df, global_data):
                logger.info("Recolección de datos correcta")
                return True
        except Exception as e:
            logger.error(f"Error recolectando datos: '{e}'", exc_info=True)
        return False

    def collect_data(self, manager: DataFormatter) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        structured_data = manager.workflow.table_data if manager.workflow else None
        if structured_data is None:
            return (pd.DataFrame(), {})
        
        df: pd.DataFrame = structured_data.df_table
        metadata = manager.workflow.metadata if manager.workflow else None
        if df.empty or metadata is None:
            return (pd.DataFrame(), {})
        
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
                    field_name = kf_map_inv.get(kf)
                    if field_name:
                        db_values[field_name] = value  # 'MontoTotalDocumento': '1024.12'
        
        image_name = metadata.image_name if metadata else ""
        id_prov = get_ids(image_name)
        date_creation = metadata.date_creation if metadata else ""

        df_correct, totals = self.structure_df(df, manager)
        db_values.update(totals)
        db_values["image_name"] = image_name
        db_values["id_proveedor"] = id_prov
        db_values["id_cliente"] = 1
        db_values["nombre_cliente"] = "cliente_demo"
        db_values["giro"] = "giro_demo"
        db_values["proveedor_norm"] = f"proveedor_demo_{id_prov}"
        db_values["fecha_captura"] = date_creation

        logger.info(f"{db_values}")
        return (df_correct, db_values)
    
    def structure_df(self, df: pd.DataFrame, manager: DataFormatter) -> Tuple[pd.DataFrame, Dict[str, str]]:
        mtl_col = df["costo_tran"]
        c_col = df["cantidad_art"]
        
        mtl_col_dec = mtl_col.map(lambda x: Decimal(x.strip()))
        c_col_dec = c_col.map(lambda x: Decimal(x.strip()))
        
        total = Decimal(str(sum(mtl_col_dec)))
        total_prod = Decimal(str(sum(c_col_dec)))

        idx: str = manager.workflow.IDRegistro if manager.workflow else ""
        totals = {"art_cal": str(total_prod), "total_cal": str(total), "id_registro": idx}

        pu_col = df["precio_unitario"]
        prodcut_col = df["producto_norm"]
        df = pd.concat([c_col, prodcut_col, pu_col, mtl_col], axis=1)
        df.insert(loc=0, column="id_registro", value=idx, allow_duplicates=True)

        return df, totals