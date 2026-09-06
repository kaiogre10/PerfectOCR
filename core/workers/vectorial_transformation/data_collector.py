# PerfectOCR/core/workers/vectorial_transformation/data_collector.py
import pandas as pd # type: ignore
import logging
from services.log_service import get_time_stamp, now
import numpy as np
from typing import Dict, Any, Tuple, List
from utils.text_utils import format_cuant, get_rfc, get_ids, noramalize_df_text, its_similar, fast_classfier
from core.assets.patterns import umd_patterns
from utils.math_utils import validate_df, check_full_df, decimalice_df
from utils.compiled_utils import validate_text, space_removal
from services.output_service import save_debug_table
from domain.abstract_worker import VectorizationAbstractWorker
from domain.data_formatter import DataFormatter
from domain.class_models import SemantiClass, KeyField, DataKeys

logger = logging.getLogger(__name__)

_umd_patterns = umd_patterns

class FinalStructurer(VectorizationAbstractWorker):
    """"Recolecta los datos importantes y formatea el df dejando todo listo para ingresar a la db."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get('collector', {})
        self.placeholder = worker_config.get("placeholder")
        self.output = config.get("normalized_table")
        self.date_id_format = worker_config.get("date_id_format")
        self.stack = config.get("stack")

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter):
        try:
            df, image_name = self.collect_data(manager)
            if df.empty:
                return False
            
            if self.output or self.stack:
                file_name: str = manager.workflow.metadata.image_name if manager.workflow else "" # type: ignore
                save_debug_table(df, file_name, self.output, self.stack)
                
            payload = self.transform_data(df)

            if manager.store_payload([payload, image_name]):
                return True
        except Exception as e:
            logger.error(f"Error recolectando datos: '{e}'", exc_info=True)
        return False

    def collect_data(self, manager: DataFormatter) -> Tuple[pd.DataFrame, str]:
        structured_data = manager.workflow.table_data if manager.workflow else None
        if structured_data is None:
            return (pd.DataFrame(), "")
        
        df = structured_data.df_table
        metadata = manager.workflow.metadata if manager.workflow else None
        if df is None or df.empty or metadata is None:
            return (pd.DataFrame(), "")

        image_name = metadata.image_name if metadata else ""
        now_id = now()
        date_creation = get_time_stamp(now_id, self.date_id_format)
        idx = f"{image_name}{date_creation}{now_id.microsecond:08d}"
        
        df, totals = self.standarice_df(df, manager, idx)
        if df.empty:
            return (pd.DataFrame(), "")

        polygons = manager.workflow.polygons if manager.workflow else {}
        if not polygons:
            return (pd.DataFrame(), "")
        
        db_values: Dict[str, Any] = {}
        for _, poly_data in enumerate(polygons.values()):
            kf_list = poly_data.key_field
            value = poly_data.ocr_text or ""
            
            if kf_list is not None and value:
                kf = kf_list[0]
                if not kf:
                    continue

                if kf != KeyField.header.value:  # Excluir KeyFields innecesarios
                    if kf == KeyField.rfc_prov.value:  # RFCProveedor
                        value = get_rfc(value)

                    elif kf in (KeyField.total_doc.value, KeyField.total_art.value):
                        value = format_cuant(value)
                    
                    try:
                        field_name = KeyField(kf).name
                    except ValueError:
                        continue
                    
                    db_values[field_name] = value  # 'MontoTotalDocumento': '1024.12'
        
        id_prov = get_ids(image_name, DataKeys.id_proveedor.value)

        db_values.update(totals)
        db_values["image_name"] = image_name
        db_values[DataKeys.id_proveedor.value] = id_prov
        db_values[DataKeys.id_cliente.value] = 1
        db_values[DataKeys.nombre_cliente.value] = "cliente_demo"
        db_values[DataKeys.giro.value] = "giro_demo"
        db_values[DataKeys.proveedor_norm.value] = f"proveedor_demo_{id_prov}"
        db_values[DataKeys.fecha_captura.value] = date_creation

        # logger.info(f"{db_values}")
        manager.reset_data()
        return (df, image_name)
    
    def standarice_df(self, df: pd.DataFrame, manager: DataFormatter, idx: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        mtl_col = df[DataKeys.costo_tran.value]
        c_col = df[DataKeys.cantidad_art.value]
        pu_col = df[DataKeys.precio_unitario.value]
        product_col = df[DataKeys.producto_norm.value]
        df = pd.concat([c_col, product_col, pu_col, mtl_col], axis=1)
        df = self.clean_df(df, manager)
        
        if df.empty or not idx:
            return (pd.DataFrame(), {})

        mtl_col = df[DataKeys.costo_tran.value]
        c_col = df[DataKeys.cantidad_art.value]
        
        mtl_col_dec = decimalice_df(mtl_col)
        # mtl_col_dec = mtl_col.map(lambda x: Decimal(x[0:-1])) # type: ignore
        c_col_dec = decimalice_df(c_col)
        # c_col_dec = c_col.map(lambda x: Decimal(x[0:-1])) # type: ignore
        
        total_total = str(sum(mtl_col_dec))
        total_prod = str(sum(c_col_dec))
        
        totals = {DataKeys.art_calc.value: str(total_prod), DataKeys.total_cal.value: str(total_total), DataKeys.id_registro.value: idx}
        
        # df.insert(loc=0, column=DataKeys.id_registro.value, value=idx, allow_duplicates=True)
        df = df.reset_index(drop=True)
        return df, totals
    
    def clean_df(self, df: pd.DataFrame, manager: DataFormatter) -> pd.DataFrame:
        pro_idx = df.columns.get_loc(DataKeys.producto_norm.value) if DataKeys.producto_norm.value in df.columns else None # type: ignore
        c_idx = df.columns.get_loc(DataKeys.cantidad_art.value) if DataKeys.cantidad_art.value in df.columns else None # type: ignore
        if not manager or not manager.workflow:
            return pd.DataFrame()
        
        all_lines_dict = manager.workflow.all_lines if manager.workflow else {}
        if not all_lines_dict:
            return pd.DataFrame()
        
        line_ids = sorted(all_lines_dict.keys())
        sorted_lines = [all_lines_dict[k] for k in line_ids]

        tabular_lines = np.array([line.line_index for line in sorted_lines if line.lineal_id in line_ids and line.tabular_line], np.uint8)
        df_rows_ids = np.asarray(df.index, np.uint8)    # índices relativos del data frame, posiblemente no continuos
        #totaal_df_rows = df_rows_ids.size

        lineal_ids_df = tabular_lines[df_rows_ids] # Líneas absolutas integradas en el df final

        list_text_df = [line.text.strip() for line in all_lines_dict.values() if line.line_index in tabular_lines]  # Lista del texto de la linea tabular
  #      del_rows = np.setdiff1d(np.arange(df_rows_ids.shape[0], dtype=np.uint8), df_rows_ids)     
   #     del_table_rows = tabular_lines[del_rows]

#        logger.info("\n"f"TABULAR: {tabular_lines}\n"f"ROWS DF:    {df_rows_ids}\n"f"linealiddf: {lineal_ids_df}")
 #       logger.info("\n"f"DOBLETES:  {del_rows}\n"f"TABULAR_DEL: {del_table_rows}")
        
      #  lineal_text = [line.text.strip() for line in all_lines_dict.values() if line.line_index in del_table_rows]
        
    #    logger.info(f"LINE: '{lineal_text}'")

     #   logger.info(f"FILAS CON REASIGNACIÓN:\n"f"{df.iloc[del_rows].to_string(index=True)}")
        for i, r in enumerate(df_rows_ids):
            p_values = str(df.iat[i, pro_idx]) # type: ignore
            cant_values = str(df.iat[i, c_idx]) # type: ignore
            cant_split = cant_values.split(" ")

            if its_similar(cant_split[-1], p_values):
                p_values = p_values[len(cant_split[-1]):]
                p_split = p_values.split(" ")
                if not validate_text(p_split[0]):
                    p_split.remove(p_split[0])
                    df.iat[i, pro_idx] = space_removal(" ".join(p_split))

                df.iat[i, pro_idx] = space_removal(" ".join(p_split))

        if tabular_lines.size != lineal_ids_df.size:
 #           for i, tab_line in enumerate(tabular_lines):
  #              for lineal_ids_df[i] in tab_line:
   #                 logger.info(f"LINEAL DF: {lineal_ids_df[i]} | {df_rows_ids[i]}")

            for i, r in enumerate(df_rows_ids):
                if tabular_lines[r] == lineal_ids_df[i]:

                    p_value = str(df.iat[i, pro_idx])
                    concat_val = list_text_df[i + 1]
                    if p_value.endswith(concat_val):
        #                logger.info(f"TEXTO: \n"f"ALL LINES IGUALES: '{list_text_df[1] + list_text_df[i + 1]}'\n"f"DF: '{p_value}'")
                        orig_p_value = p_value[:-len(concat_val)].strip() # NO REMOVER ESTE STRIP, EVITA GENERAR RUIDO

                        orig_p_value_list: List[str] = orig_p_value.split(" ")
                        con_split_values: List[str] = concat_val.split(" ")

                        p_end = orig_p_value_list[-1]
                        con_beg = con_split_values[0]
                        concat_p_text = space_removal(p_end + " " + con_beg)

        #                logger.info(f"LISTS:    '{orig_p_value_list}' | '{con_split_values}'")
         #               logger.info(f"CONCAT:   '{concat_p_text}' = '{p_end}' + '{con_beg}'")

                        sc, _ = fast_classfier(concat_p_text)
                        po = sc[0]
                        pm = sc[1]   # Texto solitario
                        if po == SemantiClass.QUANTITATIVE or pm == SemantiClass.QUANTITATIVE:
                            continue
                        if po == pm or _umd_patterns.fullmatch(concat_p_text) or (po == SemantiClass.UMD and pm == SemantiClass.NUMERIC):
                            df.iat[i, pro_idx] = space_removal(orig_p_value + concat_val)

        if validate_df(df):
            df = df.map(lambda x: noramalize_df_text(x, self.placeholder)) # type: ignore
            if check_full_df(df):
                logger.debug(f"DF NORMALIZADO:\n{df.to_string(index=True)}")
                return df

        return pd.DataFrame()
    
    def transform_data(self, df: pd.DataFrame) -> str:
        """Devuelve tamaño de cada fila y el df aplanado"""
        plain_df: List[str] = []

        for _, fila in enumerate(df.itertuples(index=False, name=None)):
            fila = list(fila)

            string_row = "".join(fila)
            plain_df.append(string_row)

        plain_text = "".join(plain_df)
        #logger.info(f"TAMAÑO: '{buffer_sizes}' PLAIN TEXT:\n"f"'{plain_text}'")
        return plain_text
