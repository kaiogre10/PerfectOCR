# core/workers/ocr/data_finder.py
import time
from typing import Dict, Any, Optional, List, Tuple, Set
import logging
import numpy as np
from core.contracts.abstract_worker import OCRAbstractWorker
from domain.data_formatter import DataFormatter
from app.models_builder import ModelsBuilder
from utils.text_utils import contains_quantitative, get_rfc
from utils.compiled_utils import validate_bytes
from core.assets.patterns import cleaner_pattern
from core.assets.assets import KF_REALOCATABLES, KF_IGNORED
from domain.class_models import SemantiClass, KeyField, TypeModels, Params

_cleaner_pattern = cleaner_pattern
_kf_relocatables = KF_REALOCATABLES
_kf_ignored = KF_IGNORED

logger = logging.getLogger(__name__)

class DataFinder(OCRAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self._model = None

    @property
    def model(self) -> Optional[Any]:
        if self._model is None:
            models_builder = ModelsBuilder.get_instance()
            self._model = models_builder.word_finder
            if self._model is None:
                logger.error("WORD FINDER no disponible")
                return None
        return self._model

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            polygon_updates = self._find_data(manager)
            if not polygon_updates:
                return  False
            elif manager.update_key_field(polygon_updates):
                logger.debug(f"DATA FINDER ÉXITOSO")
                return True
            else:
                return  False
        except Exception as e:
            logger.error(f"ERROR BUSCANDO CAMPOS DE CLAVE: {e}", exc_info=True)
        return False

    def _find_data(self, manager: DataFormatter) -> Dict[str, List[int]]:
        if self.model is None:
            raise ModuleNotFoundError("DataFinder no iniciado, no se puede buscar Key FIelds")
        
        time0 = time.perf_counter()
        try:
            polygons = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.error("No hay polygons para procesar")
                return {}

            all_idx = np.array([p.poly_index for p in polygons.values()], dtype=np.int16)

            sc = [p.semantic_clasification for p in polygons.values()]

            sc_eq = np.array([len(set(t)) for t in sc], dtype=np.int16)
            forb_sc = np.array([any(c != SemantiClass.DESCRIPTIVE for c in s) for s in sc], dtype=np.int16)

            mask_sc = (sc_eq == 1) & (forb_sc == True)
            skip_idx = np.compress(mask_sc, all_idx)

            assigned_fields: Set[int] = set()
            polygon_updates: Dict[str, List[int]] = {}
            for _, (pid, poly) in enumerate(polygons.items()):
                if poly.poly_index in skip_idx:
                    continue

                kf = poly.key_field or None
                if kf is not None:
                    logger.debug(f"KeyField redundante: {pid}: '{poly.ocr_text}' | sc: {poly.semantic_clasification}")
                    continue

                ocr_txt = poly.ocr_text or ""

                if len(ocr_txt) < 2:
                    continue

                ocr_text = ocr_txt.lower().encode(TypeModels.ASCII.value)
                if not ocr_text.isalpha():
                    ocr_text = _cleaner_pattern.sub(b"", ocr_text).strip()

                if not validate_bytes(ocr_text):
                    logger.info(f"Texto INVÁLIDO: '{ocr_text}' | sc: {poly.semantic_clasification}")
                    continue
            
                else:
                    valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text, assigned_fields)
                    if not valid_results:
                        continue

                    key_fields: List[int] = [results[Params.KEY_FIELD] for results in valid_results]
                    assigned_fields.update(key_fields)

                    logger.debug(f"VALID RESULTS: {valid_results}")
                    left_overs: List[str] = []
                    
                    if any(k[Params.KEY_FIELD] == KeyField.header.value for k in valid_results):
                        # key_field = [results[Params.KEY_FIELD] for results in valid_results]

                        full_text = valid_results[0][Params.NORM_OCR_TEXT]
                        covered: List[Tuple[int, int]] = []
                        for results in valid_results:
                            piece = results[Params.NORM_OCR_TEXT][results[Params.START]:results[Params.END]]
                            pos = full_text.find(piece)
                            if pos != -1:
                                covered.append((pos, pos + len(piece)))

                        covered.sort()
                        cursor = 0
                        for start, end in covered:
                            gap = full_text[cursor:start].strip()
                            if gap and len(gap) > 2:
                                left_overs.append(gap)
                            cursor = max(cursor, end)
                            
                        tail = full_text[cursor:].strip()
                        if tail and len(tail) > 2:
                            left_overs.append(tail)

                        polygon_updates[pid] = key_fields + [KeyField.header.value] * len(left_overs)
                        continue
                    else:
                        key_field = valid_results[0][Params.KEY_FIELD]
                        polygon_updates[pid] = [key_field]
                        continue
                        
            if polygon_updates:
                logger.debug(f"{len(polygon_updates)} KEY FIELDS ENCONTRADOS EN: {time.perf_counter() - time0:.6}'s,:\n"f"{polygon_updates}")
                return polygon_updates
                # return self.get_key_fields_values(manager, polygon_updates)
            else:
                logger.warning(f"No se hallaron Keywords, tiempo de ejecución: {time.perf_counter() - time0:.6}'s")
                return {}

        except ValueError as e:
            logger.warning(f"Error encontrando keyfields: {e}", exc_info=True)
            return {}
        
    def get_key_fields_values(self, manager: DataFormatter, polygon_updates: Dict[str, List[int]]) -> Dict[str, List[int]]:
        polygons = manager.workflow.polygons if manager.workflow else {}
        if not polygon_updates or not polygons:
            return {}
        
        updates_to_validate: Dict[str, List[int]] = dict(polygon_updates)
        for _, (pid, poly) in enumerate(polygons.items()):
            poly_kf = poly.key_field or []
            if not poly_kf:
                continue

            if any(kf in _kf_ignored for kf in poly_kf):
                continue

            if any(kf in _kf_relocatables for kf in poly_kf):
                if pid not in updates_to_validate:
                    updates_to_validate[pid] = poly_kf

        source_texts: Dict[str, Any] = {}
        for pid, key_fields in updates_to_validate.items():
            source_poly = polygons.get(pid)
            source_texts[pid] = {source_poly.ocr_text or "": key_fields} if source_poly else ""
        # logger.info(f"KF UPDATES ANTES DE REUBICACIÓN: {source_texts}")

        polyid_by_index = {p.poly_index: pid for pid, p in polygons.items()}
        new_updates: Dict[str, List[int]] = {}

        def _is_decimal_kf_value(text: str) -> bool:
            candidate = (text or "").strip()
            if not validate_text(candidate):
                return False
            if candidate.replace(" ", "").isdecimal():
                return True
            return contains_quantitative(candidate)

        def _is_rfc_kf_value(text: str) -> bool:
            candidate = (text or "").strip()
            if not validate_text(candidate):
                return False
            real_rfc = get_rfc(candidate)
            return bool(real_rfc and len(real_rfc) in (12, 13))
        
        def _is_iva_kf_value(text: str) -> bool:
            candidate = (text or "").strip()
            if not validate_text(candidate):
                return False
            if not contains_quantitative(candidate):
                return False
            # Para IVA exigimos señal monetaria explícita para evitar arrastre de totales genéricos.
            return any(ch in candidate for ch in ("$", ","))

        def _find_neighbor(poly_idx: int, kf_value: int) -> str:
            if kf_value == 7:
                validator = _is_rfc_kf_value
            elif kf_value == 8:
                validator = _is_iva_kf_value
            else:
                validator = _is_decimal_kf_value
            for neighbor_idx in (poly_idx + 1, poly_idx - 1):
                neighbor_id = polyid_by_index.get(neighbor_idx)
                if not neighbor_id:
                    continue
                neighbor_poly = polygons.get(neighbor_id)
                if not neighbor_poly or neighbor_poly.key_field is not None:
                    continue
                if validator(neighbor_poly.ocr_text or ""):
                    return neighbor_id
            return ""

        for poly_id, kf_list in updates_to_validate.items():
            if not kf_list:
                continue

            if any(kf in _kf_ignored for kf in kf_list):
                new_updates[poly_id] = kf_list
                continue

            poly_obj = polygons.get(poly_id)
            if not poly_obj:
                neg_kf = [-abs(kf) if kf in _kf_relocatables else kf for kf in kf_list]
                new_updates[poly_id] = neg_kf
                # logger.info(f"KF SIN POLÍGONO ORIGEN, REASIGNADO A NEGATIVO: {poly_id} -> {neg_kf}")
                continue

            poly_idx = poly_obj.poly_index
            should_relocate = any(kf in _kf_relocatables for kf in kf_list)

            if not should_relocate:
                new_updates[poly_id] = kf_list
                # logger.info(f"KF SIN REUBICACIÓN REQUERIDA: {poly_id} -> {kf_list}")
                continue

            target_id = ""
            for kf_value in kf_list:
                if kf_value in _kf_relocatables:
                    target_id = _find_neighbor(poly_idx, kf_value)
                    if target_id:
                        break

            if target_id:
                new_updates[target_id] = kf_list
                source_text = (poly_obj.ocr_text or "").strip()
                target_poly = polygons.get(target_id)
                target_text = ((target_poly.ocr_text or "") if target_poly else "").strip()
                source_existing_kf = poly_obj.key_field or []
                if any(kf in _kf_relocatables for kf in source_existing_kf):
                    source_neg_kf = [-abs(kf) if kf in _kf_relocatables else kf for kf in source_existing_kf]
                    new_updates[poly_id] = source_neg_kf
                logger.debug(
                    f"KF REUBICADO: {poly_id} -> {target_id} | {kf_list} | "
                    f"TEXTO ORIGEN: '{source_text}' | TEXTO DESTINO: '{target_text}'"
                )
                if any(kf in _kf_relocatables for kf in source_existing_kf):
                    logger.info(f"KF ORIGEN MARCADO NEGATIVO TRAS REUBICACIÓN: {poly_id} -> {new_updates[poly_id]}")
            else:
                neg_kf = [-abs(kf) if kf in _kf_relocatables else kf for kf in kf_list]
                new_updates[poly_id] = neg_kf
                source_text = (poly_obj.ocr_text or "").strip()
                logger.debug(
                    f"KF SIN VECINO VÁLIDO, REASIGNADO A NEGATIVO: {poly_id} -> {neg_kf} | ORIG: {kf_list} | "
                    f"TEXTO ORIGEN: '{source_text}'"
                )

        polygon_updates.clear()
        polygon_updates.update(new_updates)
        # logger.info(f"KF UPDATES DESPUÉS DE REUBICACIÓN: {polygon_updates}")
        return polygon_updates
