# core/workers/ocr/data_finder.py
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
from domain.abstract_worker import OCRAbstractWorker
from domain.data_formatter import DataFormatter
from app.models_builder import ModelsBuilder
from utils.text_utils import contains_quantitative, get_rfc
from utils.compiled_utils import validate_text

kf_decimals = {1, 2, 3, 4, 8}
kf_relocatables = set(kf_decimals.union({7}))
kf_ignored = {6, 9}

logger = logging.getLogger(__name__)

class DataFinder(OCRAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self._model = None

    @property
    def model(self) -> Optional[Any]:
        try:
            if self._model is None: #type: ignore
                models_builder = ModelsBuilder.get_instance()
                self._model = models_builder.word_finder #type: ignore
            return self._model #type: ignore

        except ModuleNotFoundError as e:
            logger.error(f"DataFinder: Modelo de búsqueda no disponible: {e}", exc_info=True)
        return None

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        polygon_updates = self._find_data(manager)
        if not polygon_updates:
            return  False
        elif manager.update_key_field(polygon_updates):
            logger.debug(f"DATA FINDER ÉXITOSO")
            return True
        else:
            return  False

    def _find_data(self, manager: DataFormatter) -> Dict[str, List[int]]:
        if self.model is None:
            logger.error("DataFinder no iniciado, no se puede buscar Key FIelds")
            return {}
        
        time0 = time.perf_counter()
        try:
            polygons = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.error("No hay polygons para procesar")
                return {}
        
            processed_count = 0
            polygon_updates: Dict[str, List[int]] = {}
            skipped_semantic = 0
            sc_forb = {0, 2, 4, 5}

            all_idx = np.array([p.poly_index for p in polygons.values()], np.int16)

            sc = [p.semantic_clasification for p in polygons.values()]
            texts = [(p.ocr_text or "") for p in polygons.values()]

            texts_length = np.array([len(t) for t in texts])
            decimal_p = np.array([t.isnumeric() for t in texts])

            # sc_length = np.array([len(c) for c in sc])
            sc_eq = np.array([len(set(t)) for t in sc])
            forb_sc = np.array([any(c in sc_forb for c in s) for s in sc])

            mask_sc = (sc_eq == 1) & (forb_sc == True) 
            mask_len = (texts_length < 2) & (decimal_p == True)
            mask = mask_sc | mask_len
            skip_idx = np.compress(mask, all_idx).tolist()

            for _, (pid, poly) in enumerate(polygons.items()):
                if poly.poly_index in skip_idx:
                    # logger.info(f"{pid} Omitido: '{poly.ocr_text}' | sc: {poly.semantic_clasification}")
                    skipped_semantic += 1
                    continue

                processed_count += 1
                kf = poly.key_field or None
                if kf is not None:
                    skipped_semantic += 1
                    logger.debug(f"KeyField redundante en WODR FINDER {pid}: '{poly.ocr_text}' | sc: {poly.semantic_clasification}")
                    continue

                ocr_text = poly.ocr_text or ""

                if len(ocr_text) < 3:
                    continue

                if not validate_text(ocr_text):
                    logger.debug(f"Texto INVÁLIDO: '{ocr_text}' | sc: {poly.semantic_clasification}")
                    skipped_semantic += 1
                    continue
            
                else:
                    valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text.lower())
                    if not valid_results:
                        continue
                    
                    logger.info(f"VALID RESULTS: {valid_results}")
                    left_overs: List[str] = []
                    if any(k['key_field'] == 6 for k in valid_results):
                        key_field = [results['key_field'] for results in valid_results]

                        full_text = valid_results[0]['norm_ocr_text']
                        covered: List[Tuple[int, int]] = []
                        for results in valid_results:
                            piece = results['norm_ocr_text'][results['start']:results['end']]
                            pos = full_text.find(piece)
                            if pos != -1:
                                covered.append((pos, pos + len(piece)))

                        covered.sort()
                        cursor = 0
                        for start, end in covered:
                            gap = full_text[cursor:start].strip()
                            if gap:
                                left_overs.append(gap)
                            cursor = max(cursor, end)
                        tail = full_text[cursor:].strip()
                        if tail:
                            left_overs.append(tail)

                        polygon_updates[pid] = key_field + [6] * len(left_overs)
                        continue
                    else:
                        key_field = valid_results[0]['key_field']
                        polygon_updates[pid] = [key_field]
                        continue
                        
            if polygon_updates:
                logger.info(f"KEY FIELDS ENCONTRADOS: '{polygon_updates}', en: {time.perf_counter() - time0:.6}'s, {skipped_semantic} omisiones")
                return polygon_updates
                # return self.get_key_fields_values(manager, polygon_updates)
            else:
                logger.warning(f"No se hallaron Keywords, tiempo de ejecución: {time.perf_counter() - time0:.6}'s")
                return {}

        except ValueError as e:
            logger.warning(f"Error encontrando keyfields: {e}")
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
            if any(kf in kf_ignored for kf in poly_kf):
                continue
            if any(kf in kf_relocatables for kf in poly_kf):
                if pid not in updates_to_validate:
                    updates_to_validate[pid] = poly_kf

        source_texts: Dict[str, Any] = {}
        for pid, key_fields in updates_to_validate.items():
            source_poly = polygons.get(pid)
            source_texts[pid] = {source_poly.ocr_text or "": key_fields} if source_poly else ""
        # logger.info(f"KF UPDATES ANTES DE REUBICACIÓN: {source_texts}")
        
        if not polygons:
            return polygon_updates

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

            if any(kf in kf_ignored for kf in kf_list):
                new_updates[poly_id] = kf_list
                continue

            poly_obj = polygons.get(poly_id)
            if not poly_obj:
                neg_kf = [-abs(kf) if kf in kf_relocatables else kf for kf in kf_list]
                new_updates[poly_id] = neg_kf
                # logger.info(f"KF SIN POLÍGONO ORIGEN, REASIGNADO A NEGATIVO: {poly_id} -> {neg_kf}")
                continue

            poly_idx = poly_obj.poly_index
            should_relocate = any(kf in kf_relocatables for kf in kf_list)

            if not should_relocate:
                new_updates[poly_id] = kf_list
                # logger.info(f"KF SIN REUBICACIÓN REQUERIDA: {poly_id} -> {kf_list}")
                continue

            target_id = ""
            for kf_value in kf_list:
                if kf_value in kf_relocatables:
                    target_id = _find_neighbor(poly_idx, kf_value)
                    if target_id:
                        break

            if target_id:
                new_updates[target_id] = kf_list
                source_text = (poly_obj.ocr_text or "").strip()
                target_poly = polygons.get(target_id)
                target_text = ((target_poly.ocr_text or "") if target_poly else "").strip()
                source_existing_kf = poly_obj.key_field or []
                if any(kf in kf_relocatables for kf in source_existing_kf):
                    source_neg_kf = [-abs(kf) if kf in kf_relocatables else kf for kf in source_existing_kf]
                    new_updates[poly_id] = source_neg_kf
                logger.debug(
                    f"KF REUBICADO: {poly_id} -> {target_id} | {kf_list} | "
                    f"TEXTO ORIGEN: '{source_text}' | TEXTO DESTINO: '{target_text}'"
                )
                if any(kf in kf_relocatables for kf in source_existing_kf):
                    logger.info(f"KF ORIGEN MARCADO NEGATIVO TRAS REUBICACIÓN: {poly_id} -> {new_updates[poly_id]}")
            else:
                neg_kf = [-abs(kf) if kf in kf_relocatables else kf for kf in kf_list]
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
