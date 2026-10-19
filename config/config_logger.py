from typing import Dict, Any, List, Tuple, Optional
from functools import cached_property
from services.log_service import get_caller_info, get_logging_info
from core.assets.assets import KF_RANGE, SC_RANGE

_kf_range = KF_RANGE
_sc_range = SC_RANGE

class ConfigLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self._auditate_areas
        
    @property
    def log_config(self):
        return self.config.get("log_config", {})
    
    @property
    def _auditate_process(self) -> bool:
        return self.log_config.get("auditate_process")
    
    @property
    def _all_logs(self) -> bool:
        auditate_process = self._auditate_process
        return auditate_process if auditate_process else self.log_config.get("all_logs")

    @property
    def _auditate_areas(self) -> Dict[str, bool]:
        _auditate_areas: Dict[str, bool] = self.log_config.get("auditate_areas", {})
        if self._auditate_process:
            for key, _ in _auditate_areas.items():
                _auditate_areas[key] = True
                
        return _auditate_areas

    @cached_property
    def _active_auditate_area(self):
        return all((val == False) for val in self._auditate_areas.values())
        
    @cached_property
    def enabled_outputs(self) -> Dict[str, Dict[str, bool]]:
        _enabled_outputs: Dict[str, Dict[str, bool]] = self.config.get("enabled_outputs", {})
        if not self._active_auditate_area:
            return _enabled_outputs
        
        for area, active in self._auditate_areas.items():
            if active:
                category = area + "_outputs"
                for subkey, category in _enabled_outputs.items():
                    category[subkey] = True

        return _enabled_outputs

    @property
    def log_params(self) -> Dict[str, str]:
        _log_params = self.log_config.get("log_params", {})
        if self._all_logs:
            _log_params["console_level"] = 'DEBUG'
            _log_params["file_level"] = 'DEBUG'
        return _log_params

    @cached_property
    def _logs_debug(self) -> Dict[str, bool | List[int]]:
        _logs_debug: Dict[str, bool | List[int]] = self.log_config.get("logs_debug", {})
        _logs_keys = _logs_debug.keys()

        if self._all_logs:
            values_true = _logs_keys

        elif self._auditate_areas.get('ocr'):
            values_true = [key for key in _logs_keys if key.startswith('text_')]

        elif self._auditate_areas.get("vectorization"):
            values_true = [key for key in _logs_keys if key.startswith('table_') or key == "lineal"]

        else:
            values_true = [key[0] for key in _logs_debug.items() if isinstance(key[1], bool) and key[1] == True]

        for key, value in _logs_debug.items():
            if key in values_true:
                if isinstance(value, bool):
                    _logs_debug[key] = True
                
                elif isinstance(value, list):
                    continue

        return _logs_debug
    
    @cached_property
    def logs_debug(self) -> Dict[str, bool | List[int]]:
        _logs_debug = self._logs_debug
        
        def _max_min_vals(type_list: List[int], lenght: int, limits: Tuple[int, int]) -> List[int]:
            if -1 in type_list:
                range_logs = limits
            elif 1 < lenght:
                min_val = max_val = type_list[0]
                boundaries: List[int] = [0, 0]
                for val in type_list[1:]:
                    if val < min_val:
                        min_val = val
                        boundaries[0] = min_val
                    elif val > max_val:
                        max_val = val
                        boundaries[1] = max_val
                range_logs = boundaries
            else:
                if type_list[0] >= limits[1]:
                    range_logs = [limits[1] + (limits[1] + 1)]
                else:
                    range_logs = [type_list[0] + (type_list[0] + 1)]
                    
            return list(range(range_logs[0], range_logs[1]))
        
        kf_list_log = _logs_debug["kf_list_log"]
        semantic_types_log = _logs_debug["semantic_types_log"]
        log_sc_size = len(semantic_types_log)
        log_kf_size = len(kf_list_log)
        
        if not self._all_logs:
            if not _logs_debug["text_keyfields"]:
                kf_range_logs = []
                
            elif log_kf_size == 1 and kf_list_log[0] != -1:
                kf_range_logs = kf_list_log
            else:
                kf_range_logs = _max_min_vals(kf_list_log, log_kf_size, _kf_range)
            
            if not _logs_debug["text_class"]:
                sc_range_logs = []
                
            elif log_sc_size == 1 and semantic_types_log[0] != -1:
                sc_range_logs = semantic_types_log
            else:
                sc_range_logs =  _max_min_vals(semantic_types_log, log_sc_size, _sc_range)
        else:
            kf_range_logs= list(range(_kf_range[0], _kf_range[1] + 1))
            sc_range_logs = list(range(_sc_range[0], _sc_range[1] + 1))

        _logs_debug["kf_list_log"] = kf_range_logs
        _logs_debug["semantic_types_log"] = sc_range_logs
        return _logs_debug
    
    @staticmethod
    def log_active_areas(message: str, manager_config: Optional[List[Tuple[str, List[str]]]] = None) -> str:
        caller_info = get_caller_info()
        message = f"{get_logging_info(caller_info)} {message}"
        if manager_config:
            stages_list = [stager[0].removesuffix("_stage").title() for stager in manager_config if stager and stager[1]]
            msg = f"{', '.join(stages_list) if stages_list else 'SOLO BUILDERS'}"
            return (f"{message} '{msg}'")
        else:
            return (f"{message}")