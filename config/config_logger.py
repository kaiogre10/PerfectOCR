from typing import Dict, Any, List, Tuple, Optional
from functools import cached_property
from services.log_service import log_simple, get_caller_info, get_logging_info
from core.assets.assets import KF_RANGE, SC_RANGE

_kf_range = KF_RANGE
_sc_range = SC_RANGE

class ConfigLogger:
    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        
    @property
    def log_config(self):
        return self.config.get("log_config", {})
    
    @property
    def _auditate_process(self):
        return self.log_config.get("auditate_process")
    
    @property
    def _all_logs(self) -> bool:
        return self._auditate_process if self._auditate_process else self.log_config.get("all_logs")
        
    @cached_property
    def enabled_outputs(self) -> Dict[str, Dict[str, bool]]:
        _enabled_outputs = self.config.get("enabled_outputs", {})
        if self._auditate_process:
            log_simple("AUDITORIA DE PROCESO COMPLETA ACTIVADA")
            for category in _enabled_outputs.values():
                for subkey in category:
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
    def logs_debug(self) -> Dict[str, bool | List[int]]:
        _logs_debug = self.log_config.get("logs_debug", {})
        
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
            if not _logs_debug["key_fields"]:
                kf_range_logs = []
                
            elif log_kf_size == 1 and kf_list_log[0] != -1:
                kf_range_logs = kf_list_log
            else:
                kf_range_logs = _max_min_vals(kf_list_log, log_kf_size, _kf_range)
            
            if not _logs_debug["seman_clas"]:
                sc_range_logs = []
                
            elif log_sc_size == 1 and semantic_types_log[0] != -1:
                sc_range_logs = semantic_types_log
            else:
                sc_range_logs =  _max_min_vals(semantic_types_log, log_sc_size, _sc_range)
        else:
             kf_range_logs= list(range(_kf_range[0], _kf_range[1] + 1))
             sc_range_logs = list(range(_sc_range[0], _sc_range[1]))
            
             for key, value in _logs_debug.items():
                if isinstance(value, bool):
                    _logs_debug[key] = True
                elif isinstance(value, list):
                    continue
            
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