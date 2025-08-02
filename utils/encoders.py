# PerfectOCR/core/utils/encoders.py
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Si tienes objetos con _asdict() (como namedtuples)
        if hasattr(obj, '_asdict'):
            return obj._asdict()
        return super(NumpyEncoder, self).default(obj)
