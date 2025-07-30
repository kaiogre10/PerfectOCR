# PerfectOCR/core/utils/__init__.py
from .encoders import NumpyEncoder
from .geometric import (
    get_polygon_bounds, get_polygon_height, get_polygon_width,
    get_polygon_y_center, get_polygon_x_center, get_shapely_polygon,
    calculate_iou, enrich_word_data_with_geometry, tighten_geometry
)
from .output_handlers import OutputHandler, TextOutputHandler

__all__ = [
    'NumpyEncoder',
    'get_polygon_bounds', 'get_polygon_height', 'get_polygon_width',
    'get_polygon_y_center', 'get_polygon_x_center', 'get_shapely_polygon',
    'calculate_iou', 'enrich_word_data_with_geometry', 'tighten_geometry',
    'OutputHandler', 'TextOutputHandler'
]
