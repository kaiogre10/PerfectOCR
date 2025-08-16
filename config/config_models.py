# services/config_models.py
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Tuple

class ConfigWithNumpy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OutputFlag(ConfigWithNumpy):
    preprocessed_image: bool
    ocr_raw: bool
    cropped_words: bool
    moire_poly: bool
    sp_poly: bool
    gauss_poly: bool
    clahe_poly: bool
    sharp_poly: bool
    binarized_polygons: bool
    problematic_polygons: bool
    reconstructed_lines: bool
    table_lines: bool
    table_structured: bool
    math_max_corrected: bool

class BatchProcessing(ConfigWithNumpy):
    small_batch_limit: int
    max_physical_cores: int
    add_extra_worker: bool
    batch_size_factor: int
    auto_mode: bool

class Processing(ConfigWithNumpy):
    max_workers: int
    valid_image_extensions: Tuple[str, ...]
    batch_mode: bool
    batch_processing: BatchProcessing

class PaddlePaths(ConfigWithNumpy):
    det_model_dir: str
    rec_model_dir: str
    cls_model_dir: str

class PaddleOCRConfig(ConfigWithNumpy):
    use_angle_cls: bool
    lang: str
    show_log: bool
    use_gpu: bool
    enable_mkldnn: bool
    models: PaddlePaths

class PathsConfig(ConfigWithNumpy):
    input_folder: str
    output_folder: str

class PipelineSecuence(ConfigWithNumpy):
    image_preparation: List[str]

class SharpeningConfig(ConfigWithNumpy):
    sharpness_threshold: float
    radius: float
    amount: float

class MoirePercentile(ConfigWithNumpy):
    percentile_threshold: int
    notch_radius: int
    min_distance_from_center: int

class MoireFactor(ConfigWithNumpy):
    mean_factor_threshold: int

class MoireAbsolute(ConfigWithNumpy):
    absolute_threshold: int

class MoireMode(ConfigWithNumpy):
    percentile: MoirePercentile
    factor: MoireFactor
    absolute: MoireAbsolute

class MoireConfig(ConfigWithNumpy):
    mode: MoireMode

class BinarizeQuality(ConfigWithNumpy):
    quality_min: float
    quality_max: float

class Binarization(ConfigWithNumpy):
    c_value: int
    height_thresholds_px: List[int]
    block_sizes_map: List[int]
    quality: BinarizeQuality

class Fragmentador(ConfigWithNumpy):
    min_contour_area: float
    density_std_factor: float
    approx_poly_epsilon: float

class SaltPepper(ConfigWithNumpy):
    kernel_size: int
    salt_pepper_threshold: float
    salt_pepper_low: int
    salt_pepper_high: int

class GaussianConfig(ConfigWithNumpy):
    laplacian_variance_threshold: float
    d: int
    sigma_color: int
    sigma_space: int

class DeskewConfig(ConfigWithNumpy):
    min_angle_for_correction: float
    canny_thresholds: List[int]
    hough_threshold: int
    hough_min_line_length_cap_px: int
    hough_max_line_gap_px: int
    hough_angle_filter_range_degrees: List[int]

class CuttingConfig(ConfigWithNumpy):
    cropping_padding: int

class ImageLoader(ConfigWithNumpy):
    deskew: DeskewConfig
    cutting: CuttingConfig

class ContrastConfigGeneral(ConfigWithNumpy):
    clahe_clip_limit: float
    dimension_thresholds_px: List[int]
    grid_sizes_map: List[List[int]]
  
class ContrastConfigLocal(ConfigWithNumpy):
    window_size: int
    std_dev_threshold: float

class ContrastConfig(ConfigWithNumpy):
    general: ContrastConfigGeneral
    local: ContrastConfigLocal
    
class MathMaxConfig(ConfigWithNumpy):
    total_mtl_abs_tolerance: float
    row_relative_tolerance: float

class TextualConfig(ConfigWithNumpy):
    min_confidence: float

class PreprocessingConfig(ConfigWithNumpy):
    moire: MoireConfig  
    median_filter: SaltPepper 
    bilateral_params: GaussianConfig  
    contrast: ContrastConfig  
    sharpening: SharpeningConfig
    
class OCRConfig(ConfigWithNumpy):
    textual: TextualConfig
    binarize: Binarization  
    fragmentation: Fragmentador  

class DBSCAN(ConfigWithNumpy):
    min_cluster_size: int
    
class Lineal(ConfigWithNumpy):
    overlap: float

class VectorConfig(ConfigWithNumpy):
    lineal: Lineal
    dbscan: DBSCAN
    math_max: MathMaxConfig

class ModulesConfig(ConfigWithNumpy):
    image_loader: ImageLoader
    preprocessing: PreprocessingConfig
    ocr: OCRConfig
    vectorization: VectorConfig

class MasterConfig(ConfigWithNumpy):
    system: Dict[str, Any]
    paths: PathsConfig
    enabled_outputs: OutputFlag
    processing: Processing
    paddle_config: PaddleOCRConfig
    modules: ModulesConfig