# services/config_models.py
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Tuple

# Configuración para permitir np.ndarray
class ConfigWithNumpy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class LoggingConfig(ConfigWithNumpy):
    log_file: str
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    console_format: str = "%(levelname)s:%(name)s:%(lineno)d - %(message)s"
    file_format: str = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(module)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

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
    temp_folder: str

class PipelineSecuence(ConfigWithNumpy):
    image_preparation: List[str]
    preprocess: List[str]

class SharpeningConfig(ConfigWithNumpy):
    sharpness_threshold: float
    radius: float
    amount: float

class Binarization(ConfigWithNumpy):
    c_value: int
    height_thresholds_px: List[int]
    block_sizes_map: List[int]
    quality_min: float
    quality_max: float

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

class ClaheConfig(ConfigWithNumpy):
    clahe_clip_limit: float
    dimension_thresholds_px: List[int]
    grid_sizes_map: List[List[int]]
    window_size: int
    std_dev_threshold: float

class MoirePercentile(ConfigWithNumpy):
    percentile_threshold: int
    notch_radius: int
    min_distance_from_center: int

class MoireFactor(ConfigWithNumpy):
    mean_factor_threshold: int

class MoireAbsolute(ConfigWithNumpy):
    absolute_threshold: int

class MoireConfig(ConfigWithNumpy):
    percentile: MoirePercentile
    factor: MoireFactor
    absolute: MoireAbsolute

class PreprocessingConfig(ConfigWithNumpy):
    salt_pepper: SaltPepper
    gaussian: GaussianConfig
    contrast: ClaheConfig
    binarize: Binarization
    fragmentation: Fragmentador
    sharpening: SharpeningConfig
    moire: MoireConfig

class VectorConfig(ConfigWithNumpy):
    min_cluster_size: int

class ModulesConfig(ConfigWithNumpy):
    image_loader: Dict[str, Any]
    preprocessing: PreprocessingConfig
    vectorization: VectorConfig

class Cleanup(ConfigWithNumpy):
    folder_extensions_to_delete: List[str]
    file_extensions_to_delete: List[str]
    folders_to_empty: List[str]
    folders_to_exclude: List[str]
    files_to_exclude_by_name: List[str]

class MasterConfig(ConfigWithNumpy):
    system: Dict[str, Any]
    logging: LoggingConfig
    paths: PathsConfig
    enabled_outputs: OutputFlag
    processing: Processing
    paddle_config: PaddleOCRConfig
    modules: ModulesConfig
    cleanup: Cleanup