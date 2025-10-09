# services/config_models.py
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List, Tuple, Optional

class ConfigWithNumpy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OutputFlag(ConfigWithNumpy):
    preprocessed_image: bool
    ocr_raw: bool
    cropped_img: bool
    moire_poly: bool
    sp_poly: bool
    gauss_poly: bool
    ink_poly: bool
    clahe_poly: bool
    sharp_poly: bool
    binarized_polygons: bool
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

class ModelsPaths(ConfigWithNumpy):
    det_model_dir: str
    rec_model_dir: str
    cls_model_dir: str
    wordfinder_model_path: str

class ModelsConfig(ConfigWithNumpy):
    use_angle_cls: bool
    lang: str
    show_log: bool
    use_gpu: bool
    enable_mkldnn: bool
    models: ModelsPaths

class PathsConfig(ConfigWithNumpy):
    input_folder: str
    output_folder: str

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

class MoireConfig(ConfigWithNumpy):
    percentile: MoirePercentile
    factor: MoireFactor
    absolute: MoireAbsolute
    
class BinarizeQuality(ConfigWithNumpy):
    quality_min: float
    quality_max: float

class Binarization(ConfigWithNumpy):
    c_value: int
    height_thresholds_px: List[int]
    block_sizes_map: List[int]
    quality: BinarizeQuality

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

class CleaningConfig(ConfigWithNumpy):
    std_low: float
    sp_thr: float
    clahe_clip: float
    clahe_grid: Tuple[int, int]

class ImagePreparation(ConfigWithNumpy):
    angle_corrector: DeskewConfig
    cleaner: CleaningConfig
    polygon_extractor: CuttingConfig

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

class InkConfig(ConfigWithNumpy):
    faded_detection_threshold: float
    contrast_boost_factor: float

class PreprocessingConfig(ConfigWithNumpy):
    moire: MoireConfig  
    sp_config: SaltPepper 
    gauss_params: GaussianConfig
    ink_enhancement: InkConfig
    contrast: ContrastConfig  
    sharpening: SharpeningConfig
    binarizator: Binarization

class TextualConfig(ConfigWithNumpy):
    min_probability: float
    min_char: int

class SemanticClasificator(ConfigWithNumpy):
    numeric: List[float]
    code: List[float]
    descriptive: List[float]

class Fragmenter(ConfigWithNumpy):
    fragment_on_text: bool
    min_contours_for_frag: int
        
class DataFinder(ConfigWithNumpy):
    wordfinder_model_path: str
    min_similarity: float
    max_q_lenght: int
    
class TextRefiner(ConfigWithNumpy):
    num_passes: int

class TextCorrector(ConfigWithNumpy):
    confidence_threshold: float

class OCRConfig(ConfigWithNumpy):
    min_confidence: float
    text_refiner: TextRefiner
    text_cleaner: TextualConfig
    semantic_clasificator: SemanticClasificator
    data_finder: DataFinder
    fragmenter: Fragmenter
    text_corrector: TextCorrector

class DBSCAN(ConfigWithNumpy):
    eps: float
    min_cluster_size: int

class Lineal(ConfigWithNumpy):
    overlap_threshold: float

class TableStructurer(ConfigWithNumpy):
    min_h: int
    min_cosine_similarity: float

class CosineSimilarity(ConfigWithNumpy):
    min_cluster: int
    similarity_threshold: float
    interval: int
    
class Vectorizer(ConfigWithNumpy):
    keywords_interval_enabled: bool

class VectorConfig(ConfigWithNumpy):
    lineal: Lineal
    dbscan: DBSCAN
    vectorizer: Vectorizer
    cos_sim: CosineSimilarity
    math_max: MathMaxConfig
    table_structurer: TableStructurer
    
class ModulesConfig(ConfigWithNumpy):
    image_preparation: ImagePreparation
    preprocessing: PreprocessingConfig
    ocr: OCRConfig
    vectorization: VectorConfig

class PipelineConfig(ConfigWithNumpy):
    imagepre_stage: Optional[List[str]]
    preprocessing_stage: Optional[List[str]]
    ocr_stage: Optional[List[str]]
    vector_stage: Optional[List[str]]

class MasterConfig(ConfigWithNumpy):
    system: Dict[str, Any]
    paths: PathsConfig
    pipeline_secuence: PipelineConfig
    enabled_outputs: OutputFlag
    processing: Processing
    models_config: ModelsConfig
    modules: ModulesConfig
