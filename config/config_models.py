# services/config_models.py
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple, Optional

class ConfigWithNumpy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ImgLoadOutputs(ConfigWithNumpy):
    deleted_polys: bool
    pre_clean: bool
    angle_corrected: bool
    cropped_img: bool
    filtered_polys: bool
    discarded_polys: bool

class PreprocessingOutputs(ConfigWithNumpy):
    moire_poly: bool
    sp_poly: bool
    gauss_poly: bool
    ink_poly: bool
    clahe_poly: bool
    sharp_poly: bool
    binarized_polygons: bool

class OCROutputs(ConfigWithNumpy):
    ocr_raw: bool
    fragmented_polys: bool
    reconstructed_lines: bool
    semantic_field: bool

class VectorizingOutputs(ConfigWithNumpy):
    table_lines: bool
    encoded_lines: bool
    features: bool
    image_features: bool
    table_structured: bool
    math_max_corrected: bool

class OutputFlags(ConfigWithNumpy):
    image_load_outputs: ImgLoadOutputs
    preprocessing_outputs: PreprocessingOutputs
    ocr_outputs: OCROutputs
    vectorization_outputs: VectorizingOutputs
    
class Processing(ConfigWithNumpy):
    small_batch_limit: int
    valid_image_extensions: Tuple[str, ...]

class ModelsConfig(ConfigWithNumpy):
    use_angle_cls: bool
    lang: str
    show_log: bool
    use_gpu: bool
    enable_mkldnn: bool
    rec_batch_num: int
    det_model_dir: str
    rec_model_dir: str
    wf_model_path: str
    
class SharpeningConfig(ConfigWithNumpy):
    sharpness_threshold: float
    kernel: int

class MoireConfig(ConfigWithNumpy):
    min_distance_from_center: int
    notch_radius: int
    percentile_threshold: int
    mean_factor_threshold: int
    absolute_threshold: int
    
class Fragmenter(ConfigWithNumpy):
    min_contours_for_frag: int
    c_value: int
    height_thresholds_px: List[int]
    block_sizes_map: List[int]
    min_area_factor: float
    k_sigma: float
    min_cc_for_frag: int
    min_gap_outlier: float
    density_threshold: float
    max_cc_for_density_rule: int
    width_var_threshold: float
    
class SaltPepper(ConfigWithNumpy):
    kernel_size: int
    salt_pepper_threshold: float
    # salt_pepper_low: int
    salt_pepper_high: int
    sobel_threshold: float

class GaussianConfig(ConfigWithNumpy):
    laplacian_variance_threshold: float

class DeskewConfig(ConfigWithNumpy):
    min_angle_for_correction: float
    canny_thresholds: List[int]
    hough_threshold: int
    hough_min_line_length_cap_px: int
    hough_max_line_gap_px: int
    hough_angle_filter_range_degrees: List[int]

class GeoDetector(ConfigWithNumpy):
    angle_thr: Tuple[float, float]

class CuttingConfig(ConfigWithNumpy):
    cropping_padding: int
    bin_interval: Tuple[int, int]
    percentil : float

class CleaningConfig(ConfigWithNumpy):
    std_low: float
    sp_thr: float
    clahe_clip: float
    dimension_thresholds_px: List[int]
    clahe_grid: List[Tuple[int, int]]
    kernel_size: int

class ImagePreparation(ConfigWithNumpy):
    angle_corrector: DeskewConfig
    cleaner: CleaningConfig
    geometry_detector: GeoDetector
    polygon_extractor: CuttingConfig

class ContrastConfig(ConfigWithNumpy):
    clahe_clip_limit: float
    contrast_threshold: int
    dimension_thresholds_px: List[int]
    grid_sizes_map: List[Tuple[int, int]]
    window_size: int
    std_dev_threshold: float
    
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

class TextualCleanConfig(ConfigWithNumpy):
    min_probability: float
    min_char: float
    min_confidence: float
    min_conf_to_clean: float

class SemanticClasificator(ConfigWithNumpy):
    semantic_range: Tuple[float, float]
    encode_mean: Tuple[float, float]
    morph_mean: Tuple[float, float]
        
class DataFinder(ConfigWithNumpy):
    min_similarity: float
    max_q_lenght: Tuple[int, int]
    
class TextRefiner(ConfigWithNumpy):
    num_passes: int

class TextCorrector(ConfigWithNumpy):
    confidence_threshold: float

class PaddleTranscription(ConfigWithNumpy):
    min_confidence: float

class OCRConfig(ConfigWithNumpy):
    paddle_wrapper: PaddleTranscription
    text_refiner: TextRefiner
    text_cleaner: TextualCleanConfig
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

class CosineSimilarity(ConfigWithNumpy):
    min_cluster: int
    similarity_threshold: float
    emergency_threshold: float
    interval: int
    dummie_weights: Tuple[float, float]

class Vectorizer(ConfigWithNumpy):
    keywords_interval_enabled: bool
    exclude_types: List[str]

class VectorConfig(ConfigWithNumpy):
    lineal: Lineal
    dbscan: DBSCAN
    vectorizer: Vectorizer
    cos_sim: CosineSimilarity
    math_max: MathMaxConfig
    table_structurer: TableStructurer

class UtilsConfig(ConfigWithNumpy):
    dpi_range: List[int]
    
class ModulesConfig(ConfigWithNumpy):
    image_preparation: ImagePreparation
    preprocessing: PreprocessingConfig
    ocr: OCRConfig
    vectorization: VectorConfig

class PipelineConfig(ConfigWithNumpy):
    imagepre_stage: Optional[List[str]] = None
    preprocessing_stage: Optional[List[str]] = None
    ocr_stage: Optional[List[str]] = None
    vector_stage: Optional[List[str]] = None

class MasterConfig(ConfigWithNumpy):
    pipeline_secuence: PipelineConfig
    enabled_outputs: OutputFlags
    processing: Processing
    models_config: ModelsConfig
    modules: ModulesConfig
    utils: UtilsConfig