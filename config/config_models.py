# services/config_models.py
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple, Optional, Dict, Any

class ConfigWithNumpy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ImgLoadOutputs(ConfigWithNumpy):
    full_img: bool
    deleted_polys: bool
    bin_full_img: bool
    angle_corrected: bool
    cropped_img: bool
    final_polys: bool
    discarded_polys: bool
    opened: bool

class PreprocessingOutputs(ConfigWithNumpy):
    sp_poly: bool
    gauss_poly: bool
    clahe_poly: bool
    sharp_poly: bool

class OCROutputs(ConfigWithNumpy):
    ocr_raw: bool
    fragmented_polys: bool
    semantic_field: bool
    cleanned_text: bool

class VectorizingOutputs(ConfigWithNumpy):
    reconstructed_lines: bool
    table_lines: bool
    features: bool
    image_features: bool
    table_structured: bool
    math_max_corrected: bool
    stack: bool

class OutputFlags(ConfigWithNumpy):
    image_load_outputs: ImgLoadOutputs
    preprocessing_outputs: PreprocessingOutputs
    ocr_outputs: OCROutputs
    vectorization_outputs: VectorizingOutputs
    
class ModelsConfig(ConfigWithNumpy):
    use_angle_cls: bool
    lang: str
    show_log: bool
    use_gpu: bool
    enable_mkldnn: bool
    cpu_threads: int
    max_batch_size: int
    det_limit_side_len: int
    rec_batch_num: int
    det_model_dir: List[str]
    rec_model_dir: List[str]
    wf_model_path: List[str]
    set_wf_params: bool
    det_db_score_mode: str
    use_mp: bool
    max_text_length: int
    return_word_box: bool
    
class InkConfig(ConfigWithNumpy):
    white: List[int]
    black: List[int]
    aspect_ratio_range: Tuple[float, float]
    angle_threshold: float
    thr: float
    black_thr: float
    solid_thr: float
    shape_thr: float

class GeometryDetect(ConfigWithNumpy):
    morph_kernel: Tuple[int, int]
    iterations: int

class SharpeningConfig(ConfigWithNumpy):
    sharpness_threshold: float
    kernel: int

class MoireConfig(ConfigWithNumpy):
    min_distance_from_center: int
    notch_radius: int
    percentile_threshold: int
    mean_factor: int
    abs_threshold: int
    
class SaltPepper(ConfigWithNumpy):
    kernel_size: int
    salt_pepper_threshold: float
    salt_pepper_low: int
    salt_pepper_high: int
    sobel_threshold: float

class GaussianConfig(ConfigWithNumpy):
    laplacian_variance_threshold: float

class DeskewConfig(ConfigWithNumpy):
    min_angle_for_correction: float
    canny_thresholds: Tuple[int, int]
    hough_threshold: int
    hough_min_line_length_cap_px: int
    hough_max_line_gap_px: int
    hough_angle_filter_range_degrees: Tuple[float, float]

class CuttingConfig(ConfigWithNumpy):
    cropping_padding: float
    
class ImagePreparation(ConfigWithNumpy):
    ink_enhancement: InkConfig
    angle_corrector: DeskewConfig
    geometry_detect: GeometryDetect
    polygon_extractor: CuttingConfig

class ContrastConfig(ConfigWithNumpy):
    clahe_clip_limit: float
    contrast_threshold: int
    dimension_thresholds_px: List[int]
    grid_sizes_map: List[Tuple[int, int]]
        
class MathMaxConfig(ConfigWithNumpy):
    row_relative_tolerance: str
    cols_name: List[str]

class RestoreConfig(ConfigWithNumpy):
    area_threshold: int
    kernel_threshold: int

class PreprocessingConfig(ConfigWithNumpy):
    restorer: RestoreConfig
    moire: MoireConfig
    sp_config: SaltPepper 
    gauss_params: GaussianConfig
    contrast: ContrastConfig
    sharpening: SharpeningConfig

class TextRefiner(ConfigWithNumpy):
    num_passes: int

class PaddleTranscription(ConfigWithNumpy):
    min_confidence: float

class OCRConfig(ConfigWithNumpy):
    paddle_wrapper: PaddleTranscription
    text_refiner: TextRefiner

class Lineal(ConfigWithNumpy):
    get_vectors: bool
    overlap_threshold: float

class TableStructurer(ConfigWithNumpy):
    min_h: int

class CosineSimilarity(ConfigWithNumpy):
    eps: float
    metric: str
    min_cluster: int
    tolerance_sim: int
    similarity_threshold: float
    emergency_threshold: float
    min_internal_sim: float

class VectorConfig(ConfigWithNumpy):
    lineal: Lineal
    cos_sim: CosineSimilarity
    math_max: MathMaxConfig
    table_structurer: TableStructurer
    
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
    
class DebugOutputs(ConfigWithNumpy):
    all_logs: bool
    text_ocr: bool
    text_clean: bool
    text_del: bool
    text_correct: bool
    frag_polys: bool
    refined_text: bool
    seman_clas: bool
    key_fields: bool
    lines: bool
    table_lines: bool
    table_geo: bool
    table_correct: bool    
    kf_list_log: List[int]
    semantic_types_log: List[int]
    time_stages_log: bool
    time_worker_log: bool

class SystemConfig(ConfigWithNumpy):
    input_dirs: List[str]
    images_names: List[str]
    valid_img_ext: List[str]
    output_paths: List[str]
    storage_bin: List[str]
    trash_ext: List[str]
    invalid_extensions: List[str]

class ExportingConfig(ConfigWithNumpy):
    destination_services: Optional[List[str]] = None

class TestingModes(ConfigWithNumpy):
    deploy_mode: bool
    test_config: bool

class MasterConfig(ConfigWithNumpy):
    system_config: SystemConfig
    pipeline_secuence: PipelineConfig
    enabled_outputs: OutputFlags
    models_config: ModelsConfig
    modules: ModulesConfig
    log_debug: DebugOutputs
    exporting_config: ExportingConfig
    test_modes: TestingModes
    env_config: Dict[str, Any]
