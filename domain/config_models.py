# services/config_models.py
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any

class ConfigWithNumpy(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True, arbitrary_types_allowed=False, strict=True)

class SystemPaths(ConfigWithNumpy):
    output_paths: List[str]
    libs_path: str
    containers: str
    buffer_handler: str
    temp_path: List[str]
    
class SystemParams(ConfigWithNumpy):
    system_paths: SystemPaths
    payloads_size: int

class DeploySettings(ConfigWithNumpy):
    deploy_mode: bool
    test_config: bool
    clean_mode: bool
    postgre_local: bool
    handle_memory: bool
    update_model: bool
    test_wf_model: bool

class PipelineConfig(ConfigWithNumpy):
    image_preparation_stager: Optional[List[str]] = None
    preprocessing_stager: Optional[List[str]] = None
    ocr_stager: Optional[List[str]] = None
    vectorization_stager: Optional[List[str]] = None
    db_stage: Optional[List[str]] = None

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

class ImgLoadOutputs(ConfigWithNumpy):
    full_img: bool
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
    training_data: bool
    features: bool
    image_features: bool
    table_structured: bool
    math_max_corrected: bool
    normalized_table: bool
    stack: bool

class OutputFlags(ConfigWithNumpy):
    image_load_outputs: ImgLoadOutputs
    preprocessing_outputs: PreprocessingOutputs
    ocr_outputs: OCROutputs
    vectorization_outputs: VectorizingOutputs

class PaddleConfig(ConfigWithNumpy):
    use_angle_cls: bool
    lang: str
    show_log: bool
    use_gpu: bool
    enable_mkldnn: bool
    cpu_threads: int
    max_batch_size: int
    table: bool
    det_limit_side_len: int
    det_db_score_mode: str
    use_mp: bool
    max_text_length: int
    return_word_box: bool
    rec_batch_num: int

class WordFinderConfig(ConfigWithNumpy):
    char_ngrams: List[int]
    threshold_similarity: float
    global_filter_threshold: float
    window_flexibility: int
    pkl_path: str
    matrix_path: str
    kf_path: str
    kf_idx: str
    ngrams_name: str
    matrix_name: str
    index_dict: str

class ModelsPaths(ConfigWithNumpy):
    models_dir: str
    paddle_path: str
    word_finder_path: str
    det_model: str
    rec_model: str
    
class ModelsConfig(ConfigWithNumpy):
    wf_config: WordFinderConfig
    paddle_config: PaddleConfig
    models_paths: ModelsPaths

class InkConfig(ConfigWithNumpy):
    white: List[int]
    black: List[int]
    aspect_ratio_range: List[float]
    angle_threshold: float
    thr: float
    black_thr: float
    solid_thr: float
    shape_thr: float

class GeometryDetect(ConfigWithNumpy):
    morph_kernel: List[int]
    iterations: int

class DeskewConfig(ConfigWithNumpy):
    min_angle_for_correction: float
    canny_thresholds: List[int]
    hough_threshold: int
    hough_min_line_length_cap_px: int
    hough_max_line_gap_px: int
    hough_angle_filter_range_degrees: List[float]

class CuttingConfig(ConfigWithNumpy):
    cropping_padding: float
    
class ImagePreparation(ConfigWithNumpy):
    ink_enhancement: InkConfig
    angle_corrector: DeskewConfig
    geometry_detect: GeometryDetect
    polygon_extractor: CuttingConfig

class MoireConfig(ConfigWithNumpy):
    min_distance_from_center: int
    notch_radius: int
    percentile_threshold: int
    mean_factor: int
    abs_threshold: int

class SharpeningConfig(ConfigWithNumpy):
    sharpness_threshold: float
    kernel: int

class SaltPepper(ConfigWithNumpy):
    kernel_size: int
    salt_pepper_threshold: float
    salt_pepper_low: int
    salt_pepper_high: int
    sobel_threshold: float

class GaussianConfig(ConfigWithNumpy):
    laplacian_variance_threshold: float

class ContrastConfig(ConfigWithNumpy):
    clahe_clip_limit: float
    contrast_threshold: int
    dimension_thresholds_px: List[int]
    grid_sizes_map: List[List[int]]

class PreprocessingConfig(ConfigWithNumpy):
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
    text_refine: TextRefiner

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

class DataCollector(ConfigWithNumpy):
    placeholder: str

class VectorConfig(ConfigWithNumpy):
    lineal: Lineal
    cos_sim: CosineSimilarity
    table_structurer: TableStructurer
    collector: DataCollector
    
class ModulesConfig(ConfigWithNumpy):
    image_preparation: ImagePreparation
    preprocessing: PreprocessingConfig
    ocr: OCRConfig
    vectorization: VectorConfig

class InputPaths(ConfigWithNumpy):
    input_dirs: List[str]
    images_names: List[str]
    skip_names: List[str]

class PayloadRequest(ConfigWithNumpy):
    payload_cols: List[str]
    
class UserRequests(ConfigWithNumpy):
    input_paths: InputPaths
    payload_request: PayloadRequest

class SystemSetUp(ConfigWithNumpy):
    system_params: SystemParams
    deploy_settings: DeploySettings
    pipeline_secuence: PipelineConfig
    log_debug: DebugOutputs
    enabled_outputs: OutputFlags
    env_config: Dict[str, Any]

class ConfigParams(ConfigWithNumpy):
    models_config: ModelsConfig
    modules: ModulesConfig
    user_requests: UserRequests

class MasterConfig(ConfigWithNumpy):
    system_params: SystemParams
    deploy_settings: DeploySettings
    pipeline_secuence: PipelineConfig
    log_debug: DebugOutputs
    enabled_outputs: OutputFlags
    models_config: ModelsConfig
    modules: ModulesConfig
    env_config: Dict[str, Any]
    user_requests: UserRequests