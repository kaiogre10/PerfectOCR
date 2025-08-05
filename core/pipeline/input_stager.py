# PerfectOCR/core/coordinators/geometric_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor
from core.workers.image_preparation.image_loader import ImageLoader
from core.workers.image_preparation.cleanner import ImageCleaner
from core.workers.image_preparation.angle_corrector import AngleCorrector
from core.workers.image_preparation.geometry_detector import GeometryDetector
from core.workers.image_preparation.lineal_reconstructor import LineReconstructor
from core.workers.image_preparation.poly_gone import PolygonExtractor
from core.domain.workflow_job import WorkflowJob, ProcessingStage, DocumentMetadata, ImageDimensions, BoundingBox, PolygonGeometry, Polygon, LineInfo

logger = logging.getLogger(__name__)

class InputManager:
    """
    Coordina la fase de extracción de polígonos, delegando todo el trabajo
    a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, stage_config: Dict, input_path: Dict, project_root: str):
        self.project_root = project_root
        self.image_loader = config
        self.manager_config = stage_config
        self.input_path = input_path
        paddle_config = self.image_loader.get('paddle_config', {})
        self._image_load = ImageLoader(config=self.image_loader.get('extensions', {}), input_path = self.input_path, project_root=self.project_root)
        self._cleaner = ImageCleaner(config=self.image_loader.get('cleaning', {}), project_root=self.project_root)
        self._angle_corrector = AngleCorrector(config=self.image_loader.get('deskew', {}), project_root=self.project_root)
        self._geometry_detector = GeometryDetector(paddle_config=paddle_config, project_root=self.project_root)
        self._poly = PolygonExtractor(config=self.image_loader.get('cutting', {}), project_root=self.project_root)
        self._lineal = LineReconstructor(config={}, project_root=self.project_root)
        self._lines_geometry: Optional[Dict[str, Any]] = None
        self._binarization: Optional[Dict[str, Any]] = None
                
    def _generate_polygons(self) -> Tuple[Optional[WorkflowJob], float]:
        """
        Recibe WorkflowJob del ImageLoader y lo enriquece con polígonos y líneas.
        """
        pipeline_start = time.time()
        
        workflow_job, metadata = self._image_load._load_image_and_metadata(self.input_path)
        
        if workflow_job is None or workflow_job.doc_metadata is None or workflow_job.full_img is None:
            return None, 0.0

        metadata_dict = {
            "image_name": workflow_job.doc_metadata.doc_name,
            "formato": workflow_job.doc_metadata.formato,
            "img_dims": {
                "width": workflow_job.doc_metadata.img_dims.width,
                "height": workflow_job.doc_metadata.img_dims.height
            },
            "dpi": workflow_job.doc_metadata.dpi
        }
        
        clean_image, doc_data = self._cleaner._quick_enhance(workflow_job.full_img, metadata_dict)
        deskewed_img, new_dims = self._angle_corrector.correct(clean_image, doc_data.get("metadata", {}).get("img_dims", {}))
        
        if new_dims and (new_dims["width"] != workflow_job.doc_metadata.img_dims.width or new_dims["height"] != workflow_job.doc_metadata.img_dims.height):
            updated_dims = ImageDimensions(
                width=int(new_dims.get("width", workflow_job.doc_metadata.img_dims.width)),
                height=int(new_dims.get("height", workflow_job.doc_metadata.img_dims.height))
            )
            workflow_job.doc_metadata = DocumentMetadata(
                doc_name=workflow_job.doc_metadata.doc_name,
                img_dims=updated_dims,
                formato=workflow_job.doc_metadata.formato,
                dpi=workflow_job.doc_metadata.dpi,
                fecha_creacion=workflow_job.doc_metadata.fecha_creacion
            )
            workflow_job.full_img = deskewed_img
            logger.info(f"[InputManager] Dimensiones de imagen actualizadas a {updated_dims.width}x{updated_dims.height}")

        enriched_doc = self._geometry_detector.detect(deskewed_img.copy(), doc_data)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_lineal = executor.submit(self._lineal._reconstruct_lines, enriched_doc)
            future_poly = executor.submit(self._poly._extract_individual_polygons, deskewed_img, enriched_doc)
            
            lineal_result = future_lineal.result()
            extracted_polygons_dict = future_poly.result()
        
        total_lines_dict = self._lineal._get_lines_geometry()
        
        for poly_id, poly_data in extracted_polygons_dict.items():
            if poly_id in lineal_result:
                poly_data["line_id"] = lineal_result[poly_id]

        for poly_id, poly_data in extracted_polygons_dict.items():
            try:
                geometry_data = poly_data.get("geometry", {})
                bbox_list = geometry_data.get("bounding_box", [0, 0, 0, 0])
                bbox = BoundingBox(
                    x_min=float(bbox_list[0]), y_min=float(bbox_list[1]),
                    x_max=float(bbox_list[2]), y_max=float(bbox_list[3])
                )
                coords_list = geometry_data.get("polygon_coords", [])
                coords_tuples = [(float(p[0]), float(p[1])) for p in coords_list]
                centroid_list = geometry_data.get("centroid", [0, 0])
                
                geometry = PolygonGeometry(
                    polygon_coords=coords_tuples,
                    bounding_box=bbox,
                    centroid=(float(centroid_list[0]), float(centroid_list[1]))
                )
                
                polygon = Polygon(
                    polygon_id=poly_data.get("polygon_id", poly_id),
                    geometry=geometry,
                    line_id=poly_data.get("line_id", "line_0000"),
                    cropped_img=poly_data.get("cropped_img"),
                    padding_coords=poly_data.get("padding_coords")
                )
                workflow_job.add_polygon(polygon)
            except Exception as e:
                workflow_job.add_error(f"Error convirtiendo polígono {poly_id}: {e}")
                continue
        
        for line_id, line_data in total_lines_dict.items():
            try:
                bbox_list = line_data.get("bounding_box", [0, 0, 0, 0])
                bbox = BoundingBox(
                    x_min=float(bbox_list[0]), y_min=float(bbox_list[1]),
                    x_max=float(bbox_list[2]), y_max=float(bbox_list[3])
                )
                centroid_list = line_data.get("centroid", [0, 0])
                line_info = LineInfo(
                    line_id=line_id,
                    bounding_box=bbox,
                    centroid=(float(centroid_list[0]), float(centroid_list[1])),
                    polygon_ids=line_data.get("polygon_ids", [])
                )
                workflow_job.add_line(line_info)
            except Exception as e:
                workflow_job.add_error(f"Error convirtiendo línea {line_id}: {e}")
                continue
        
        workflow_job.update_stage(ProcessingStage.POLYGONS_EXTRACTED)
        total_time = time.time() - pipeline_start
        workflow_job.processing_times["polygon_generation"] = total_time
        
        logger.info(f"[InputManager] Generación de polígonos completada en {total_time:.3f}s")
        return workflow_job, total_time

    def _get_polygons_to_binarize(self) -> Dict[str, Any]:
        """Expone los IDs problemáticos para que el builder pueda usarlos."""
        polygons_to_binarize = self._poly._get_polygons_copy()
        polygons_to_bin = polygons_to_binarize
        return polygons_to_bin
