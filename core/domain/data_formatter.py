# core/domain/workflow_manager.py
from core.domain.data_models import WorkflowDict, Metadata, ImageData, WORKFLOW_SCHEMA, CroppedImage, Polygons, Geometry, CroppedGeometry
from dataclasses import asdict
import numpy as np
import jsonschema
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """

    def __init__(self):
        self.workflow: Optional[WorkflowDict] = None
        self.schema = WORKFLOW_SCHEMA

    def create_dict(self, dict_id: str, full_img: np.ndarray[Any, Any], metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo dict con validación automática"""
        try:
            meta = Metadata(
                image_name=str(metadata.get("image_name", "")),
                format=str(metadata.get("format", "")),
                img_dims={
                    "width": int(metadata.get("img_dims", {}).get("width", 2)),
                    "height": int(metadata.get("img_dims", {}).get("height", 2))
                },
                dpi={
                    "x": float(metadata.get("dpi", 0.0)),
                    "y": None
                } if metadata.get("dpi") is not None else None,
                date_creation=metadata.get("date_creation", datetime.now().isoformat()),
                color=str(metadata.get("color", "")) if metadata.get("color") is not None else None
            )
            full_img = full_img.tolist() if isinstance(full_img, list) else full_img
            if full_img.tolist() is None:
                logger.info("Full_img está vacia")
                return full_img.tolist()
            self.workflow = WorkflowDict(
                dict_id=dict_id,
                full_img=full_img.tolist(),
                metadata=meta,
                image_data=ImageData(polygons={})
            )
            jsonschema.validate(asdict(self.workflow), self.schema)
            return True
        except Exception as e:
            logger.error(f"Error creando dict: {e}")
            return False

    def create_polygon_dicts(self, results: Optional[List[Any]]) -> bool:
        """
        Procesa los resultados de PaddleOCR y los guarda como objetos Polygons en el workflow.
        """
        polygons: Dict[str, Polygons] = {}
        try:
            if not results or not results[0]:
                logger.warning("No se encontraron polígonos en los resultados de PaddleOCR.")
                return False

            for idx, poly_pts in enumerate(results[0]):
                xs = [float(p[0]) for p in poly_pts]
                ys = [float(p[1]) for p in poly_pts]
                poly_id = f"poly_{idx:04d}"

                geometry = Geometry(
                    polygon_coords=[[xs[i], ys[i]] for i in range(len(xs))],
                    bounding_box=[min(xs), min(ys), max(xs), max(ys)],
                    centroid=[sum(xs) / len(xs), sum(ys) / len(ys)]
                )
                
                cropped_geometry = CroppedGeometry(
                    padding_bbox=[],
                    padd_centroid=[],
                    padding_coords=[]
                )

                polygons[poly_id] = Polygons(
                    polygon_id=poly_id,
                    geometry=geometry,
                    cropedd_geometry=cropped_geometry,
                    cropped_img=None,
                    perimeter=None,
                    line_id="",
                    ocr_text="",
                    ocr_confidence=None,
                    was_fragmented=False,
                    status=False,
                    stage=""
                )

            if self.workflow:
                self.workflow.image_data.polygons = polygons
                logger.info(f"Polígonos estructurados: {len(polygons)}")
                return True
            else:
                logger.error("No hay workflow inicializado.")
                return False
        except Exception as e:
            logger.error(f"Error creando diccionario de polígonos: {e}", exc_info=True)
            return False

    def get_dict_data(self) -> Dict[str, Any]:
        """Devuelve copia completa del dict"""
        return asdict(self.workflow) if self.workflow else {}
    
    def get_metadata(self) -> Optional[Metadata]:
        """Devuelve los metadatos del dict"""
        return self.workflow.metadata if self.workflow else None

    def get_polygons(self) -> Dict[str, Polygons]:
        return self.workflow.image_data.polygons if self.workflow else {}
        
    def get_workflow_schema(self) -> Dict[str, Any]:
        """Devuelve el esquema de workflow definido en los datamodels"""
        return self.schema    
    
    def get_polygons_with_cropped_img(self) -> Dict[str, Polygons]:
        """
        Devuelve el diccionario de polígonos con sus imágenes recortadas listas para el contexto de los workers.
        """
        if self.workflow is None:
            return {}
        return self.workflow.image_data.polygons

    def update_full_img(self, new_img: Optional[np.ndarray[Any, Any]] = None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow"""
        try:
            if self.workflow is None:
                logger.error("No hay workflow inicializado para actualizar full_img.")
                return False
                
            if new_img is None:
                # Si se pasa None, vaciamos la imagen para liberar memoria
                self.workflow.full_img = None
                logger.info("full_img liberada del workflow.")
            else:
                # Si se pasa una imagen, la actualizamos
                self.workflow.full_img = new_img
            return True
        except Exception as e:
            logger.error(f"Error actualizando full_img: {e}")
            return False
            
    def save_cropped_images(
        self,
        cropped_images: Dict[str, np.ndarray[Any, Any]],
        line_ids: Dict[str, str],
        cropped_geometries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Guarda imágenes recortadas, line_ids y geometría de recorte en los polígonos del workflow"""
        try:
            if self.workflow is None:
                logger.error("No hay workflow inicializado para guardar imágenes recortadas.")
                return False

            for poly_id, img in cropped_images.items():
                if poly_id in self.workflow.image_data.polygons:
                    polygon = self.workflow.image_data.polygons[poly_id]
                    if polygon:
                        polygon.cropped_img = img
                        if poly_id in cropped_geometries:
                            geom_data = cropped_geometries[poly_id]
                            polygon.cropedd_geometry = CroppedGeometry(
                                padding_bbox=geom_data.get('padding_bbox', []),
                                padd_centroid=geom_data.get('padd_centroid', []),
                                padding_coords=geom_data.get('padding_coords', [])
                            )

            for poly_id, line_id in line_ids.items():
                if poly_id in self.workflow.image_data.polygons:
                    polygon = self.workflow.image_data.polygons[poly_id]
                    if polygon:
                        polygon.line_id = line_id

            logger.info(f"Guardadas {len(cropped_images)} imágenes recortadas, {len(line_ids)} line_ids y geometría de recorte.")
            return True
        except Exception as e:
            logger.error(f"Error guardando imágenes recortadas y geometría: {e}")
            return False
            
    def get_cropped_images_for_preprocessing(self) -> Dict[str, CroppedImage]:
        """Excepción controlada: convierte dict interno a CroppedImage para preprocesamiento
    cropped_images = {
        "poly_0000": CroppedImage(
            cropped_img=np.ndarray,  # Imagen numpy del polígono
            polygon_id="poly_0000"   # ID del polígono
        ),
        "poly_0001": CroppedImage(
            cropped_img=np.ndarray,  # Imagen numpy del polígono
            polygon_id="poly_0001"   # ID del polígono
        ),
        """
        
        cropped_images = {}
        if not self.workflow or not self.workflow.image_data:
            return cropped_images
            
        for poly_id, poly_data in self.workflow.image_data.polygons.items():
            if poly_data.cropped_img is not None:
                cropped_images[poly_id] = CroppedImage(
                    cropped_img=poly_data.cropped_img,
                    polygon_id=poly_id  
                )
        return cropped_images
        
        
    # NUEVO método en DataFormatter  
    def update_preprocessing_result(self, poly_id: str, cropped_img: Optional[CroppedImage],
                                worker_name: str, success: bool):
        """Actualiza resultado de preprocesamiento y marca stage/status"""
        if self.workflow and poly_id in self.workflow.image_data.polygons:
            polygon = self.workflow.image_data.polygons[poly_id]
            if polygon and cropped_img:
                polygon.cropped_img = cropped_img.cropped_img
                polygon.stage = worker_name
                polygon.status = success
