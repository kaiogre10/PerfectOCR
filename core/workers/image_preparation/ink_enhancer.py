# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_cc_metrics
from services.output_service import save_croped_image
from core.utils.image_utils import normalice_image

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """
    Worker robusto para restauración de tinta y eliminación de ruido adaptativo.
    Utiliza análisis estadístico (histograma) para diferenciar ruido de puntuación
    y morfología matemática segura (Closing) para reparar trazos rotos.
    """

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('ink_enhancement', {})
        
        # Configuración base (se usa como fallback o para la ventana de análisis)
        self.kernel_threshold: int = self.worker_config.get("kernel_threshold", 2) # Tamaño del padding para análisis de vecinos
        self.output = config.get("bin_full_img", False)
        
        # Configuración para reparación morfológica (segura)
        self.morph_kernel_size = (2, 2) # Kernel conservador para Closing

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Flujo principal de corrección de tinta."""
        try:
            start_time = time.perf_counter()
            logger.info("Iniciando InkCorrector (Modo Robusto/Adaptativo)...")
            
            image_name = manager.workflow.metadata.image_name if manager.workflow else "unknown"
            worker_name = context.get("worker_name") or "ink_corrector"
            output_paths = context.get("output_paths", [])

            # 1. Obtener imagen original
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error("No se encontró full_img en el DataFormatter")
                return False
                
            # 2. Pre-procesamiento: Decoloración y Binarización para análisis
            # Convertimos a gris limpiando colores de fondo/resaltadores
            gray_img = self._decolorate(full_img)
            
            # Obtenemos métricas de componentes conectados (CC) para el análisis estadístico
            # extract_cc_metrics devuelve el diccionario de contornos y la imagen binaria (Text=255, BG=0)
            metrics, full_bin_img = extract_cc_metrics(gray_img.copy(), worker_config={}, binarice=True)
            
            # 3. FASE 1: Limpieza Estadística (Sustractiva)
            # Analiza el histograma y elimina ruido sin tocar la estructura
            cleaned_img = self._adaptive_noise_removal(gray_img.copy(), full_bin_img, metrics)
            
            # 4. FASE 2: Reparación Estructural (Aditiva/Constructiva)
            # Aplica Morphological Closing para unir tinta rota
            final_img = self._apply_morphological_repair(cleaned_img)

            # 5. Guardar resultado y generar salidas de depuración
            if not manager.update_full_img(corrected=True, full_img=final_img):
                logger.warning("No se pudo actualizar la imagen en el manager")

            if self.output:
                self._save_debug_images(image_name, worker_name, output_paths, gray_img, full_bin_img, cleaned_img, final_img)
                    
            elapsed = time.perf_counter() - start_time
            logger.info(f"InkCorrector completado para '{image_name}' en {elapsed:.4f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error crítico en InkCorrector: {e}", exc_info=True)
            return False

    def _adaptive_noise_removal(self, target_img: np.ndarray, bin_img: np.ndarray, metrics: Dict[str, Any]) -> np.ndarray:
        """
        Elimina el ruido basándose en la distribución estadística de las áreas de los blobs.
        Determina automáticamente si la imagen está 'sucia' o 'limpia'.
        """
        img_h, img_w = bin_img.shape
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics.get("cont_array_dict", {})
        
        # A. Recolectar datos estadísticos
        all_areas = [c["cont_area"] for c in cont_array_dict.values()]
        
        if not all_areas:
            logger.warning("No se detectaron blobs para analizar.")
            return target_img

        # B. Calcular Histograma (Auto-calibración)
        # Usamos 'fd' (Freedman-Diaconis) para una elección robusta del ancho del bin
        hist_counts, bin_edges = np.histogram(all_areas, bins='auto')
        
        total_blobs = len(all_areas)
        blobs_in_first_bin = hist_counts[0]
        first_bin_ratio = blobs_in_first_bin / total_blobs
        
        # C. El "Test de Pánico": Decidir Umbral Dinámico
        # Si > 30% de los blobs están en el primer bin, asumimos imagen sucia.
        PANIC_THRESHOLD_RATIO = 0.30
        
        if first_bin_ratio > PANIC_THRESHOLD_RATIO:
            # Escenario SUCIO: Umbral agresivo (Límite superior del primer bin)
            dynamic_threshold = bin_edges[1]
            logger.info(f"DISTRIBUCIÓN SUCIA DETECTADA (Ratio: {first_bin_ratio:.2%}). Umbral dinámico fijado en: {dynamic_threshold:.2f} px")
        else:
            # Escenario LIMPIO: Umbral mínimo de seguridad (bypass)
            dynamic_threshold = 5.0 # Solo borrar cosas microscópicas
            logger.info(f"DISTRIBUCIÓN LIMPIA DETECTADA (Ratio: {first_bin_ratio:.2%}). Modo Bypass activo.")

        # D. Ejecución Selectiva (Divide y Vencerás)
        blobs_removed = 0
        
        for countours in cont_array_dict.values():
            area = countours["cont_area"]
            
            # REGLA 1: Inmunidad Diplomática
            if area >= dynamic_threshold:
                continue # Es texto grande, ignorar.
            
            # REGLA 2: Juicio a los Sospechosos (Test de Aislamiento)
            # Solo analizamos vecinos si el blob es menor al umbral dinámico
            bbox = countours["cont_bbox"]
            coords = countours["cont_coords"]
            x, y, w, h = bbox
            
            # Definir ventana con padding (kernel_threshold)
            win_x1 = max(0, x - self.kernel_threshold)
            win_y1 = max(0, y - self.kernel_threshold)
            win_x2 = min(img_w, x + w + self.kernel_threshold)
            win_y2 = min(img_h, y + h + self.kernel_threshold)
            
            window = bin_img[win_y1:win_y2, win_x1:win_x2]
            
            # Optimización: Si la ventana completa es negra (0), es ruido aislado instantáneo.
            # (Asumiendo bin_img: Texto=255, Fondo=0)
            if np.sum(window) == 0:
                # Caso imposible si el blob existe, pero por seguridad
                continue

            # Verificar bordes de la ventana
            border_mask = np.zeros_like(window, dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            
            border_pixels = window[border_mask]
            
            # Si todos los píxeles del borde son 0 (Fondo), está aislado -> ELIMINAR
            if np.all(border_pixels == 0):
                # Pintamos de blanco (255) en la imagen destino (asumiendo fondo blanco)
                cv2.drawContours(target_img, [coords], -1, color=(255, 255, 255), thickness=cv2.FILLED)
                blobs_removed += 1

        logger.info(f"Limpieza completada. Blobs eliminados: {blobs_removed} de {total_blobs}")
        return target_img

    def _apply_morphological_repair(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica Cierre Morfológico (Closing) para reparar tinta rota.
        Seguro de usar porque el ruido ya fue eliminado.
        """
        # Kernel rectangular pequeño (2x2) para no fusionar líneas de texto
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        
        # MORPH_CLOSE = Dilatación seguida de Erosión
        # Une grietas internas sin engrosar excesivamente el contorno exterior
        repaired = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        logger.debug(f"Reparación morfológica aplicada con kernel {self.morph_kernel_size}")
        return repaired

    def _decolorate(self, full_img: np.ndarray) -> np.ndarray:
        """
        Elimina colores (rayones, resaltados) dejando solo blanco y negro.
        Mantiene la lógica original del usuario.
        """
        # Umbrales para detectar "no blanco" y "no negro"
        threshold_black = 160 
        threshold_white = 180 

        # Máscaras
        mask_black = np.all(full_img <= threshold_black, axis=2)
        mask_white = np.all(full_img >= threshold_white, axis=2)
        mask_valid = mask_black | mask_white
        
        # Limpiar píxeles de color -> Blanco
        clean_img = full_img.copy()
        clean_img[~mask_valid] = [255, 255, 255]

        # Normalizar a escala de grises
        gray = normalice_image(clean_img)
        
        if gray is not None:
            return gray
        else:
            return cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    def _save_debug_images(self, base_name: str, worker: str, paths: List[str], 
                          original: np.ndarray, binary: np.ndarray, 
                          cleaned: np.ndarray, final: np.ndarray):
        """Guarda pasos intermedios para depuración."""
        save_croped_image(base_name, f"01_gray_input_{worker}", original, paths, worker)
        save_croped_image(base_name, f"02_binary_analysis_{worker}", binary, paths, worker)
        save_croped_image(base_name, f"03_cleaned_noise_{worker}", cleaned, paths, worker)
        save_croped_image(base_name, f"04_repaired_final_{worker}", final, paths, worker)
