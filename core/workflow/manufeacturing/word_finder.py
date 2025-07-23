import json
import cv2
import shapely
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

class WordSeparator:
    def __init__(self):
        self.debug_mode = True
        
    def load_ocr_results(self, file_path: str) -> Dict[str, Any]:
        """Cargar resultados OCR desde archivo JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_polygon_bbox(self, polygon_coords: List[List[float]]) -> Tuple[int, int, int, int]:
        """Obtener bounding box de un polígono"""
        points = np.array(polygon_coords)
        x_min = int(np.min(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        x_max = int(np.max(points[:, 0]))
        y_max = int(np.max(points[:, 1]))
        return x_min, y_min, x_max, y_max
    
    def create_text_mask(self, image_shape: Tuple[int, int], polygon_coords: List[List[float]]) -> np.ndarray:
        """Crear máscara del polígono de texto"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        points = np.array(polygon_coords, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask
    
    def analyze_horizontal_spacing(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analizar espaciado horizontal usando histograma"""
        x_min, y_min, x_max, y_max = bbox
        
        # Extraer la región de interés
        roi_mask = mask[y_min:y_max, x_min:x_max]
        
        # Proyección horizontal (suma vertical)
        horizontal_projection = np.sum(roi_mask, axis=0)
        
        # Normalizar
        if np.max(horizontal_projection) > 0:
            horizontal_projection = horizontal_projection / np.max(horizontal_projection)
        
        # Detectar valles (espacios) y picos (texto)
        valleys = []
        peaks = []
        
        threshold = 0.1  # Umbral para considerar un valle
        
        for i in range(1, len(horizontal_projection) - 1):
            # Valle: valor bajo rodeado de valores altos
            if (horizontal_projection[i] < threshold and 
                horizontal_projection[i-1] > threshold and 
                horizontal_projection[i+1] > threshold):
                valleys.append(i)
            
            # Pico: valor alto rodeado de valores bajos
            if (horizontal_projection[i] > 0.7 and 
                horizontal_projection[i-1] < 0.5 and 
                horizontal_projection[i+1] < 0.5):
                peaks.append(i)
        
        # Calcular métricas de espaciado
        valley_widths = []
        if len(valleys) > 1:
            for i in range(1, len(valleys)):
                valley_widths.append(valleys[i] - valleys[i-1])
        
        return {
            "horizontal_projection": horizontal_projection,
            "valleys": valleys,
            "peaks": peaks,
            "valley_widths": valley_widths,
            "avg_valley_width": np.mean(valley_widths) if valley_widths else 0,
            "total_width": x_max - x_min,
            "roi_shape": roi_mask.shape
        }
    
    def detect_missing_spaces(self, text: str, spacing_info: Dict[str, Any]) -> List[int]:
        """Detectar posiciones donde faltan espacios basándose en el histograma"""
        projection = spacing_info["horizontal_projection"]
        valleys = spacing_info["valleys"]
        avg_valley_width = spacing_info["avg_valley_width"]
        
        # Posiciones candidatas para insertar espacios
        missing_spaces = []
        
        # Si hay pocas separaciones detectadas pero el texto es largo
        chars_per_valley = len(text) / (len(valleys) + 1) if valleys else len(text)
        
        if chars_per_valley > 8:  # Probablemente hay palabras agrupadas
            # Buscar en el histograma zonas donde debería haber separación
            for i in range(len(projection)):
                # Buscar transiciones bruscas que indican cambio de palabra
                if i > 0 and i < len(projection) - 1:
                    if (projection[i-1] > 0.8 and projection[i] < 0.2 and projection[i+1] > 0.8):
                        missing_spaces.append(i)
        
        return missing_spaces
    
    def visualize_spacing_analysis(self, text: str, spacing_info: Dict[str, Any], 
                                 save_path: str = None) -> None:
        """Visualizar el análisis de espaciado"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Histograma horizontal
        projection = spacing_info["horizontal_projection"]
        x_coords = np.arange(len(projection))
        
        ax1.plot(x_coords, projection, 'b-', linewidth=2)
        ax1.fill_between(x_coords, projection, alpha=0.3)
        ax1.set_title(f'Proyección Horizontal: "{text}"')
        ax1.set_xlabel('Posición Horizontal (píxeles)')
        ax1.set_ylabel('Densidad Normalizada')
        ax1.grid(True, alpha=0.3)
        
        # Marcar valles (espacios detectados)
        valleys = spacing_info["valleys"]
        for valley in valleys:
            ax1.axvline(x=valley, color='red', linestyle='--', alpha=0.7, label='Espacio detectado')
        
        # Marcar picos (centros de texto)
        peaks = spacing_info["peaks"]
        for peak in peaks:
            ax1.axvline(x=peak, color='green', linestyle=':', alpha=0.7, label='Centro de texto')
        
        # Estadísticas
        stats_text = f"""
        Ancho total: {spacing_info['total_width']} px
        Espacios detectados: {len(valleys)}
        Ancho promedio entre espacios: {spacing_info['avg_valley_width']:.1f} px
        Caracteres por espacio: {len(text)/(len(valleys)+1):.1f}
        """
        
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Estadísticas de Espaciado')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.debug_mode:
            plt.show()
        plt.close()
    
    def process_line(self, line_data: Dict[str, Any], image_shape: Tuple[int, int] = (3000, 2000)) -> Dict[str, Any]:
        """Procesar una línea de texto para detectar espaciado"""
        text = line_data['text']
        coords = line_data['polygon_coords']
        bbox = self.get_polygon_bbox(coords)
        
        # Crear máscara del texto
        mask = self.create_text_mask(image_shape, coords)
        
        # Analizar espaciado
        spacing_info = self.analyze_horizontal_spacing(mask, bbox)
        
        # Detectar espacios faltantes
        missing_spaces = self.detect_missing_spaces(text, spacing_info)
        
        # Resultado
        result = {
            "original_text": text,
            "bbox": bbox,
            "spacing_analysis": spacing_info,
            "missing_spaces": missing_spaces,
            "needs_separation": len(missing_spaces) > 0 or len(text) > 15,
            "confidence": line_data.get('confidence', 0)
        }
        
        return result

def main():
    separator = WordSeparator()
    
    # Cargar archivo de ejemplo
    input_file = "prueba12_ocr_raw_results.json"
    ocr_data = separator.load_ocr_results(input_file)
    
    print("ANÁLISIS DE ESPACIADO CON HISTOGRAMAS")
    print("=" * 50)
    
    # Procesar las primeras 10 líneas como ejemplo
    lines = ocr_data['ocr_raw_results']['paddleocr']['lines'][:10]
    
    for i, line in enumerate(lines):
        print(f"\nLínea {i+1}: '{line['text']}'")
        
        result = separator.process_line(line)
        
        print(f"  Necesita separación: {result['needs_separation']}")
        print(f"  Espacios faltantes: {len(result['missing_spaces'])}")
        print(f"  Ancho total: {result['bbox'][2] - result['bbox'][0]} px")
        
        # Visualizar si hay problemas de espaciado
        if result['needs_separation']:
            save_path = f"spacing_analysis_line_{i+1}.png"
            separator.visualize_spacing_analysis(
                result['original_text'], 
                result['spacing_analysis'], 
                save_path
            )
            print(f"  Visualización guardada: {save_path}")

if __name__ == "__main__":
    main()