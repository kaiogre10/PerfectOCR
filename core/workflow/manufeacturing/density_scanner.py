# PerfectOCR/core/workflow/preprocessing/density_scanner.py
from sklearnex import patch_sklearn
patch_sklearn
import math
import json
import os
import numpy as np
from typing import Dict, Any
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class DensityScanner:
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
    
    def calculate_values(self, config: Dict, ):
    
    def calculate_mean(values):
    """Calcula la media"""
    return sum(values) / len(values)

    def calculate_variance(values):
    """Calcula la varianza muestral"""
    mean = calculate_mean(values)
    squared_deviations = [(x - mean) ** 2 for x in values]
    return sum(squared_deviations) / (len(values) - 1)

    def calculate_std_dev(variance):
    """Calcula la desviación estándar"""
    return math.sqrt(variance)

    def calculate_skewness(values):
    """Calcula la asimetría (skewness)"""
    mean = calculate_mean(values)
    n = len(values)
    std_dev = calculate_std_dev(calculate_variance(values))
    
    if std_dev == 0:
        return 0
    
    moment3 = sum(((x - mean) / std_dev) ** 3 for x in values)
    return moment3 / n

    def calculate_percentiles(values):
    """Calcula percentiles 25, 50, 75"""
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    def percentile(p):
        index = (p / 100) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    return {
        'p25': percentile(25),
        'p50': percentile(50),
        'p75': percentile(75)
    }

    def calculate_iqr(p25, p75):
    """Calcula rango intercuartil"""
    return p75 - p25

    def analyze_line(line):
        """Analiza una línea de texto y retorna estadísticas"""
        values = convert_line_to_values(line)
        
        if len(values) < 2:
            return None
        
        mean = calculate_mean(values)
        variance = calculate_variance(values)
        std_dev = calculate_std_dev(variance)
        skewness = calculate_skewness(values)
        percentiles = calculate_percentiles(values)
        iqr = calculate_iqr(percentiles['p25'], percentiles['p75'])
        
        return {
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev,
            'skewness': skewness,
            'p25': percentiles['p25'],
            'p50': percentiles['p50'],
            'p75': percentiles['p75'],
            'iqr': iqr,
            'count': len(values)
        }

    def process_lines(lines):
    """Procesa múltiples líneas y muestra resultados"""
    # Encabezados con ancho fijo
    print(f"{'Media':<10}{'Varianza':<12}{'Desv.Est':<10}{'Asimetría':<12}{'P25':<8}{'P50':<8}{'P75':<8}{'IQR':<8}{'Conteo':<8}")
    print("-" * 80)

    for i, line in enumerate(lines):
        result = analyze_line(line)
        if result:
            # Valores con ancho fijo
            print(f"{result['mean']:<10.2f}{result['variance']:<12.2f}{result['std_dev']:<10.2f}{result['skewness']:<12.2f}{result['p25']:<8.2f}{result['p50']:<8.2f}{result['p75']:<8.2f}{result['iqr']:<8.2f}{result['count']:<8}")
        else:
            print(f"Línea {i+1}: Datos insuficientes")

    def prepare_features_for_clustering(results):
    """Prepara features para DBSCAN con características mejoradas"""
    features = []
    for result in results:
        if result:
            features.append([
                result['count'],      # Número de caracteres
                result['mean'],       # Media
                result['std_dev'],    # Desviación estándar
                result['iqr'],        # Rango intercuartil
                result['p50']         # Mediana
            ])
    return np.array(features)



    def cluster_lines(lines):
        """Aplica DBSCAN a las líneas"""
        results = [analyze_line(line) for line in lines]
        
        valid_results = [r for r in results if r is not None]
        valid_indices = [i for i, r in enumerate(results) if r is not None]
        
        if len(valid_results) < 2:
            return [], []
        
        features = prepare_features_for_clustering(valid_results)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        n_samples = len(features_scaled)
        eps = 1.0
        min_samples = 2
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(features_scaled)
        
        return labels, valid_indices

    def identify_table_lines(lines):
    """Identifica líneas de tabla"""
    labels, valid_indices = cluster_lines(lines)
    
    if len(labels) == 0:
        return []
    
    # Encontrar cluster más grande (excluyendo ruido -1)
    unique_labels = [l for l in set(labels) if l != -1]
    if not unique_labels:
        return []
    
    cluster_sizes = {label: list(labels).count(label) for label in unique_labels}
    main_cluster = max(cluster_sizes, key=cluster_sizes.get)
    
    # Devolver índices originales de líneas de tabla
    table_indices = [valid_indices[i] for i, label in enumerate(labels) if label == main_cluster]
    
    return table_indices

    def expand_to_consecutive_interval(indices):
        """Expande lista de índices a intervalo consecutivo"""
        start = indices[0]
        end = indices[-1]
        return list(range(start, end + 1))

    def process_lines_with_clustering(lines):
        """Procesa líneas con clustering y señala las líneas agrupadas"""
        print(f"{'Línea':<8}{'Media':<10}{'Varianza':<12}{'Desv.Est':<10}{'Asimetría':<12}{'P25':<8}{'P50':<8}{'P75':<8}{'IQR':<8}{'Conteo':<8}")
        print("-" * 90)
        
        for i, line in enumerate(lines):
            result = analyze_line(line)
            if result:
                print(f"{i+1:<8}{result['mean']:<10.2f}{result['variance']:<12.2f}{result['std_dev']:<10.2f}{result['skewness']:<12.2f}{result['p25']:<8.2f}{result['p50']:<8.2f}{result['p75']:<8.2f}{result['iqr']:<8.2f}{result['count']:<8}")
            else:
                print(f"{i+1:<8}Línea {i+1}: Datos insuficientes")
        
        # Identificar líneas de tabla
        table_indices = identify_table_lines(lines)
        consecutive_indices = expand_to_consecutive_interval(table_indices)
        
        print(f"\nLíneas agrupadas (base 1): {[i+1 for i in consecutive_indices]}")
        for idx in consecutive_indices:
            print(f"{idx+1}: {lines[idx]}")
        
        return consecutive_indices

    def process_lines_with_dbscan(lines, output_file="lineas_agrupadas.txt"):
        """Procesa líneas con DBSCAN y guarda resultados en archivo"""
        # Preparar contenido para el archivo
        output_content = []
        output_content.append("DETECCIÓN DE LÍNEAS TABULARES CON DBSCAN")
        output_content.append("=" * 50)
        
        # Mostrar tabla de estadísticas
        stats_header = f"{'Línea':<8}{'Media':<10}{'Varianza':<12}{'Desv.Est':<10}{'Asimetría':<12}{'P25':<8}{'P50':<8}{'P75':<8}{'IQR':<8}{'Conteo':<8}"
        output_content.append(stats_header)
        
        separator = "-" * 90
        output_content.append(separator)
        
        for i, line in enumerate(lines):
            result = analyze_line(line)
            if result:
                stats_line = f"{i+1:<8}{result['mean']:<10.2f}{result['variance']:<12.2f}{result['std_dev']:<10.2f}{result['skewness']:<12.2f}{result['p25']:<8.2f}{result['p50']:<8.2f}{result['p75']:<8.2f}{result['iqr']:<8.2f}{result['count']:<8}"
                output_content.append(stats_line)
            else:
                error_line = f"{i+1:<8}Línea {i+1}: Datos insuficientes"
                output_content.append(error_line)
        
        output_content.append("")
        output_content.append("RESULTADOS DBSCAN:")
        
        # Aplicar DBSCAN
        dbscan_indices = identify_table_lines(lines)
        dbscan_consecutive = expand_to_consecutive_interval(dbscan_indices)
        
        dbscan_result = f"Líneas agrupadas (base 1): {[i+1 for i in dbscan_consecutive]}"
        output_content.append(dbscan_result)
        
        output_content.append("")
        output_content.append(f"Total de líneas agrupadas: {len(dbscan_consecutive)}")
        
        # Agregar las líneas agrupadas
        output_content.append("")
        output_content.append("LÍNEAS TABULARES DETECTADAS:")
        output_content.append("=" * 30)
        
        if dbscan_consecutive:
            for idx in dbscan_consecutive:
                output_content.append(f"{idx+1}: {lines[idx]}")
        else:
            output_content.append("No se detectaron líneas tabulares.")
        
        # Guardar en archivo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_content))
        
        return dbscan_consecutive

    def load_lines_from_json(json_file_path):
        """Carga líneas desde un archivo JSON"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraer líneas del JSON con estructura grouped_lines
            if isinstance(data, dict) and 'grouped_lines' in data:
                lines = []
                for line_group in data['grouped_lines']:
                    if isinstance(line_group, list):
                        # Extraer texto de cada palabra en la línea
                        line_text = ' '.join([word.get('text', '') for word in line_group if isinstance(word, dict)])
                        if line_text.strip():  # Solo agregar líneas no vacías
                            lines.append(line_text)
                return lines
            elif isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Buscar la clave que contenga las líneas
                for key in ['lines', 'text', 'content', 'data']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
            return []
        except Exception as e:
            print(f"Error al cargar {json_file_path}: {e}")
            return []

    def process_json_files():
        """Procesa todos los archivos JSON en la carpeta input/"""
        input_dir = "input"
        
        if not os.path.exists(input_dir):
            return
        
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        if not json_files:
            return
        
        for json_file in json_files:
            json_path = os.path.join(input_dir, json_file)
            
            lines = load_lines_from_json(json_path)
            
            if not lines:
                continue
            
            # Generar nombre de archivo de salida basado en el JSON de entrada
            output_filename = json_file.replace('.json', '_lineas_agrupadas.txt')
            
            # Procesar las líneas con DBSCAN
            process_lines_with_dbscan(lines, output_filename)

    # Ejemplo de uso:
    if __name__ == "__main__":
        process_json_files()