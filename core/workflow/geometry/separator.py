# PerfectOCR/core/workflow/preprocessing/binarization.py
from sklearnex import patch_sklearn
patch_sklearn
import cv2
import numpy as np
from skimage.measure import regionprops

def process_polygons(binarized_img, polygons_coords):
    """
    Procesa los polígonos para detectar y separar los problemáticos.
    
    Args:
        binarized_img (np.ndarray): Imagen binarizada.
        polygons_coords (list): Lista de coordenadas de los polígonos.
    
    Returns:
        list: Lista de polígonos refinados.
    """
    refined_polygons = []
    image_height, image_width = binarized_img.shape[:2]
    
    # Escalar umbrales según el ancho de la imagen
    min_area = 10 * (image_width / 1000)  # Área mínima adaptativa
    tolerance = 1.0 * (image_width / 1000)  # Tolerancia para simplificación
    
    # Paso 1: Calcular relaciones blanco/negro para todos los polígonos
    relations = []
    for coords in polygons_coords:
        # Simplificar el contorno del polígono
        simplified = cv2.approxPolyDP(np.array(coords), epsilon=tolerance, closed=True)
        mask = np.zeros_like(binarized_img)
        cv2.fillPoly(mask, [simplified], 255)
        props = regionprops(mask // 255)[0]
        white_pixels = np.sum(binarized_img[mask == 255])
        total_pixels = props.area
        relation = white_pixels / total_pixels if total_pixels > 0 else 0
        relations.append(relation)
    
    # Paso 2: Identificar outliers usando estadísticas
    mean_relation = np.mean(relations)
    std_relation = np.std(relations)
    outlier_threshold = 2 * std_relation  # Umbral para considerar outliers
    
    # Paso 3: Procesar cada polígono
    for i, coords in enumerate(polygons_coords):
        relation = relations[i]
        if abs(relation - mean_relation) > outlier_threshold:
            # Polígono problemático: segmentar en sub-regiones
            roi = extract_roi(binarized_img, coords)
            contours = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if filtered_contours:
                # Calcular centroides y espaciado promedio
                centroids = [cv2.moments(c)['m10'] / cv2.moments(c)['m00'] for c in filtered_contours]
                spacing = np.mean([abs(centroids[j] - centroids[j-1]) for j in range(1, len(centroids))]) if len(centroids) > 1 else 0
                grouped = group_contours(filtered_contours, spacing * 1.5 if spacing > 0 else min_area)
                sub_bboxes = [cv2.boundingRect(np.concatenate(g)) for g in grouped]
                refined_polygons.extend(sub_bboxes)
        else:
            # Polígono no problemático: mantenerlo como está
            refined_polygons.append(coords)
    
    return refined_polygons

def extract_roi(img, coords):
    """
    Extrae la región de interés (ROI) de la imagen basada en las coordenadas del polígono.
    
    Args:
        img (np.ndarray): Imagen binarizada.
        coords (list): Coordenadas del polígono.
    
    Returns:
        np.ndarray: ROI recortada.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [np.array(coords)], 255)
    x, y, w, h = cv2.boundingRect(np.array(coords))
    roi = img[y:y+h, x:x+w]
    return roi

def group_contours(contours, max_distance):
    """
    Agrupa contornos cercanos basándose en la distancia máxima.
    
    Args:
        contours (list): Lista de contornos.
        max_distance (float): Distancia máxima para considerar contornos cercanos.
    
    Returns:
        list: Lista de grupos de contornos.
    """
    if not contours:
        return []
    
    # Ordenar contornos por posición horizontal
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    groups = [[contours[0]]]
    
    for contour in contours[1:]:
        last_group = groups[-1]
        last_contour = last_group[-1]
        dist = cv2.boundingRect(contour)[0] - (cv2.boundingRect(last_contour)[0] + cv2.boundingRect(last_contour)[2])
        if dist < max_distance:
            last_group.append(contour)
        else:
            groups.append([contour])
    
    return groups