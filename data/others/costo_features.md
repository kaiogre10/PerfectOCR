
### Tabla de features de regionprops (ordenadas por costo computacional)

| Feature                      | Descripción breve                                                                 | Costo computacional (↑ = más costoso) |
|------------------------------|----------------------------------------------------------------------------------|:-------------------------------------:|
| **convex_area**              | Área de la envolvente convexa                                                    | Muy alto (requiere convex hull)       |
| **convex_image**             | Imagen binaria de la envolvente convexa                                          | Muy alto (requiere convex hull)       |
| **feret_diameter_max**       | Diámetro máximo de Feret (distancia máxima entre bordes)                         | Muy alto (requiere convex hull)       |
| **perimeter_crofton**        | Perímetro usando la fórmula de Crofton                                           | Muy alto (requiere múltiples líneas)  |
| **moments_hu**               | Momentos invariantes de Hu                                                       | Alto (requiere todos los momentos)    |
| **weighted_moments_hu**      | Momentos de Hu ponderados por intensidad                                         | Alto (requiere todos los momentos)    |
| **inertia_tensor**           | Tensor de inercia 2x2                                                            | Alto (requiere momentos centrales)    |
| **inertia_tensor_eigvals**   | Eigenvalores del tensor de inercia                                               | Alto (requiere descomposición)        |
| **moments**                  | Matriz de momentos espaciales                                                    | Alto (requiere recorrer todos los píxeles) |
| **moments_central**          | Momentos centrales                                                               | Alto                                 |
| **moments_normalized**       | Momentos normalizados                                                            | Alto                                 |
| **weighted_moments**         | Momentos ponderados por intensidad                                               | Alto                                 |
| **weighted_moments_central** | Momentos centrales ponderados                                                    | Alto                                 |
| **weighted_moments_normalized** | Momentos normalizados ponderados                                              | Alto                                 |
| **filled_area**              | Área con huecos rellenados                                                       | Medio-alto (requiere rellenar huecos) |
| **filled_image**             | Imagen con huecos rellenados                                                     | Medio-alto (requiere rellenar huecos) |
| **coords**                   | Coordenadas de todos los píxeles de la región                                    | Medio (requiere recorrer la región)   |
| **image**                    | Imagen binaria de la región (recortada)                                          | Medio (requiere recorte)              |
| **intensity_image**          | Imagen de intensidad de la región                                                | Medio (requiere recorte)              |
| **bbox_area**                | Área de la bounding box                                                          | Bajo (cálculo directo)                |
| **bbox**                     | Bounding box (min_row, min_col, max_row, max_col)                                | Bajo (cálculo directo)                |
| **area**                     | Número de píxeles en la región                                                   | Bajo (conteo)                         |
| **centroid**                 | Centro de masa de la región                                                      | Bajo (promedio de coordenadas)        |
| **local_centroid**           | Centroide relativo al bbox                                                       | Bajo                                  |
| **weighted_centroid**        | Centroide ponderado por intensidad                                               | Bajo-medio (si la imagen es binaria, es igual al centroid) |
| **weighted_local_centroid**  | Centroide ponderado local                                                        | Bajo-medio                            |
| **eccentricity**             | Excentricidad de la elipse equivalente                                           | Bajo-medio (requiere momentos)        |
| **extent**                   | Proporción area/bbox_area                                                        | Bajo                                  |
| **major_axis_length**        | Longitud del eje mayor de la elipse equivalente                                  | Bajo-medio (requiere momentos)        |
| **minor_axis_length**        | Longitud del eje menor de la elipse equivalente                                  | Bajo-medio (requiere momentos)        |
| **orientation**              | Ángulo del eje mayor respecto al horizontal                                      | Bajo-medio (requiere momentos)        |
| **equivalent_diameter**      | Diámetro de círculo con misma área                                               | Bajo                                  |
| **equivalent_diameter_area** | Diámetro de círculo con misma área (puede ser redundante)                        | Bajo                                  |
| **perimeter**                | Perímetro de la región (bordes)                                                  | Bajo-medio (requiere recorrer bordes) |
| **solidity**                 | Proporción area/convex_area                                                      | Bajo-medio (depende de convex_area)   |
| **euler_number**             | Número Euler (objetos - huecos)                                                  | Bajo-medio (requiere análisis de huecos) |
| **max_intensity**            | Valor máximo de intensidad                                                       | Bajo                                  |
| **mean_intensity**           | Valor promedio de intensidad                                                     | Bajo                                  |
| **min_intensity**            | Valor mínimo de intensidad                                                       | Bajo                                  |
| **label**                    | Etiqueta de la región                                                            | Bajo                                  |
| **slice**                    | Objeto slice para extraer la región                                              | Bajo                                  |
