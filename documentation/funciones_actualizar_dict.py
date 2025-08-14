# *MANUAL COMPLETO: Creación de Métodos Específicos para DataFormatter**

# ## **1. FILOSOFÍA Y ARQUITECTURA**

# ### **Principios Fundamentales:**
# - **Schema JSON = Verdad Absoluta**: Todo debe validar contra `WORKFLOW_DICT`
# - **DataFormatter = Válvula Única**: Los workers NUNCA tocan el dict directamente
# - **Dataclasses = Estructura Interna**: Se usan para tipado y validación antes de insertar
# - **Contenedor Universal**: El dict es el único almacén de datos del pipeline

# ### **Flujo General:**

# Worker → Datos Crudos → DataFormatter → [Dataclass → Dict] → Validación → Inserción


## **2. PATRÓN PARA CREAR MÉTODOS ESPECÍFICOS**

### **Estructura Base de Cualquier Método:**

def nombre_metodo(self, parametros_necesarios) -> bool:
    """Descripción clara de qué hace el método"""
    try:
        # 1. PREPARAR DATOS
        # - Validar parámetros de entrada
        # - Crear dataclass si aplica
        
        # 2. UBICAR DESTINO EN EL DICT
        # - Navegar a la sección correcta del dict
        # - Verificar que existe el contenedor padre
        
        # 3. INSERTAR/ACTUALIZAR
        # - Convertir dataclass a dict con asdict() si aplica
        # - Insertar en la ubicación correcta
        
        # 4. VALIDAR
        # - Siempre llamar self._validate_structure()
        
        return True
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error en {nombre_metodo}: {e}")
        return False

## **3. CASOS DE USO Y EJEMPLOS DETALLADOS**

### **CASO 1: Actualizar Campo Simple (Sin Dataclass)**

#**Objetivo**: Actualizar el color de la imagen en metadata

def update_image_color(self, new_color: str) -> bool:
    """Actualiza el color de la imagen en metadata"""
    try:
        # 1. VALIDAR ENTRADA
        if not isinstance(new_color, str):
            raise TypeError("Color debe ser string")
            
        # 2. UBICAR DESTINO
        # Ruta: dict_id["metadata"]["color"]
        if "metadata" not in self.dict_id:
            raise KeyError("Metadata no existe en el dict")
            
        # 3. INSERTAR
        self.dict_id["metadata"]["color"] = new_color
        
        # 4. VALIDAR
        self._validate_structure()
        return True
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error actualizando color: {e}")
        return False


# En tu worker
formatter = context["formatter"]  # Recibido del pipeline
success = formatter.update_image_color("RGB")
if not success:
    logger.error("No se pudo actualizar el color")
```

---

### **CASO 2: Actualizar Campo con Dataclass**
def update_polygon_geometry(self, poly_id: str, polygon_coords: List[List[float]], 
                          bounding_box: List[float], centroid: List[float]) -> bool:
    """Actualiza solo la geometría básica de un polígono específico"""
    try:
        # 1. VALIDAR ENTRADA
        if poly_id not in self.dict_id["polygons"]:
            raise ValueError(f"Polígono {poly_id} no existe")
            
        # 2. CREAR DATACLASS
        geometry_obj = Geometry(
            polygon_coords=polygon_coords,
            bounding_box=bounding_box,
            centroid=centroid
        )
        
        # 3. UBICAR DESTINO Y ACTUALIZAR
        # Ruta: dict_id["polygons"][poly_id]["geometry"]
        self.dict_id["polygons"][poly_id]["geometry"] = asdict(geometry_obj)
        
        # 4. VALIDAR
        self._validate_structure()
        return True
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error actualizando geometría: {e}")
        return False
polygon_coords = [[100.0, 200.0], [150.0, 250.0], [200.0, 300.0]]
bounding_box = [100.0, 200.0, 200.0, 300.0]
centroid = [150.0, 250.0]

formatter = context["formatter"]
success = formatter.update_polygon_geometry("poly_0001", polygon_coords, bounding_box, centroid)

### **CASO 3: Actualizar Subcampo Específico**

def update_polygon_perimeter(self, poly_id: str, perimeter: float) -> bool:
    """Actualiza solo el perímetro en cropedd_geometry de un polígono"""
    try:
        # 1. VALIDAR ENTRADA
        if poly_id not in self.dict_id["polygons"]:
            raise ValueError(f"Polígono {poly_id} no existe")
            
        # 2. VERIFICAR ESTRUCTURA
        poly = self.dict_id["polygons"][poly_id]
        if "cropedd_geometry" not in poly:
            raise KeyError(f"cropedd_geometry no existe en polígono {poly_id}")
            
        # 3. ACTUALIZAR SUBCAMPO
        # Ruta: dict_id["polygons"][poly_id]["cropedd_geometry"]["perimeter"]
        self.dict_id["polygons"][poly_id]["cropedd_geometry"]["perimeter"] = float(perimeter)
        
        # 4. VALIDAR
        self._validate_structure()
        return True
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error actualizando perímetro: {e}")
        return False

# En tu worker (ej: después de calcular perímetro)
calculated_perimeter = 450.75
formatter = context["formatter"]
success = formatter.update_polygon_perimeter("poly_0001", calculated_perimeter)

def update_polygon_cropped_image(self, poly_id: str, cropped_img: np.ndarray) -> bool:
    """Actualiza imagen recortada solo si el polígono existe y tiene cropedd_geometry"""
    try:
        # 1. VALIDACIONES MÚLTIPLES
        if poly_id not in self.dict_id["polygons"]:
            raise ValueError(f"Polígono {poly_id} no existe")
            
        poly = self.dict_id["polygons"][poly_id]
        if "cropedd_geometry" not in poly or not poly["cropedd_geometry"]:
            raise ValueError(f"Polígono {poly_id} no tiene geometría de recorte")
            
        # 2. VERIFICAR QUE HAY COORDS DE PADDING
        padding_coords = poly["cropedd_geometry"].get("padding_coords")
        if not padding_coords or all(coord == 0 for coord in padding_coords):
            raise ValueError(f"Polígono {poly_id} no tiene coordenadas de padding válidas")
            
        # 3. ACTUALIZAR
        self.dict_id["polygons"][poly_id]["cropped_img"] = cropped_img
        
        # 4. VALIDAR
        self._validate_structure()
        return True
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error actualizando imagen recortada: {e}")
        return False

## **4. PATRÓN DE LLAMADA DESDE WORKERS**

### **Patrón General en Workers:**
class MiWorker(AbstractWorker):
    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        # 1. OBTENER FORMATTER
        formatter = context.get("formatter")
        if not formatter:
            logger.error("Formatter no disponible en contexto")
            return image
            
        # 2. PROCESAR DATOS
        mi_resultado = self.mi_logica_de_procesamiento(image)
        
        # 3. ENVIAR AL FORMATTER
        success = formatter.mi_metodo_especifico(mi_resultado)
        if not success:
            logger.error("Error enviando datos al formatter")
            
        return image
```

### **Ejemplo Concreto:**
class PolygonExtractor(AbstractWorker):
    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        formatter = context["formatter"]
        
        # Tu lógica de extracción
        # ...código de recorte...
        
        # Actualizar padding geometry
        success = formatter.update_polygon_padding_coords("poly_0001", padding_coords)
        
        return image

## **5. TEMPLATE PARA COPIAR Y PEGAR**
def update_CAMPO_ESPECIFICO(self, parametros_necesarios) -> bool:
    """Descripción clara de la funcionalidad"""
    try:
        # 1. VALIDAR ENTRADA
        # Agregar validaciones necesarias
        
        # 2. CREAR DATACLASS (si aplica)
        # obj = DataclassName(param1=value1, param2=value2)
        
        # 3. UBICAR Y ACTUALIZAR
        # self.dict_id["ruta"]["al"]["campo"] = valor_o_asdict(obj)
        
        # 4. VALIDAR
        self._validate_structure()
        return True
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error en update_CAMPO_ESPECIFICO: {e}")
        return False

## **6. CHECKLIST DE VALIDACIÓN**

# Antes de implementar cualquier método, verifica:

# El método sigue el patrón try/except  
# Valida parámetros de entrada  
# Verifica existencia de contenedores padre  
# Usa dataclass si mejora la estructura  
# Inserta en la ruta correcta del dict  
# Llama a _validate_structure() al final  
# Retorna bool indicando éxito/fallo  
# Tiene docstring descriptivo  
# Maneja errores informativamente  
# Es consistente con el schema JSON  