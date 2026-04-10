# PerfectOCR

Pipeline de reconocimiento óptico de caracteres (OCR) de alto rendimiento para la extracción y estructuración de texto desde imágenes de documentos, con especial énfasis en facturas y órdenes de compra.

---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Pipeline de Procesamiento](#pipeline-de-procesamiento)
- [Base de Datos](#base-de-datos)
- [Salidas y Depuración](#salidas-y-depuración)
- [Tests](#tests)

---

## Descripción

PerfectOCR es un sistema de procesamiento de documentos que combina técnicas avanzadas de visión por computadora con motores OCR modernos para extraer información estructurada de imágenes de documentos. El sistema implementa un pipeline de cuatro etapas: preparación de imagen, preprocesamiento, reconocimiento OCR y vectorización/estructuración de datos.

## Características

- Pipeline modular de cuatro etapas totalmente configurable mediante YAML
- Preprocesamiento de imagen: corrección de ángulo, mejora de tinta, eliminación de ruido (sal y pimienta, Gaussian, CLAHE, nitidez, restauración)
- Motor OCR basado en **PaddleOCR** con soporte para español
- Clasificación semántica del texto extraído (numérico, descriptivo, códigos)
- Estructuración tabular usando similitud coseno y análisis geométrico
- Integración con **PostgreSQL** para persistencia de datos
- Aceleración Intel MKL y paralelismo OpenMP
- Modo de depuración con guardado de imágenes intermedias en cada etapa
- Detección automática de entorno de ejecución (local, remoto, Codespaces)

## Arquitectura

El proyecto sigue un diseño orientado a patrones:

- **Factory Pattern** — Fábricas de trabajadores (`WorkerFactory`, `StagersFactory`) para creación dinámica de componentes
- **Builder Pattern** — `ProcessBuilder` y `MainBuilder` orquestan el flujo completo
- **Stager Pattern** — Cada etapa del pipeline es un `Stager` independiente y reemplazable
- **Singleton Pattern** — `ModelsManager` gestiona instancias únicas de los modelos PaddleOCR

```
main.py
└── MainBuilder.activate_main()
    ├── ConfigService        → Carga y valida master_config.yaml
    ├── WorkFlowBuilder      → Descubre y planifica las imágenes a procesar
    ├── ModelsManager        → Inicializa modelos PaddleOCR (Singleton)
    ├── StagersFactory       → Construye las etapas del pipeline
    └── ProcessBuilder       → Ejecuta el pipeline por cada imagen
        ├── ImagePreparationStager
        ├── PreprocessingStager   (opcional)
        ├── OCRStager
        └── VectorizationStager
```

## Estructura del Proyecto

```
PerfectOCR/
├── app/                          # Orquestación y builders principales
│   ├── main_builder.py
│   ├── process_builder.py
│   ├── workflow_builder.py
│   └── database_builder.py
├── config/                       # Gestión de configuración
│   ├── master_config.yaml        # Configuración principal del pipeline
│   └── config_models.py          # Modelos Pydantic para validación
├── core/                         # Lógica de procesamiento central
│   ├── domain/                   # Modelos de dominio y datos
│   ├── factory/                  # Implementaciones del patrón Factory
│   ├── pipeline/                 # Stagers del pipeline
│   ├── workers/                  # Implementaciones de trabajadores
│   │   ├── image_preparation/    # Carga, corrección, detección geométrica
│   │   ├── preprocessing/        # Filtros de mejora de imagen
│   │   ├── ocr/                  # Reconocimiento OCR y refinamiento
│   │   └── vectorial_transformation/ # Vectorización y estructuración tabular
│   └── utils/                    # Utilidades generales
├── services/                     # Capa de servicios
│   ├── config_service.py
│   ├── cache_service.py
│   ├── output_service.py
│   ├── database_service.py
│   └── postgres_service.py
├── data/
│   └── models/                   # Modelos PaddleOCR (no incluidos en el repo)
├── data_base/
│   └── data_base.json            # Esquema de base de datos
├── documentation/                # Documentación técnica adicional
├── tests/
│   └── metrics.py                # Analizador de métricas de código
├── tools/                        # Herramientas auxiliares
├── main.py                       # Punto de entrada principal
└── master_config.yaml            # Configuración principal
```

## Requisitos

- Python 3.8+
- Sistema operativo: Windows (las rutas por defecto apuntan a `C:/` y `D:/`)
- Modelos PaddleOCR en español (detección y reconocimiento)

### Dependencias principales

```
paddleocr
pydantic
pyyaml
numpy
pandas
opencv-python
pillow
scikit-learn
psycopg2
scipy
```

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/kaiogre10/PerfectOCR.git
cd PerfectOCR

# 2. Instalar dependencias
pip install paddleocr pydantic pyyaml numpy pandas opencv-python pillow scikit-learn psycopg2 scipy

# 3. Descargar los modelos PaddleOCR en español y ubicarlos en:
#    data/models/paddle/det/es/   (modelos de detección)
#    data/models/paddle/rec/es/   (modelos de reconocimiento)

# 4. Crear la carpeta de entrada
mkdir input2
# Colocar las imágenes a procesar en input2/
```

## Configuración

La configuración principal se encuentra en `config/master_config.yaml` (y una copia en la raíz `master_config.yaml`).

### Parámetros clave

| Sección | Descripción |
|---------|-------------|
| `pipeline_secuence` | Define qué trabajadores se ejecutan en cada etapa |
| `models_config` | Rutas a modelos PaddleOCR, uso de GPU, threads |
| `output_flags` | Activa/desactiva guardado de imágenes intermedias |
| `modules_config` | Parámetros de ajuste fino por trabajador |
| `utils_config` | Rangos de DPI, extensiones de imagen, umbrales |

### Variables de entorno relevantes

| Variable | Valor por defecto | Descripción |
|----------|-------------------|-------------|
| `OMP_NUM_THREADS` | `4` | Threads de OpenMP |
| `MKL_NUM_THREADS` | `4` | Threads de Intel MKL |
| `FLAGS_use_mkldnn` | `1` | Habilita Intel Deep Learning Library |

### Modo de prueba

En `main.py` existe la constante `TEST_MODE = True`. Cuando está activa, el sistema omite validaciones estrictas y facilita la depuración.

## Uso

```bash
# Ejecución básica (usa rutas por defecto definidas en main.py)
python main.py
```

El sistema buscará imágenes en la carpeta `input2/` y escribirá los resultados en `output/`.

Para cambiar las rutas de entrada/salida o el archivo de configuración, editar las constantes al inicio de `main.py`:

```python
DEFAULT_INPUT_PATH  = ["input2"]
DEFAULT_OUTPUT_PATH = ["output"]
config_path         = "config/master_config.yaml"
```

## Pipeline de Procesamiento

### Etapa 1 — Preparación de Imagen (`imagepre_stage`)

| Trabajador | Descripción |
|------------|-------------|
| `image_loader` | Carga la imagen y aplica corrección de ángulo |
| `ink_enhancement` | Mejora la visibilidad de la tinta en el documento |
| `geometry_detector` | Detecta regiones de texto mediante análisis geométrico |
| `polygon_extractor` | Extrae recortes de cada región de texto detectada |

### Etapa 2 — Preprocesamiento (`preprocessing_stage`) *(opcional)*

| Trabajador | Descripción |
|------------|-------------|
| `restorer` | Restauración general de imagen |
| `moire` | Eliminación del patrón moiré |
| `sp` | Filtro de ruido sal y pimienta |
| `gauss` | Desenfoque gaussiano |
| `clahe` | Ecualización adaptativa de histograma con límite de contraste |
| `sharp` | Filtro de nitidez |

### Etapa 3 — OCR (`ocr_stage`)

| Trabajador | Descripción |
|------------|-------------|
| `paddle_wrapper` | Ejecuta PaddleOCR sobre los recortes preprocesados |
| `text_refiner` | Refinamiento semántico y clasificación del texto |
| `data_finder` | Extracción y localización de datos estructurados |

### Etapa 4 — Vectorización (`vector_stage`)

| Trabajador | Descripción |
|------------|-------------|
| `vectorizer` | Convierte el texto en vectores de características |
| `cos_sim` | Agrupa texto relacionado usando similitud coseno |
| `table_structurer` | Organiza la información en tablas estructuradas |
| `math_max` | Validación matemática de cantidades y totales |

## Base de Datos

El esquema de base de datos objetivo (PostgreSQL) está definido en `data_base/data_base.json` e incluye las siguientes tablas:

| Tabla | Descripción |
|-------|-------------|
| `RegistrosCompra` | Registros generales de compra |
| `DetallesCompra` | Líneas de detalle de cada compra |
| `TransaccionesCompra` | Transacciones asociadas a cada compra |
| `Productos` | Catálogo de productos |
| `Proveedores` | Información de proveedores |
| `Clientes` | Información de clientes |

La integración con PostgreSQL se gestiona a través de `services/postgres_service.py` y `services/database_service.py`.

## Salidas y Depuración

El sistema puede guardar imágenes intermedias en disco para facilitar la depuración. Esto se controla mediante las flags en la sección `output_flags` de `master_config.yaml`.

Algunas de las salidas disponibles:

- `full_img` — Imagen completa cargada
- `bin_full_img` — Imagen binarizada completa
- `angle_corrected` — Imagen con ángulo corregido
- `cropped_img` — Recortes de regiones de texto
- `ocr_raw` — Resultado OCR sin procesar
- `cleanned_text` — Texto limpiado tras aplicar umbrales de confianza
- `reconstructed_lines` — Líneas reconstruidas
- `table_structured` — Tabla final estructurada

## Tests

El directorio `tests/` contiene `metrics.py`, un analizador de métricas de código que inspecciona el proyecto.

```bash
python tests/metrics.py
```
