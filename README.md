# PerfectOCR

PerfectOCR es un pipeline OCR para digitalizar imágenes de documentos, extraer texto, reconstruir líneas, estructurar tablas de compra y, opcionalmente, insertar el resultado en PostgreSQL.

El proyecto está organizado como una cadena configurable de *stagers* y *workers*: carga/preparación de imagen, preprocesamiento opcional, OCR, vectorización/estructuración y persistencia.

---

## Tabla de contenidos

- [Estado actual](#estado-actual)
- [Arquitectura](#arquitectura)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Descripción de módulos y archivos](#descripción-de-módulos-y-archivos)
- [Configuración](#configuración)
- [Instalación](#instalación)
- [Uso](#uso)
- [Pipeline configurado](#pipeline-configurado)********
- [Base de datos](#base-de-datos)
- [Salidas y depuración](#salidas-y-depuración)
- [Notas de mantenimiento](#notas-de-mantenimiento)

---

## Estado actual

- Punto de entrada real: `main.py`.
- Configuración activa: `config/master_config.yaml`.
- Entradas configuradas: `input/`, `input2/`, `input3/`.
- Salida configurada: `output/`.
- Log principal: `perfectocr.txt`.
- El modo `TEST_MODE` está activo en `main.py`.
- La etapa `preprocessing_stage` está configurada pero actualmente no ejecuta workers porque está vacía en el YAML.
- La etapa `db_stage` existe en configuración, pero no tiene un stager propio; la inserción se dispara desde `app/main_builder.py` después de la vectorización, usando `services/db_service.py`.
- El repositorio contiene modelo Paddle de detección en `data/models/paddle/det/es/` y modelos `WordFinder` en `data/models/wf/`. La ruta de reconocimiento `data/models/paddle/rec/es/` está referenciada en configuración y debe existir para OCR completo.

---

## Arquitectura

El flujo principal es:

```text
main.py
└── app.main_builder.activate_main(...)
    ├── services.config_service.ConfigService
    │   ├── carga config/master_config.yaml
    │   ├── valida estructura con Pydantic
    │   └── arma paquetes de configuración por módulo
    ├── services.system_service.count_and_plan(...)
    │   └── localiza imágenes en rutas input
    ├── app.models_manager.ModelsManager
    │   ├── inicializa PaddleOCR de forma selectiva
    │   └── inicializa WordFinder si aplica
    ├── core.pipeline.stagers_factory.StagersFactory
    │   └── crea stagers a partir de factories de workers
    ├── app.process_builder.ProcessingBuilder
    │   └── procesa cada imagen con un DataFormatter nuevo
    └── services.db_service.DataBaseService
        └── inserta resultados si db_stage está habilitado y hay conexión
```

Patrones usados:

- **Factory**: `MainFactory`, `ImagePreparationFactory`, `PreprocessingFactory`, `OCRFactory`, `VectorizingFactory`.
- **Stager**: cada etapa del pipeline hereda una interfaz común y ejecuta workers en orden.
- **Builder/Director**: `ProcessingBuilder` coordina una imagen completa; `main_builder.py` coordina todo el lote.
- **Singleton**: `ModelsManager` mantiene una instancia compartida de PaddleOCR y WordFinder.
- **Data transfer por dataclasses**: `core/domain/data_models.py` define la estructura de datos que viaja por el pipeline.

---

## Estructura del proyecto

```text
PerfectOCR/
├── app/
│   ├── main_builder.py                      # Orquestador de ejecución completa
│   ├── models_manager.py                    # Singleton de modelos PaddleOCR y WordFinder
│   └── process_builder.py                   # Coordina la transformación de la imagen en el output y recibe el output de la fase 1
├── config/
│   ├── config_models.py                     # Esquemas Pydantic del YAML
│   └── master_config.yaml                   # Configuración principal del pipeline
├── core/
│   ├── domain/
│   │   ├── data_formatter.py                # API de lectura/escritura del workflow en memoria, estandariza
│   │   └── data_models.py                   # Dataclasses del dominio OCR
│   ├── factory/
│   │   ├── abstract_factory.py              # Factory base genérica
│   │   ├── abstract_stager.py               # Interfaz base de stagers
│   │   ├── abstract_worker.py               # Interfaces base de workers
│   │   └── main_factory.py                  # Agrupa factories por módulo
│   ├── pipeline/
│   │   ├── image_preparation_stager.py      # Ejecuta workers de preparación de la imagen para ocr
│   │   ├── ocr_stager.py                    # Ejecuta workers de OCR y procesamiento textual
│   │   ├── preprocessing_stager.py          # Ejecuta workers de preprocesamiento de imagen
│   │   ├── vectorization_stager.py          # Ejecuta reconstrucción y extracción del output del sistema
│   │   └── stagers_factory.py               # Ensambla stagers según configuración
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_utils.py                    # Constantes, codificadores usadas a lo largo del pipeline
│   │   ├── image_utils.py                   # Utilidades OpenCV/skimage para imágenes
│   │   ├── math_utils.py                    # Features, clustering, similitud y geometría
│   │   ├── text_utils.py                    # Normalización, clasificación y post procesamiento de texto
│   │   └── word_finder.py                   # Buscador de palabras clave pre entrenado usando n-gramas
│   └── workers/
│       ├── image_preparation/
│       │   ├── image_preparation_factory.py # Factory de workers de preparación
│       │   ├── image_loader.py              # Carga imagen y crea workflow inicial
│       │   ├── angle_corrector.py           # Deskew por líneas/Hough
│       │   ├── ink_enhancer.py              # Mejora de tinta y limpieza geométrica
│       │   ├── geometry_detector.py         # Detección de regiones con PaddleOCR det
│       │   └── poly_gone.py                 # Extrae recortes por polígonos detectados
│       ├── preprocessing/
│       │   ├── clahe.py                     # Mejora contraste con CLAHE
│       │   ├── gauss.py                     # Detección/aplicación de suavizado gaussiano
│       │   ├── preprocessing_factory.py     # Factory de preprocesamiento
│       │   ├── restorer.py                  # Restauración morfológica de recortes
│       │   ├── sharp.py                     # Mejora de nitidez
│       │   └── sp.py                        # Limpieza de ruido sal y pimienta
│       ├── ocr/
│       │   ├── ocr_factory.py               # Factory OCR
│       │   ├── paddle_wrapper.py            # Wrapper del reconocimiento PaddleOCR
│       │   ├──  text_refiner.py             # Clasifica y refina textos OCR
│       │   ├── text_cleaner.py              # Filtra/limpia OCR por probabilidad
│       │   ├── text_corrector.py            # Corrige tokens y cantidades
│       │   ├──  fragmenter.py               # Fragmenta texto por clasificación semántica
│       │   └── data_finder.py               # Busca datos clave con WordFinder
│       └── vectorial_transformation/
│           ├── vectorizing_factory.py       # Factory de vectorización
│           ├── lineal_reconstructor.py      # Reconstruye líneas desde polígonos OCR
│           ├── matricial_cosine.py          # Agrupa líneas por similitud coseno/clustering 
│           ├── geometric_table_structurer.py# Asigna celdas a columnas por geometría
│           ├── math_max.py                  # Valida/reconstruye el dataframe
│           └── data_collector.py            # Encuentra la metadata global
├── data/
│   ├── __init__.py
│   ├── ladas_mexico.csv                     # Catálogo auxiliar de claves lada
│   └── models/
│       ├── paddle/
│       │   └── det/es/                      # Modelo PaddleOCR de detección
│       └── wf/                              # Modelos pickle de WordFinder
├── documentation/
│   ├── algoritmo_estructuración_tabular.sty # Estilo/documento auxiliar LaTeX
│   ├── algoritmo_matricial.py               # Prototipo/documentación del algoritmo matricial
│   ├── data_base_creation.sql               # DDL PostgreSQL objetivo
│   ├── funciones_actualizar_dict.py         # Script auxiliar histórico/no integrado
│   ├── metadatos.csv                        # Metadata de referencia
│   ├── requeriments.txt                     # Comandos Conda/Paddle usados como referencia
│   └── vida_datos.txt                       # Notas sobre ciclo de vida de datos
├── gui/
│   └── ui.py                                # GUI mínima PySide6 para lanzar main.py
├── input/                                   # Imágenes de entrada
├── input2/                                  # Imágenes de entrada
├── input3/                                  # Imágenes de entrada
├── output/                                  # Salidas generadas; se limpia al iniciar
├── services/
│   ├── config_service.py                    # Carga, valida y deriva configuración
│   ├── db_service.py                        # Conexión e inserción PostgreSQL
│   ├── output_service.py                    # Escritura de imágenes, JSON, CSV y tablas para debug visual
│   └── system_service.py                    # Limpieza, planificación y utilidades de sistema automatizadas
├── test/
│   ├── metrics.py                           # Métricas simples estadísticas del sistema
│   └── test_registros_compra.sql            # Consulta manual para validar registros
├── main.py                                  # Entrada CLI principal
├── perfectocr.txt                           # Log de ejecución; se sobrescribe al correr
└── README.md                                # Esta documentación
```

Directorios de IDE como `.idea/`, `.vscode/` y `.cursor/` existen localmente, pero no forman parte de la arquitectura del proyecto.

---

## Descripción de módulos y archivos

### Raíz

| Archivo | Descripción |
|---|---|
| `main.py` | Configura `PROJECT_ROOT`, variables de rendimiento (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `FLAGS_use_mkldnn`), logging, rutas de salida y ejecuta `app.main_builder.activate_main`. Limpia `output/` antes de procesar. |
| `perfectocr.txt` | Archivo de log de ejecución. Se abre en modo escritura y se sobrescribe al iniciar. |
| `.env` | Debe contener `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS` para PostgreSQL. No documentar ni exponer valores reales. |
| `.gitignore`, `.aiignore`, `.cursorignore` | Reglas de exclusión para Git y herramientas locales. |

### `app/`

| Archivo | Descripción |
|---|---|
| `main_builder.py` | Coordina la ejecución global: configuración, plan de imágenes, modelos, stagers, procesamiento secuencial e inserción en base de datos. |
| `models_manager.py` | Singleton thread-safe. Inicializa PaddleOCR con detección/reconocimiento según workers activos e inicializa `WordFinder` cuando `data_finder` está habilitado. |
| `process_builder.py` | Director de una imagen. Crea un `DataFormatter`, ejecuta stagers disponibles en orden y devuelve `(DataFrame, global_data)`. |

### `config/`

| Archivo | Descripción |
|---|---|
| `master_config.yaml` | Define entradas, workers activos, flags de depuración, modelos, parámetros por worker y utilidades globales. |
| `config_models.py` | Modelos Pydantic que validan la forma del YAML: sistema, pipeline, salidas, modelos, módulos, utilidades y depuración. |

### `core/domain/`

| Archivo | Descripción |
|---|---|
| `data_models.py` | Dataclasses: `WorkflowData`, `FullImage`, `Metadata`, `Polygons`, `Geometry`, `AllLines`, `StructuredData`, entre otras. |
| `data_formatter.py` | Fachada mutable del workflow. Crea el estado inicial, actualiza imagen completa, polígonos, OCR, líneas, tablas y datos finales. |

### `core/factory/`

| Archivo | Descripción |
|---|---|
| `abstract_factory.py` | Clase base genérica que registra nombres de workers y crea instancias desde una lista configurada. |
| `abstract_stager.py` | Contrato base para etapas con método `execute`. |
| `abstract_worker.py` | Contratos base por tipo de worker: preparación, preprocesamiento, OCR y vectorización. |
| `main_factory.py` | Crea y expone factories específicas por módulo. |

### `core/pipeline/`

| Archivo | Descripción |
|---|---|
| `stagers_factory.py` | Lee workers activos desde la configuración derivada y ensambla stagers. Evita `polygon_extractor` si no está `geometry_detector`. |
| `image_preparation_stager.py` | Ejecuta `process` sobre workers de preparación. |
| `preprocessing_stager.py` | Ejecuta `preprocess` sobre filtros de recortes si hay workers activos. |
| `ocr_stager.py` | Ejecuta `transcribe` sobre workers OCR. |
| `vectorization_stager.py` | Ejecuta `vectorize` sobre workers de reconstrucción, tabla y validación. |

### `core/utils/`

| Archivo | Descripción |
|---|---|
| `data_utils.py` | Constantes de codificación textual, frecuencias, features, encabezados y diccionarios auxiliares. |
| `image_utils.py` | Normalización de imágenes, binarización, recortes, contornos, filtros y validaciones de imagen. |
| `math_utils.py` | Encoders, similitud coseno, clustering DBSCAN, geometría de líneas, features numéricas y estadísticas. |
| `text_utils.py` | Normalización de texto, detección de RFC, unidades, cantidades, códigos, palabras clave y clasificación semántica. |
| `word_finder.py` | Carga modelo pickle y resuelve coincidencias de palabras clave con n-gramas y scoring híbrido. |
| `__init__.py` | Marca el paquete `utils`. |

### `core/workers/image_preparation/`

| Archivo | Worker | Descripción |
|---|---|---|
| `image_loader.py` | `image_loader` | Carga imagen, convierte/normaliza y crea el `WorkflowData` inicial. |
| `angle_corrector.py` | `angle_corrector` | Corrige inclinación usando Canny/Hough y rotación con recorte. |
| `ink_enhancer.py` | `ink_enhancement` | Mejora tinta y rellena/filtra regiones según métricas geométricas. |
| `geometry_detector.py` | `geometry_detector` | Usa motor de detección PaddleOCR para producir polígonos de texto. |
| `poly_gone.py` | `polygon_extractor` | Recorta regiones detectadas y guarda recortes en el workflow. |
| `moire.py` | No registrado actualmente | Implementa detección/filtro de patrones moiré; no aparece en `ImagePreparationFactory`. |
| `image_preparation_factory.py` | Factory | Mapea nombres YAML a clases de preparación. |

### `core/workers/preprocessing/`

| Archivo | Worker | Descripción |
|---|---|---|
| `restorer.py` | `restorer` | Restauración morfológica de recortes según área/kernel. |
| `sp.py` | `sp` | Analiza y corrige ruido sal y pimienta. |
| `gauss.py` | `gauss` | Decide y aplica suavizado gaussiano según varianza Laplaciana. |
| `clahe.py` | `clahe` | Aplica corrección de contraste adaptativa. |
| `sharp.py` | `sharp` | Aplica nitidez si el recorte está por debajo del umbral. |
| `preprocessing_factory.py` | Factory | Mapea nombres YAML a filtros de preprocesamiento. |

### `core/workers/ocr/`

| Archivo | Worker | Descripción |
|---|---|---|
| `paddle_wrapper.py` | `paddle_wrapper` | Ejecuta reconocimiento PaddleOCR sobre recortes y filtra por confianza mínima. |
| `text_refiner.py` | `text_refiner` | Orquesta limpieza, corrección, fragmentación y clasificación semántica. |
| `text_cleaner.py` | Interno del refinador | Limpia resultados OCR por probabilidad. |
| `text_corrector.py` | Interno del refinador | Corrige tokens, cantidades y errores frecuentes. |
| `fragmenter.py` | Interno del refinador | Divide texto por clases semánticas. |
| `data_finder.py` | `data_finder` | Usa `WordFinder` y reglas de texto para extraer campos clave. |
| `ocr_factory.py` | Factory | Mapea workers OCR y comparte subworkers del refinador si `num_passes > 0`. |

### `core/workers/vectorial_transformation/`

| Archivo | Worker | Descripción |
|---|---|---|
| `lineal_reconstructor.py` | `lineal` | Reconstruye líneas desde polígonos OCR y marca líneas tabulares. |
| `matricial_cosine.py` | `cos_sim` | Compara vectores, valida similitud y agrupa candidatos por clustering. |
| `geometric_table_structurer.py` | `table_structurer` | Construye estructura tabular usando encabezados, centroides y asignación geométrica. |
| `math_max.py` | `math_max` | Resuelve hipótesis numéricas, corrige celdas y valida totales/cantidades. |
| `data_collector.py` | `collector` | Normaliza la tabla final y genera metadata global para salida/DB. |
| `vectorizing_factory.py` | Factory | Mapea nombres YAML a workers de vectorización. |

### `services/`

| Archivo | Descripción |
|---|---|
| `config_service.py` | Carga YAML, valida con Pydantic, calcula workers activos, decide qué modelos cargar y arma configuración por módulo. |
| `system_service.py` | Limpia `output/` de forma controlada, elimina cachés Python, planifica imágenes a procesar y puede truncar tablas de PostgreSQL. |
| `output_service.py` | Guarda imágenes, contornos, JSON, CSV, tablas finales y valores de depuración. |
| `db_service.py` | Construye DSN desde `.env`, prueba conexión e inserta clientes, proveedores, registros y detalles en PostgreSQL. |

### `data/`

| Ruta | Descripción |
|---|---|
| `data/ladas_mexico.csv` | Catálogo auxiliar para identificación de teléfonos/ladas. |
| `data/models/paddle/det/es/` | Modelo PaddleOCR de detección presente en el árbol local. |
| `data/models/paddle/rec/es/` | Ruta esperada por `master_config.yaml` para reconocimiento; debe agregarse si no existe. |
| `data/models/wf/` | Modelos pickle usados por `WordFinder`. |

### `documentation/`, `gui/` y `test/`

| Archivo | Descripción |
|---|---|
| `documentation/data_base_creation.sql` | Esquema PostgreSQL objetivo: `clientes`, `proveedores`, `productos`, `registros_compra`, `detalles_compra`. |
| `documentation/requeriments.txt` | Referencia de instalación Conda/Paddle usada por el proyecto. |
| `documentation/algoritmo_matricial.py` | Prototipo o referencia técnica del algoritmo matricial. |
| `documentation/funciones_actualizar_dict.py` | Script auxiliar histórico; no forma parte del flujo principal. |
| `documentation/metadatos.csv` | Datos de referencia. |
| `documentation/vida_datos.txt` | Notas de ciclo de vida de datos. |
| `documentation/algoritmo_estructuración_tabular.sty` | Archivo de estilo/documentación LaTeX. |
| `gui/ui.py` | Interfaz PySide6 mínima con botón para lanzar `python main.py` y ver stdout/stderr. |
| `test/metrics.py` | Script de conteo de líneas por archivo. |
| `test/test_registros_compra.sql` | Consulta manual de validación contra `public.registros_compra`. |

---

## Configuración

La configuración principal vive en `config/master_config.yaml`.

Secciones principales:

| Sección | Uso |
|---|---|
| `system_config` | Directorios de entrada, nombres específicos de imágenes y extensiones válidas. |
| `pipeline_secuence` | Orden de workers por etapa. El nombre mantiene la ortografía actual del código. |
| `log_debug` | Flags de logging semántico, OCR, líneas y tablas. |
| `enabled_outputs` | Controla qué artefactos intermedios se escriben en `output/`. |
| `models_config` | Parámetros PaddleOCR, rutas de modelos y modelo WordFinder. |
| `modules` | Parámetros finos por worker. |
| `utils` | Rangos y umbrales compartidos. |

Variables de entorno para base de datos:

```env
DB_HOST=...
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASS=...
```

Variables de rendimiento configuradas en `main.py`:

```env
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
FLAGS_use_mkldnn=1
```

---

## Instalación

Referencia Conda usada por el proyecto:

```bash
conda create -n intel python=3.12 numpy=1.26.4 pip conda scipy pandas scikit-learn opencv=4.11.0 -c https://software.repos.intel.com/python/conda/
conda install paddlepaddle==2.6.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
conda install pydantic scikit-image python-dotenv psycopg2
```

Dependencias Python usadas por el código:

```text
paddleocr
paddlepaddle
pydantic
pyyaml
numpy
pandas
opencv-python / opencv
scikit-image
scikit-learn
psycopg2
python-dotenv
PySide6        # solo para gui/ui.py
```

Modelos requeridos para ejecución completa:

```text
data/models/paddle/det/es/      # detección PaddleOCR
data/models/paddle/rec/es/      # reconocimiento PaddleOCR
data/models/wf/wf_model.pkl     # WordFinder
```

---

## Uso

Ejecución CLI:

```bash
python main.py
```

Ejecución de GUI local:

```bash
python gui/ui.py
```

Flujo esperado:

1. `main.py` configura logging y rutas.
2. `services.system_service.clear_output_folders` limpia `output/`.
3. `ConfigService` valida `config/master_config.yaml`.
4. `count_and_plan` busca imágenes en `input/`, `input2/`, `input3/`.
5. `ModelsManager` carga modelos necesarios.
6. `ProcessingBuilder` procesa cada imagen.
7. Si `db_stage` está activo y PostgreSQL responde, se limpian tablas y se inserta el payload.

---

## Pipeline configurado

Según `config/master_config.yaml`, el orden activo es:

```yaml
imagepre_stage:
  - image_loader
  - angle_corrector
  - geometry_detector
  - polygon_extractor

preprocessing_stage: []

ocr_stage:
  - paddle_wrapper
  - text_refiner
  - data_finder

vector_stage:
  - lineal
  - cos_sim
  - table_structurer
  - math_max
  - collector

db_stage:
  - db_service
```

Resumen funcional:

| Etapa | Resultado |
|---|---|
| Preparación | Imagen cargada, enderezada, regiones detectadas y recortes creados. |
| Preprocesamiento | Actualmente inactivo; puede habilitar restauración, ruido, contraste y nitidez. |
| OCR | Texto reconocido, refinado y clasificado; campos clave detectados. |
| Vectorización | Líneas reconstruidas, tabla estructurada, cantidades validadas y DataFrame final. |
| DB | Inserción opcional en PostgreSQL si hay conexión y configuración completa. |

---

## Base de datos

El DDL de referencia está en `documentation/data_base_creation.sql`.

Tablas definidas:

| Tabla | Descripción |
|---|---|
| `clientes` | Maestro de clientes. |
| `proveedores` | Maestro de proveedores. |
| `productos` | Catálogo global de productos. |
| `registros_compra` | Encabezado de compra/documento. |
| `detalles_compra` | Líneas de detalle por registro. |

`services/db_service.py` inserta actualmente:

- `clientes`
- `proveedores`
- `registros_compra`
- `detalles_compra`

Antes de insertar, `services/system_service.clean_db` trunca las tablas del schema `public` si la conexión está disponible.

---

## Salidas y depuración

Las salidas dependen de `enabled_outputs` en `config/master_config.yaml`. Actualmente casi todas están desactivadas.

Tipos de salida soportados por `services/output_service.py`:

- Imágenes intermedias (`.png`, `.jpg`, etc.).
- JSON crudo o serializado.
- CSV de tablas.
- CSV maestro `tables_master.csv`.
- Tablas y features de depuración.
- Contornos y recortes por worker.

El archivo `perfectocr.txt` concentra logs de consola/archivo con formato:

```text
HH:MM:SS - archivo.py:línea - mensaje
```

---

## Notas de mantenimiento

- Solo `config/master_config.yaml` debe considerarse configuración principal; no hay `master_config.yaml` en la raíz.
- No existe `tests/`; el directorio real es `test/`.
- No existen `app/workflow_builder.py` ni `app/database_builder.py`; esas referencias pertenecían a una versión anterior del README.
- `services/system_service.py` funciona como servicio de sistema/caché aunque el comentario interno conserve un nombre antiguo.
- `core/workers/image_preparation/moire.py` existe, pero no está registrado en `ImagePreparationFactory`; en cambio, el worker `moire` configurado en preprocesamiento no tiene entrada en `PreprocessingFactory`.
- `README.md` está ignorado por `.gitignore` (`*.md`), por lo que los cambios al README pueden no aparecer en `git status` salvo que se fuerce su seguimiento.
