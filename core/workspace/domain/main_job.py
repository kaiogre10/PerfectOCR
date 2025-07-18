# PerfectOCR/core/workspace/domain/main_job
import time
import uuid  # Para generar identificadores únicos universales (UUID) para cada trabajo
from dataclasses import dataclass, field  # Para simplificar la definición de clases de datos
from datetime import datetime  # Para manejar fechas y horas
import numpy as np  # Para trabajar con arrays de imágenes

@dataclass  # Convierte la clase en una dataclass, añadiendo automáticamente métodos útiles
class ProcessingJob:
    source_uri: str   # La ruta del archivo original, URL, etc. (de dónde se toma la imagen)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Identificador único generado automáticamente para cada trabajo
    status: str = "PENDING"  # Estado del trabajo: PENDING, PROCESSING, COMPLETED, FAILED
    image_data: np.ndarray | None = None  # Imagen cargada en memoria (array de NumPy) o None si aún no se ha cargado
    workflow_stage: dict[str] = field(default_factory=str)  # Módulo en el está siendo procesado
    final_result: dict | None = None  # Resultado final del procesamiento (por ejemplo, texto extraído), o None si aún no está disponible
    error_message: str | None = None  # Mensaje de error si el trabajo falla, o None si no hay error
    created_at: datetime = field(default_factory=datetime.utcnow)  # Fecha y hora de creación del trabajo (en UTC)
    completed_at: datetime | None = None  # Fecha y hora de finalización, o None si aún no ha terminado
 