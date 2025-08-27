import os
import logging
from core.workers.vectorial_transformation.word_finder import WordFinder

logger = logging.getLogger(__name__)

# resolver rutas desde el root del proyecto (no hardcodear absol.)
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
MODEL = os.path.join(ROOT, "data", "models", "word_finder", "word_finder_model.pkl")
MODEL_WEIGHTED = os.path.join(ROOT, "data", "models", "word_finder", "word_finder_model_weighted.pkl")

# Crear una instancia global al iniciar el worker (reusar por petición)
try:
    WF = WordFinder(MODEL, weighted_model_path=MODEL_WEIGHTED, default="weighted")
    logger.info("WordFinder cargado. Modelos disponibles: %s", WF.available_models())
except Exception as e:
    logger.exception("No se pudo inicializar WordFinder: %s", e)
    WF = None

def process_text_block(text: str):
    """Llamar desde el flujo de procesamiento por cada bloque de texto OCR."""
    if WF is None:
        logger.error("WordFinder no disponible")
        return []
    # find_keywords acepta string o lista de strings
    try:
        results = WF.find_keywords(text)
        return results  # [] si no hay coincidencias por debajo del umbral
    except Exception:
        logger.exception("Error en búsqueda; devolviendo []")
        return []