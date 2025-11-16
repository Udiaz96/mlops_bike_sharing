from pathlib import Path
import yaml
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from loguru import logger

from bike_sharing.dataset import build_pipeline as dataset_pipeline
from bike_sharing.features import build_pipeline as features_pipeline

from bike_sharing.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

BASE_DIR = Path(__file__).resolve().parent  # directorio donde está api_utils.py
models_dir = BASE_DIR / "../models"
models_dir = models_dir.resolve()
modelsLocation = models_dir

input_path: Path = PROCESSED_DATA_DIR / "bike_sharing_cleaned.csv"
output_path: Path = PROCESSED_DATA_DIR / "bike_sharing_transformed.csv"
original_path: Path = RAW_DATA_DIR / "bike_sharing_original.csv"


def buscarModelo(modelo: str):
    """
    Carga un modelo .pkl desde el directorio models usando el nombre base.
    
    Args:
        modelo (str): Nombre base del modelo (sin extensión), por ejemplo 'RandomForestRegressor_casual'.
    
    Returns:
        Carga y devuelve el modelo con joblib.
    """
    # Construir la ruta al archivo .pkl
    model_path = models_dir / f"{modelo}.pkl"

    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Modelo '{modelo}' cargado exitosamente desde: {model_path}")
        return model
    else:
        raise FileNotFoundError(f"No se encontró el archivo: {model_path}")


def build_full_pipeline(bike_sharing_df: pd.DataFrame):

    # Log para inspeccionar original_path
    logger.debug(f"📄 'original_path' recibido: {original_path} (tipo: {type(original_path)})")

    try:
        original_bike_sharing_df = pd.read_csv(original_path)
        logger.debug(f"📊 Archivo CSV cargado desde: {original_path}")
    except Exception as e:
        logger.error(f"❌ Error al cargar archivo CSV: {e}")
        raise e

    # Build pipeline
    logger.info("\nBuilding pipeline...")

    full_pipeline = Pipeline([
        ('dataset', dataset_pipeline(original_bike_sharing_df))
        #('features', features_pipeline())
    ])

    logger.debug(f"🧱 Pipeline creado: {full_pipeline}")

    # Run the pipeline
    logger.info("\nEjecutando el pipeline...")
    logger.info("-" * 80)

    # 1. Ejecutar dataset pipeline
    try:
        logger.info("🔄 Ejecutando etapa 'dataset'...")
        bike_sharing_df = full_pipeline.named_steps['dataset'].fit_transform(bike_sharing_df)
        logger.debug(f"📄 Resultado parcial después de 'dataset': shape={bike_sharing_df.shape}")
    except Exception as e:
        logger.error(f"❌ Error en etapa 'dataset': {e}", exc_info=True)
        raise e

    # 2. Ejecutar features pipeline paso a paso
    try:
        logger.info("🎯 Ejecutando etapa 'features' paso a paso...")
        features_steps = full_pipeline.named_steps['features'].named_steps

        for name, step in features_steps.items():
            logger.debug(f"🔧 Ejecutando transformador '{name}'...")
            try:
                bike_sharing_df = step.transform(bike_sharing_df)
                logger.debug(f"✓ '{name}' ejecutado correctamente. Shape = {bike_sharing_df.shape}")
            except Exception as e:
                logger.error(f"❌ Error en transformador '{name}': {e}", exc_info=True)
                raise e

    except Exception as e:
        logger.error("❌ Error en la etapa 'features'", exc_info=True)
        raise e

    logger.info("-" * 80)
    logger.info(f"📏 Dimensiones finales del dataset: {bike_sharing_df.shape}")
    logger.info(f"📄 Columnas finales: {list(bike_sharing_df.columns)}")
    logger.success("=" * 80)
    logger.success("🚀 Pipeline completado exitosamente!")
    logger.success("=" * 80)

    return bike_sharing_df
