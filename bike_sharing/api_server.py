"""
API Server para bike sharing

Uso:
    uvicorn bike_sharing.api_server:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pathlib import Path
import numpy as np
from datetime import date
from loguru import logger
import logging

from bike_sharing.schemas_api import PredictionRequest, PredictionRequestRaw
from bike_sharing.api_utils import buscarModelo, build_full_pipeline

import pandas as pd

app = FastAPI(title="Bike Sharing API")

logging.basicConfig(
    level=logging.DEBUG,  # Cambia a DEBUG
    format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
)


@app.get("/")
async def root():
    """Endpoint raíz."""
    return {"message": "Hello world"}


@app.get('/health/')
def healthService():
    return {'status': 'Service up and running :D'}


@app.post('/predict/')
def predict(request: PredictionRequest):
    """
    Realiza predicciones usando dos modelos:
    - Uno para usuarios casuales
    - Otro para usuarios registrados
    El nombre base del modelo debe contener '_casual' o '_registered'
    """
    logger.info(f"💡 Nueva solicitud de predicción recibida: {request.model_name}")
    model_name = request.model_name.lower()  # Case-insensitive

    # Determinación del modelo base
    if '_casual' in model_name:
        model_name_casual = model_name
        model_name_registered = model_name.replace('_casual', '_registered')
    elif '_registered' in model_name:
        model_name_registered = model_name
        model_name_casual = model_name.replace('_registered', '_casual')
    else:
        logger.error("❌ Nombre del modelo inválido")
        raise HTTPException(status_code=400, detail="El nombre del modelo debe contener '_casual' o '_registered'")

    logger.debug(f"Modelos a cargar -> Casual: {model_name_casual}, Registered: {model_name_registered}")

    try:
        # 1. Cargar los modelos
        model_casual = buscarModelo(model_name_casual)
        model_registered = buscarModelo(model_name_registered)
        logger.info("📦 Modelos cargados exitosamente.")

        # 2. Obtener los features como array numpy
        try:
            feature_values = list(request.features.model_dump().values())
        except AttributeError:
            feature_values = list(request.features.dict().values())

        logger.debug(f"Features recibidos ({len(feature_values)}): {feature_values[:5]}...")

        X_test = np.array(feature_values).reshape(1, -1)

        # 3. Comprobar cantidad de features
        EXPECTED_FEATURES = 177
        if X_test.shape[1] != EXPECTED_FEATURES:
            error_msg = (
                f"Error de Feature Count: "
                f"Se esperaban {EXPECTED_FEATURES} características, pero se recibieron {X_test.shape[1]}."
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        # 4. Hacer predicciones
        logger.debug("Realizando predicciones...")
        y_pred_casual = np.expm1(model_casual.predict(X_test)[0])
        y_pred_registered = np.expm1(model_registered.predict(X_test)[0])
        total_count = y_pred_casual + y_pred_registered

        # Convertir a floats nativos antes de devolver la respuesta
        result = {
            "predicted_total_count": float(round(total_count, 2)),
            "casual": float(round(y_pred_casual, 2)),
            "registered": float(round(y_pred_registered, 2)),
        }

        logger.info(f"🔮 Predicciones generadas: {result}")
        return result

    except Exception as e:
        logger.exception(f"Error en la predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post('/predictraw/')
def predictRaw(request: PredictionRequestRaw):
    logger.info(f"🔮 Iniciando predicción con el modelo: {request.model_name}")

    # 1. Validar nombres de modelos
    if '_casual' in request.model_name.lower():
        model_name_casual = request.model_name
        model_name_registered = request.model_name.lower().replace('_casual', '_registered')
    elif '_registered' in request.model_name.lower():
        model_name_registered = request.model_name
        model_name_casual = request.model_name.lower().replace('_registered', '_casual')
    else:
        logger.error("❌ Error: El nombre del modelo no contiene los sufijos esperados.")
        raise HTTPException(status_code=400, detail="El nombre del modelo debe contener '_casual' o '_registered'.")

    try:
        # 2. Cargar modelos
        model_casual = buscarModelo(model_name_casual)
        logger.info(f"📦 Modelo '{model_name_casual}' cargado correctamente.")
        model_registered = buscarModelo(model_name_registered)
        logger.info(f"📦 Modelo '{model_name_registered}' cargado correctamente.")

        # 3. Preparar datos de entrada
        try:
            data_dict = request.features.model_dump()
        except AttributeError:
            data_dict = request.features.dict()

        df_input = pd.DataFrame([data_dict])
        logger.debug(f"🔍 DataFrame de entrada:\n{df_input}")
        
        # 4. After pipeline
        df_clean = build_full_pipeline(bike_sharing_df=df_input)
        logger.debug(f"🧼 Columnas finales para predicción ({len(df_clean.columns)} columnas): {df_clean.columns.tolist()}")
        logger.debug(f"📊 Primeras filas del DataFrame:\n{df_clean.head()}")

        # 5. Prepare input array        
        X_test = df_clean.to_numpy()
        logger.debug(f"📊 Dimensión de X_test: {X_test.shape}")
        logger.debug(f"📊 Modelo '{model_name_casual}' espera {model_casual.n_features_in_} features")

        # 6. Hacer predicciones
        logger.info("🤖 Generando predicciones...")
        y_pred_casual_log = model_casual.predict(X_test)[0]
        y_pred_registered_log = model_registered.predict(X_test)[0]

        # 7. Revertir log
        y_pred_casual = np.expm1(y_pred_casual_log)
        y_pred_registered = np.expm1(y_pred_registered_log)
        total_pred = max(0, y_pred_casual + y_pred_registered)

        response = {
            "model_requested": request.model_name,
            "prediction_strategy": "Double Prediction (Casual + Registered)",
            "predicted_casual_log": round(y_pred_casual_log, 4),
            "predicted_registered_log": round(y_pred_registered_log, 4),
            "predicted_casual_count": round(y_pred_casual, 2),
            "predicted_registered_count": round(y_pred_registered, 2),
            "predicted_total_count": round(total_pred, 2)
        }

        logger.info(f"✅ Predicción generada exitosamente: {response}")
        return response

    except Exception as e:
        logger.error("❌ Error en la predicción:", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en la predicción: {type(e).__name__}: {str(e)}"
        )