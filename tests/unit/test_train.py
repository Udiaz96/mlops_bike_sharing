import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from bike_sharing.modeling.train import (
    train_model,
    MODEL_REGISTRY
)


def test_model_registry_contains_expected_models():
    """Prueba que el registro de modelos contiene los modelos esperados."""
    expected_models = ['random_forest', 'elastic_net', 'svr', 'xgboost', 'lightgbm']

    for model_name in expected_models:
        assert model_name in MODEL_REGISTRY, f"Model {model_name} not found in registry"

    # Verificar que todos los valores son clases (callables)
    for model_name, model_class in MODEL_REGISTRY.items():
        assert callable(model_class), f"Model {model_name} is not callable"


def test_model_registry_instantiation():
    """Prueba que los modelos del registro pueden ser instanciados correctamente."""
    # Random Forest
    rf_model = MODEL_REGISTRY['random_forest'](n_estimators=10, random_state=42)
    assert rf_model is not None
    assert hasattr(rf_model, 'fit')
    assert hasattr(rf_model, 'predict')

    # ElasticNet
    en_model = MODEL_REGISTRY['elastic_net'](alpha=0.1, random_state=42)
    assert en_model is not None
    assert hasattr(en_model, 'fit')
    assert hasattr(en_model, 'predict')

    # XGBoost
    xgb_model = MODEL_REGISTRY['xgboost'](n_estimators=10, random_state=42)
    assert xgb_model is not None
    assert hasattr(xgb_model, 'fit')
    assert hasattr(xgb_model, 'predict')


def test_metric_calculations():
    """Prueba cálculos de métricas (R2, RMSE, MAE) con datos sintéticos."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Datos sintéticos en escala logarítmica
    y_casual_test_log = np.array([2.0, 3.0, 4.0, 5.0])
    y_registered_test_log = np.array([3.0, 4.0, 5.0, 6.0])

    y_pred_casual_log = np.array([2.1, 2.9, 4.2, 4.8])
    y_pred_registered_log = np.array([3.1, 3.9, 5.1, 5.9])

    # Convertir de log a escala original
    y_pred_casual_original = np.expm1(y_pred_casual_log)
    y_pred_registered_original = np.expm1(y_pred_registered_log)
    y_pred_combined_original = y_pred_casual_original + y_pred_registered_original

    y_test_combined_original = np.expm1(y_casual_test_log) + np.expm1(y_registered_test_log)

    # Calcular métricas
    r2 = r2_score(y_test_combined_original, y_pred_combined_original)
    rmse = np.sqrt(mean_squared_error(y_test_combined_original, y_pred_combined_original))
    mae = mean_absolute_error(y_test_combined_original, y_pred_combined_original)

    # Verificar que las métricas están en rangos razonables
    assert 0 <= r2 <= 1, f"R2 should be between 0 and 1, got {r2}"
    assert rmse >= 0, f"RMSE should be non-negative, got {rmse}"
    assert mae >= 0, f"MAE should be non-negative, got {mae}"

    # Verificar que MAE <= RMSE (siempre debe ser cierto)
    assert mae <= rmse, f"MAE ({mae}) should be <= RMSE ({rmse})"


def test_negative_predictions_handling():
    """Prueba que las predicciones negativas son manejadas correctamente."""
    # Simular predicciones que podrían resultar en valores negativos
    y_pred_casual_log = np.array([0.5, 1.0, -0.5, 2.0])  # Uno negativo
    y_pred_registered_log = np.array([1.0, 2.0, 0.0, 3.0])

    # Convertir de log a escala original
    y_pred_casual_original = np.expm1(y_pred_casual_log)
    y_pred_registered_original = np.expm1(y_pred_registered_log)
    y_pred_combined_original = y_pred_casual_original + y_pred_registered_original

    # Aplicar la corrección de valores negativos (como en train_model)
    y_pred_combined_original[y_pred_combined_original < 0] = 0

    # Verificar que no hay valores negativos
    assert np.all(y_pred_combined_original >= 0), "All predictions should be non-negative"


@patch('bike_sharing.modeling.train.mlflow')
def test_train_model_logs_to_mlflow(mock_mlflow):
    """Prueba que train_model registra correctamente los parámetros y métricas en MLFlow."""
    # Mock MLFlow run context
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    # Crear datos sintéticos pequeños para entrenamiento rápido
    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'feature3': np.random.rand(50)
    })
    X_test = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20),
        'feature3': np.random.rand(20)
    })

    y_casual_train = np.random.rand(50) * 2 + 2  # Log scale values
    y_casual_test = np.random.rand(20) * 2 + 2
    y_registered_train = np.random.rand(50) * 2 + 3
    y_registered_test = np.random.rand(20) * 2 + 3

    # Parámetros del modelo
    params = {'n_estimators': 10, 'max_depth': 3, 'random_state': 42}

    # Ejecutar train_model con RandomForest
    from sklearn.ensemble import RandomForestRegressor

    rmse, run_id, model_name, _, _, _, _ = train_model(
        model_type=RandomForestRegressor,
        model_name='RandomForest',
        params=params,
        X_train=X_train,
        X_test=X_test,
        y_casual_train=y_casual_train,
        y_casual_test=y_casual_test,
        y_registered_train=y_registered_train,
        y_registered_test=y_registered_test
    )

    # Verificar que MLFlow fue llamado correctamente
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called_once_with(params)

    # Verificar que se registraron las métricas
    assert mock_mlflow.log_metric.call_count == 3  # R2, RMSE, MAE

    # Verificar que se registraron los modelos
    assert mock_mlflow.sklearn.log_model.call_count == 2  # Casual y Registered

    # Verificar valores de retorno
    assert isinstance(rmse, float)
    assert rmse >= 0
    assert run_id == "test_run_123"
    assert model_name == 'RandomForest'
