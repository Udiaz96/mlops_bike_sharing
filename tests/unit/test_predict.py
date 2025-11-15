import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from bike_sharing.modeling.predict import (
    predict,
    evaluate_predictions
)


def test_evaluate_predictions_with_cnt_column():
    """Prueba evaluación de predicciones cuando existe columna 'cnt'."""
    # Datos reales
    df_actual = pd.DataFrame({
        'cnt': [100, 200, 150, 250, 180]
    })

    # Predicciones
    df_predicted = pd.DataFrame({
        'predicted_count': [105, 195, 155, 240, 185]
    })

    # Ejecutar evaluación (sin verificar el output de logger, solo que no falle)
    with patch('bike_sharing.modeling.predict.logger'):
        evaluate_predictions(df_actual, df_predicted)


def test_evaluate_predictions_with_cnt_log_column():
    """Prueba evaluación de predicciones cuando existe columna 'cnt_log'."""
    # Datos reales en escala logarítmica
    actual_counts = np.array([100, 200, 150, 250, 180])
    df_actual = pd.DataFrame({
        'cnt_log': np.log1p(actual_counts)
    })

    # Predicciones en escala original
    df_predicted = pd.DataFrame({
        'predicted_count': [105, 195, 155, 240, 185]
    })

    # Ejecutar evaluación
    with patch('bike_sharing.modeling.predict.logger'):
        evaluate_predictions(df_actual, df_predicted)


def test_evaluate_predictions_missing_target_column():
    """Prueba que evaluate_predictions maneja correctamente columna de target faltante."""
    # DataFrame sin 'cnt' ni 'cnt_log'
    df_actual = pd.DataFrame({
        'other_column': [1, 2, 3, 4, 5]
    })

    df_predicted = pd.DataFrame({
        'predicted_count': [100, 200, 150, 250, 180]
    })

    # Debe loggear un error y retornar sin fallar
    with patch('bike_sharing.modeling.predict.logger') as mock_logger:
        result = evaluate_predictions(df_actual, df_predicted)

        # Verificar que se llamó a logger.error
        mock_logger.error.assert_called_once()
        assert result is None


def test_evaluate_predictions_missing_predicted_column():
    """Prueba que evaluate_predictions maneja correctamente columna de predicción faltante."""
    df_actual = pd.DataFrame({
        'cnt': [100, 200, 150, 250, 180]
    })

    # DataFrame sin 'predicted_count'
    df_predicted = pd.DataFrame({
        'other_column': [1, 2, 3, 4, 5]
    })

    # Debe loggear un error y retornar sin fallar
    with patch('bike_sharing.modeling.predict.logger') as mock_logger:
        result = evaluate_predictions(df_actual, df_predicted)

        # Verificar que se llamó a logger.error
        mock_logger.error.assert_called_once()
        assert result is None


def test_negative_predictions_handling_in_evaluation():
    """Prueba que valores negativos son manejados correctamente en evaluación."""
    # Incluir algunos valores que podrían ser negativos
    df_actual = pd.DataFrame({
        'cnt': [100, 200, 150, 250, 180]
    })

    # Predicciones con un valor negativo (simulando un error del modelo)
    df_predicted = pd.DataFrame({
        'predicted_count': [105, -50, 155, 240, 185]  # Uno negativo
    })

    with patch('bike_sharing.modeling.predict.logger'):
        # No debe fallar, debe corregir valores negativos a 0
        evaluate_predictions(df_actual, df_predicted)


@patch('bike_sharing.modeling.predict.mlflow.pyfunc.load_model')
def test_predict_function_output_format(mock_load_model):
    """Prueba que la función predict genera el formato correcto de salida."""
    # Crear mocks de los modelos
    mock_model_casual = MagicMock()
    mock_model_registered = MagicMock()

    # Configurar predicciones en escala logarítmica
    mock_model_casual.predict.return_value = np.array([2.0, 3.0, 4.0])  # log scale
    mock_model_registered.predict.return_value = np.array([3.0, 4.0, 5.0])  # log scale

    # Configurar load_model para devolver los mocks apropiados
    def load_model_side_effect(uri):
        if 'Casual' in uri:
            return mock_model_casual
        elif 'Registered' in uri:
            return mock_model_registered

    mock_load_model.side_effect = load_model_side_effect

    # Crear datos de entrada
    features_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9]
    })

    predictions_path = Path("test_predictions.csv")

    # Ejecutar predict
    with patch('bike_sharing.modeling.predict.logger'):
        results_df = predict(features_df, predictions_path)

    # Verificar estructura del DataFrame de resultados
    assert isinstance(results_df, pd.DataFrame)
    assert 'predicted_casual' in results_df.columns
    assert 'predicted_registered' in results_df.columns
    assert 'predicted_count' in results_df.columns

    # Verificar número de filas
    assert len(results_df) == 3

    # Verificar tipos de datos (deben ser enteros)
    assert results_df['predicted_casual'].dtype == int
    assert results_df['predicted_registered'].dtype == int
    assert results_df['predicted_count'].dtype == int

    # Verificar que predicted_count = predicted_casual + predicted_registered
    for idx in range(len(results_df)):
        expected_count = results_df['predicted_casual'].iloc[idx] + results_df['predicted_registered'].iloc[idx]
        assert results_df['predicted_count'].iloc[idx] == expected_count

    # Verificar que no hay valores negativos
    assert (results_df['predicted_casual'] >= 0).all()
    assert (results_df['predicted_registered'] >= 0).all()
    assert (results_df['predicted_count'] >= 0).all()


@patch('bike_sharing.modeling.predict.mlflow.pyfunc.load_model')
def test_predict_with_negative_log_predictions(mock_load_model):
    """Prueba que predict maneja correctamente predicciones logarítmicas negativas."""
    # Crear mocks de los modelos
    mock_model_casual = MagicMock()
    mock_model_registered = MagicMock()

    # Predicciones que podrían resultar en valores negativos después de expm1
    mock_model_casual.predict.return_value = np.array([-0.5, 2.0, 3.0])  # Primera negativa
    mock_model_registered.predict.return_value = np.array([3.0, -1.0, 4.0])  # Segunda negativa

    # Configurar load_model
    def load_model_side_effect(uri):
        if 'Casual' in uri:
            return mock_model_casual
        elif 'Registered' in uri:
            return mock_model_registered

    mock_load_model.side_effect = load_model_side_effect

    # Crear datos de entrada
    features_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })

    predictions_path = Path("test_predictions.csv")

    # Ejecutar predict
    with patch('bike_sharing.modeling.predict.logger'):
        results_df = predict(features_df, predictions_path)

    # Verificar que todos los valores son no negativos
    assert (results_df['predicted_casual'] >= 0).all()
    assert (results_df['predicted_count'] >= 0).all()


def test_metric_calculations_accuracy():
    """Prueba precisión de cálculos de métricas con valores conocidos."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Valores reales y predichos
    y_actual = np.array([100, 200, 150, 250, 180])
    y_predicted = np.array([105, 195, 155, 240, 185])

    # Calcular métricas manualmente
    rmse_expected = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mae_expected = mean_absolute_error(y_actual, y_predicted)
    r2_expected = r2_score(y_actual, y_predicted)

    # Verificar que las métricas están en rangos esperados
    assert rmse_expected > 0
    assert mae_expected > 0
    assert 0 <= r2_expected <= 1

    # Verificar valores específicos (aproximados)
    assert np.isclose(mae_expected, 5.6, atol=0.5)
    assert np.isclose(rmse_expected, 6.5, atol=1.0)
    assert r2_expected > 0.95  # Predicciones muy buenas

    # Ahora probar con evaluate_predictions
    df_actual = pd.DataFrame({'cnt': y_actual})
    df_predicted = pd.DataFrame({'predicted_count': y_predicted})

    with patch('bike_sharing.modeling.predict.logger'):
        # Debe ejecutarse sin errores
        evaluate_predictions(df_actual, df_predicted)
