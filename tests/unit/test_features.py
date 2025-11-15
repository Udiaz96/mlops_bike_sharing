import pytest
import pandas as pd
import numpy as np

from bike_sharing.features import (
    LogTransformer,
    CyclicTransformer,
    OHETransformer,
    CyclicScalerTransformer,
    ColumnDropperTransformer
)


def test_log_transformer():
    """Prueba transformación logarítmica de columnas numéricas."""
    df = pd.DataFrame({
        'cnt': [100, 200, 300],
        'casual': [20, 30, 40],
        'registered': [80, 170, 260],
        'temp': [0.5, 0.6, 0.7],
        'atemp': [0.48, 0.58, 0.68],
        'hum': [50.0, 60.0, 70.0],
        'windspeed': [10.0, 15.0, 20.0]
    })

    transformer = LogTransformer()
    result = transformer.fit_transform(df)

    # Verificar que las columnas transformadas existen
    assert 'cnt_log' in result.columns
    assert 'casual_log' in result.columns
    assert 'registered_log' in result.columns
    assert 'temp_log' in result.columns

    # Verificar valores de transformación logarítmica
    assert np.isclose(result['cnt_log'].iloc[0], np.log1p(100))
    assert np.isclose(result['casual_log'].iloc[1], np.log1p(30))

    # Verificar que las columnas originales aún existen
    assert 'cnt' in result.columns
    assert 'casual' in result.columns


def test_cyclic_transformer():
    """Prueba transformación cíclica de columnas temporales."""
    df = pd.DataFrame({
        'mnth': [1, 6, 12],
        'hr': [0, 12, 23],
        'weekday': [0, 3, 6]
    })

    transformer = CyclicTransformer()
    result = transformer.fit_transform(df)

    # Verificar que las columnas cíclicas sin/cos fueron creadas
    assert 'mnth_sin' in result.columns
    assert 'mnth_cos' in result.columns
    assert 'hr_sin' in result.columns
    assert 'hr_cos' in result.columns
    assert 'weekday_sin' in result.columns
    assert 'weekday_cos' in result.columns

    # Verificar que las columnas originales fueron eliminadas
    assert 'mnth' not in result.columns
    assert 'hr' not in result.columns
    assert 'weekday' not in result.columns

    # Verificar valores de transformación cíclica para hr=0 (debería ser sin=0, cos=1)
    assert np.isclose(result['hr_sin'].iloc[0], 0, atol=1e-10)
    assert np.isclose(result['hr_cos'].iloc[0], 1, atol=1e-10)

    # Verificar valores para hr=12 (debería ser sin=0, cos=-1 aproximadamente)
    assert np.isclose(result['hr_sin'].iloc[1], 0, atol=1e-10)
    assert np.isclose(result['hr_cos'].iloc[1], -1, atol=1e-10)


def test_ohe_transformer():
    """Prueba codificación one-hot de columnas categóricas."""
    df = pd.DataFrame({
        'season': [1, 2, 3, 4],
        'yr': [0, 1, 0, 1],
        'weathersit': [1, 2, 1, 3],
        'holiday': [0, 1, 0, 0],
        'workingday': [1, 0, 1, 1]
    })

    transformer = OHETransformer()
    result = transformer.fit_transform(df)

    # Verificar que las columnas originales fueron eliminadas
    assert 'season' not in result.columns
    assert 'yr' not in result.columns

    # Verificar que las columnas one-hot fueron creadas (drop_first=True)
    assert 'season_2' in result.columns
    assert 'season_3' in result.columns
    assert 'season_4' in result.columns
    assert 'season_1' not in result.columns  # Primera categoría eliminada

    assert 'yr_1' in result.columns
    assert 'yr_0' not in result.columns  # Primera categoría eliminada

    # Verificar valores de codificación
    assert result['season_2'].iloc[1] == 1
    assert result['season_2'].iloc[0] == 0
    assert result['yr_1'].iloc[1] == 1
    assert result['yr_1'].iloc[0] == 0


def test_cyclic_scaler_transformer():
    """Prueba escalado de columnas cíclicas."""
    df = pd.DataFrame({
        'mnth_sin': [0.0, 0.5, 1.0],
        'mnth_cos': [1.0, 0.866, 0.0],
        'hr_sin': [0.0, 0.707, 1.0],
        'hr_cos': [1.0, 0.707, 0.0],
        'other_col': [10, 20, 30]  # Columna no cíclica
    })

    transformer = CyclicScalerTransformer()
    transformer.fit(df)
    result = transformer.transform(df)

    # Verificar que las columnas cíclicas fueron escaladas
    assert 'mnth_sin' in result.columns
    assert 'mnth_cos' in result.columns
    assert 'hr_sin' in result.columns
    assert 'hr_cos' in result.columns

    # Verificar que la columna no cíclica no fue modificada
    assert result['other_col'].iloc[0] == 10
    assert result['other_col'].iloc[1] == 20

    # Verificar que las columnas cíclicas tienen media cercana a 0 y desviación estándar cercana a 1
    cyclic_cols = ['mnth_sin', 'mnth_cos', 'hr_sin', 'hr_cos']
    for col in cyclic_cols:
        assert np.isclose(result[col].mean(), 0, atol=1e-10)


def test_column_dropper_transformer():
    """Prueba eliminación de columnas especificadas."""
    df = pd.DataFrame({
        'cnt': [100, 200, 300],
        'casual': [20, 30, 40],
        'registered': [80, 170, 260],
        'dteday': pd.to_datetime(['2011-01-01', '2011-01-02', '2011-01-03']),
        'cnt_log': [4.6, 5.3, 5.7],
        'other_col': [1, 2, 3]
    })

    transformer = ColumnDropperTransformer()
    result = transformer.fit_transform(df)

    # Verificar que las columnas especificadas fueron eliminadas
    assert 'cnt' not in result.columns
    assert 'casual' not in result.columns
    assert 'registered' not in result.columns
    assert 'dteday' not in result.columns

    # Verificar que las columnas no especificadas permanecen
    assert 'cnt_log' in result.columns
    assert 'other_col' in result.columns

    # Verificar que los valores de las columnas restantes no cambiaron
    assert result['cnt_log'].iloc[0] == 4.6
    assert result['other_col'].iloc[1] == 2
