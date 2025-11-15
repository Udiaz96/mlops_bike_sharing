import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from bike_sharing.dataset import (
    map_season,
    get_cols_to_drop,
    cast_values,
    clean_year,
    impute_bikes_total_count
)


def test_map_season():
    """Prueba mapeo de estaciones con varias fechas."""
    # Winter (Dec 21 - Mar 20)
    winter_date = pd.Timestamp('2011-01-15')
    assert map_season(winter_date) == 1.0

    # Spring (Mar 21 - Jun 20)
    spring_date = pd.Timestamp('2011-04-15')
    assert map_season(spring_date) == 2.0

    # Summer (Jun 21 - Sep 22)
    summer_date = pd.Timestamp('2011-07-15')
    assert map_season(summer_date) == 3.0

    # Fall (Sep 23 - Dec 20)
    fall_date = pd.Timestamp('2011-10-15')
    assert map_season(fall_date) == 4.0

    # Test with NaN
    assert pd.isna(map_season(pd.NaT))


def test_get_cols_to_drop():
    """Prueba detección de diferencias de columnas entre dataframes original y modificado."""
    original_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })

    modified_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9],
        'col4': [10, 11, 12],  # Extra column
        'col5': [13, 14, 15]   # Extra column
    })

    cols_to_drop = get_cols_to_drop(original_df, modified_df)

    assert len(cols_to_drop) == 2
    assert 'col4' in cols_to_drop
    assert 'col5' in cols_to_drop


def test_cast_values():
    """Prueba de conversión de tipos de columnas en un DataFrame."""
    df = pd.DataFrame({
        'instant': ['1', '2', '3'],
        'season': ['1', '2', '3'],
        'temp': ['0.5', '0.6', '0.7'],
        'hum': ['50.5', '60.2', '70.8'],
        'cnt': ['100', '200', '300']
    })

    cast_values(df)

    # Check integer columns
    assert df['instant'].dtype == 'Int64'
    assert df['season'].dtype == 'Int64'
    assert df['cnt'].dtype == 'Int64'

    # Check float columns
    assert df['temp'].dtype == 'float64'
    assert df['hum'].dtype == 'float64'

    # Check values are correctly converted
    assert df['instant'].iloc[0] == 1
    assert df['temp'].iloc[0] == 0.5


def test_clean_year():
    """Test year cleaning and validation."""
    df = pd.DataFrame({
        'yr': [0.0, 1.0, np.nan, 5.0, -1.0],
        'dteday': pd.to_datetime([
            '2011-01-01',
            '2012-01-01',
            '2011-06-15',
            '2012-08-10',
            '2011-12-31'
        ])
    })

    result = clean_year(df, 'yr', 'dteday')

    # Check valid years remain unchanged
    assert result['yr'].iloc[0] == 0.0  # 2011
    assert result['yr'].iloc[1] == 1.0  # 2012

    # Check NaN is filled from date
    assert result['yr'].iloc[2] == 0.0  # 2011

    # Check invalid values are corrected
    assert result['yr'].iloc[3] == 1.0  # Should be 2012
    assert result['yr'].iloc[4] == 0.0  # Should be 2011


def test_impute_bikes_total_count():
    """Prueba de validación e imputación de conteo total de bicicletas."""
    df = pd.DataFrame({
        'cnt': [100, 230, np.nan, 150, 300],
        'registered': [80, 180, 60, 200, 250],
        'casual': [20, 50, 40, 50, 50]
    })

    result = impute_bikes_total_count(df)

    assert result['cnt'].iloc[2] == 100  # 60 + 40
