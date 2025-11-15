"""
Bike Sharing Pipeline de Limpieza de Datos

Este modulo implementa un pipeline de limpieza de datos utilizando scikit-learn


Uso:
    python -m bike_sharing.dataset
"""

from pathlib import Path

import calendar
import holidays
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import typer

from bike_sharing.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


# ============================================================================
# SCIKIT-LEARN TRANSFORMERS
# ============================================================================

class TypeCastingTransformer(BaseEstimator, TransformerMixin):
    """Transformer para castear las columnas a los tipos de dato correcto."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cast_values(X)
        logger.info("✓ Casteado de tipos completado")
        return X


class ColumnManagementTransformer(BaseEstimator, TransformerMixin):
    """Transformer para eliminar columnas no necesarias."""

    def __init__(self, original_df):
        self.original_df = original_df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols_to_drop = get_cols_to_drop(self.original_df, X)
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            logger.info(f"✓ Dropped columns: {cols_to_drop}")
        return X


class DateTimeCleaningTransformer(BaseEstimator, TransformerMixin):
    """Transformer para limpiar columnas de fecha y tiempo."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = convert_date_format(X, 'dteday')
        X = clean_hour(X, 'hr', 'dteday')
        X = impute_date(X, 'dteday', 'hr')
        logger.info("✓ Limpieza de DateTime completada")
        return X


class FeatureCleaningTransformer(BaseEstimator, TransformerMixin):
    """Transformer para limpiar caracteristicas."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = clean_year(X, 'yr', 'dteday')
        X = clean_month(X, 'mnth', 'dteday')
        X = clean_weekday(X, 'weekday', 'dteday')
        X = clean_holiday(X, 'holiday', 'dteday')
        X = clean_workday(X, 'workingday', 'weekday', 'holiday')
        X = clean_season(X, 'season', 'dteday')
        logger.info("✓ Limpieza de caracteristicas completada")
        return X


class InstantImputationTransformer(BaseEstimator, TransformerMixin):
    """Transformer para imputar la columna 'instant'."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = impute_instant(X, 'instant', 'dteday', 'hr')
        logger.info("✓ Imputación de 'instant' completada")
        return X


class DuplicateHandlingTransformer(BaseEstimator, TransformerMixin):
    """Transformer para manejar registros duplicados."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = crosscheck_duplicates(X)
        logger.info("✓ Manejo de duplicados completado")
        return X


class WeatherImputationTransformer(BaseEstimator, TransformerMixin):
    """Transformer para imputar detalles meteorológicos."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = impute_weather_details(X)
        logger.info("✓ Imputación de detalles meteorológicos completada")
        return X


class CountValidationTransformer(BaseEstimator, TransformerMixin):
    """Transformer para validar e imputar conteos de bicicletas."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = impute_bikes_total_count(X)
        logger.info("✓ Validación de conteos completada")
        return X


class FinalCleanupTransformer(BaseEstimator, TransformerMixin):
    """Transformer para limpieza final del dataset."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = rebuild_instant(X, 'instant')
        logger.info("✓ Limpieza final completada")
        return X


# ============================================================================
# TYPER APP
# ============================================================================

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "bike_sharing_modified.csv",
    original_path: Path = RAW_DATA_DIR / "bike_sharing_original.csv",
    output_path: Path = PROCESSED_DATA_DIR / "bike_sharing_cleaned.csv",
):
    """
    Entrada principal para ejecutar el pipeline de limpieza de datos del dataset de bike sharing.


    Args:
        input_path: Path al dataset modificado de bike sharing
        original_path: Path al dataset original de bike sharing
        output_path: Path al dataset limpio de bike sharing
    """
    logger.info("="*80)
    logger.info("🚴 Bike Sharing Pipeline de Limpieza de Datos")
    logger.info("="*80)

    # Load datasets
    logger.info("Cargando los datasets...")
    original_bike_sharing_df = pd.read_csv(original_path)
    bike_sharing_df = pd.read_csv(input_path)

    # Show differences between original and modified
    show_diff(original_bike_sharing_df, bike_sharing_df)

    # Build the scikit-learn pipeline
    logger.info("\nBuilding pipeline...")
    pipeline = pipeline(original_bike_sharing_df)

    # Run the pipeline
    logger.info("\nEjecutando el pipeline...")
    logger.info("-"*80)
    bike_sharing_df = pipeline.fit_transform(bike_sharing_df)
    logger.info("-"*80)

    # Save the cleaned dataset
    logger.info(f'\n💾 Guardando el dataset a: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bike_sharing_df.to_csv(output_path, index=False)

    logger.success("\n" + "="*80)
    logger.success("✅ Pipeline completedo exitosamente!")
    logger.success(f"📊 Dimensiones finales del dataset: {bike_sharing_df.shape}")
    logger.success("="*80)


def pipeline(original_bike_sharing_df=None):
    """
    Construye y retorna el pipeline de limpieza de datos.
    return: Pipeline
    """
    pipeline = Pipeline([
        ('type_casting', TypeCastingTransformer()),
        ('column_management', ColumnManagementTransformer(original_bike_sharing_df)),
        ('datetime_cleaning', DateTimeCleaningTransformer()),
        ('feature_cleaning', FeatureCleaningTransformer()),
        ('instant_imputation', InstantImputationTransformer()),
        ('duplicate_handling', DuplicateHandlingTransformer()),
        ('weather_imputation', WeatherImputationTransformer()),
        ('count_validation', CountValidationTransformer()),
        ('final_cleanup', FinalCleanupTransformer()),
    ])

    return pipeline


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cast_values(df):
    """
    Castea las columnas del DataFrame a los tipos de datos correctos.

    param df: pd.DataFrame
    """
    column_types_map = {
        'instant': 'int',
        'season': 'int',
        'yr': 'int',
        'mnth': 'int',
        'hr': 'int',
        'holiday': 'int',
        'weekday': 'int',
        'workingday': 'int',
        'weathersit': 'int',
        'temp': 'float',
        'atemp': 'float',
        'hum': 'float',
        'windspeed': 'float',
        'casual': 'int',
        'registered': 'int',
        'cnt': 'int',
        'mixed_type_col': 'int'
    }

    cast_map = {
        'int': lambda df, col: pd.to_numeric(df[col], errors='coerce').astype('Int64'),
        'float': lambda df, col: pd.to_numeric(df[col], errors='coerce').astype('float64')
    }

    for col_name, target_dtype in column_types_map.items():
        if col_name in df.columns:
            cast_map.get(target_dtype.lower())

            cast_action = cast_map.get(target_dtype.lower())

            if cast_action:
                df[col_name] = cast_action(df, col_name)

def show_diff(original_df, modified_df):
    """
    Muestra las diferencias entre el DataFrame original y el modificado.

    param original_df: pd.DataFrame
    param modified_df: pd.DataFrame
    """
    original_shape = original_df.shape
    modified_shape = modified_df.shape

    rows_diff = modified_shape[0] - original_shape[0]
    cols_diff = modified_shape[1] - original_shape[1]

    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Modified shape: {modified_shape}")
    logger.info(f"Row difference: {rows_diff}")
    logger.info(f"Column difference: {cols_diff}")

def get_cols_to_drop(original_df, modified_df):
    """
    Obtiene las columnas que deben ser eliminadas del DataFrame modificado.

    param original_df: pd.DataFrame
    param modified_df: pd.DataFrame
    return: list
    """
    original_cols = set(original_df.columns)
    modified_cols = set(modified_df.columns)

    cols_to_drop = list(modified_cols - original_cols)

    if cols_to_drop:
        logger.info(f'Columna(s) a eliminar: {cols_to_drop}')

    return cols_to_drop

# Funciones especificas para formateo de fecha

def convert_date_format(df, date_col):
    """
    Convierte la columna de fecha a un formato datetime estándar.

    param df: pd.DataFrame
    param date_col: str
    return: pd.DataFrame
    """
    count = 0

    df[date_col] = df[date_col].astype(str).str.strip()
    initial_series = df[date_col].copy()

    # Formato 1: YYYY-MM-DD
    df[date_col] = pd.to_datetime(initial_series, format='%Y-%m-%d', errors='coerce')

    # Formato 2: MM/DD/YYYY (US)
    mask_nan_us = df[date_col].isna()

    df.loc[mask_nan_us, date_col] = pd.to_datetime(
        initial_series[mask_nan_us], 
        format='%m/%d/%Y', 
        errors='coerce'
    )

    count += (~df[date_col].isna() | mask_nan_us).sum()

    # Formato 3: DD/MM/YYYY (EU)
    mask_nan_eu = df[date_col].isna()

    df.loc[mask_nan_eu, date_col] = pd.to_datetime(
        initial_series[mask_nan_eu], 
        format='%d/%m/%Y', 
        errors='coerce'
    )

    count += (~df[date_col].isna() | mask_nan_eu).sum()

    print(f'Datos manipulados: {count}')

    return df

def clean_hour(df, hour_col, date_col):
    """
    Limpia y corrige la columna de horas en el DataFrame.

    param df: pd.DataFrame
    param hour_col: str
    param date_col: str
    return: pd.DataFrame
    """
    CUTOFF_DATE = pd.to_datetime('2012-12-31')
    CUTOFF_HR = 23.0 

    count = 0

    df[hour_col] = pd.to_numeric(df[hour_col], errors='coerce') # cast string a numérica para convertir valores corruptos a NaN
    df[hour_col] = df[hour_col].fillna(-1.0)
    df[hour_col] = df[hour_col].astype(float)

    for i in range(1, len(df)):

        current_date = df.loc[i, date_col]
        current_hr = df.loc[i, hour_col]

        prev_date = df.loc[i - 1, date_col]
        prev_hr = df.loc[i - 1, hour_col]

        if prev_date == CUTOFF_DATE and prev_hr == CUTOFF_HR: # poka-yoke en la última hora del último día de 2012
            break

        expected_hr = (prev_hr + 1) % 24 # formato 24 horas

        out_of_range_flag = (current_hr < 0) or (current_hr > 23)

        if out_of_range_flag:
            # Caso 1: si el valor es 1) corrupto o 2) fuera del rango de 24 hr, imputamos el valor secuencial esperado
            df.loc[i, hour_col] = expected_hr

            count += 1

            if expected_hr == 0:
                if pd.notna(prev_date):
                    df.loc[i, date_col] = prev_date + pd.Timedelta(days=1)

        elif current_hr != expected_hr:
            # Caso 2: si es un salto en la secuencia numérica (ej. 1 -> 4) y NO es un cambio de día, no se realiza imputación
            pass

        df[hour_col] = df[hour_col].astype(int)

        mask_final_outlier = (df[hour_col] < 0) | (df[hour_col] > 23)

        if mask_final_outlier.sum() > 0:
            indices_to_fix = df[mask_final_outlier].index

            for idx in indices_to_fix:
                if idx > 0:
                    prev_hr_fixed = df.loc[idx - 1, hour_col]
                    prev_date_fixed = df.loc[idx - 1, date_col]

                    expected_hr_current = (prev_hr_fixed + 1) % 24

                    df.loc[idx, hour_col] = expected_hr_current

                    count += 1

                    if expected_hr_current == 0: # corregir dteday si hay salto de día
                        if pd.notna(prev_date_fixed):
                            df.loc[idx, date_col] = prev_date_fixed + pd.Timedelta(days=1)
                else:
                   df.loc[idx, hour_col] = 0.0

        print(f'Datos manipulados: {count}')

        df[hour_col] = df[hour_col].astype(int)

        return df


def impute_date(df, date_col, hour_col):
    """
    Imputa valores faltantes en la columna de fecha basándose en la columna de hora.

    param df: pd.DataFrame
    param date_col: str
    param hour_col: str
    return: pd.DataFrame
    """
    count = 0

    mask_to_impute = df[date_col].isna() & \
                     df[hour_col].notna()

    indixes_to_impute = df[mask_to_impute].index

    for idx in indixes_to_impute:
        if idx > 0:

            current_hr = df.loc[idx, hour_col]
            prev_idx = idx - 1
            prev_dteday = df.loc[prev_idx, date_col]
            prev_hr = df.loc[prev_idx, hour_col]

            if pd.isna(prev_dteday):
                continue # si no hay fecha anterior, se omite

            sequential_jump_flag = (current_hr == (prev_hr + 1)) or (prev_hr == 23 and current_hr == 0)

            if sequential_jump_flag:

                if current_hr == 0:
                    new_date = prev_dteday + pd.Timedelta(days=1)

                else:
                    new_date = prev_dteday

                df.loc[idx, date_col] = new_date
                count += 1

    print(f'Datos manipulados: {count}')

    return df


def clean_year(df, year_col, date_col):
    """
    Limpia y corrige la columna de años en el DataFrame.

    param df: pd.DataFrame
    param year_col: str
    param date_col: str
    return: pd.DataFrame
    """
    YEAR_MAPPING = {2011: 0.0, 2012: 1.0}

    count = 0

    count_before_fillna = df[year_col].isna().sum()
    df[year_col] = df[year_col].fillna(df[date_col].dt.year.map(YEAR_MAPPING))
    count += count_before_fillna - df[year_col].isna().sum()

    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    mask_corrupt_year = (df[year_col].isna()) | \
                    (~df[year_col].isin([0.0, 1.0])) # rango [0,1] para años (2011, 2012)
    count += mask_corrupt_year.sum() 

    imputed_yr_values = df.loc[mask_corrupt_year, date_col].dt.year.map(YEAR_MAPPING)

    df.loc[mask_corrupt_year, year_col] = imputed_yr_values

    print(f'Datos manipulados: {count}')

    return df


def clean_month(df, month_col, date_col):
    """
    Limpia y corrige la columna de meses en el DataFrame.

    param df: pd.DataFrame
    param month_col: str
    param date_col: str
    return: pd.DataFrame
    """
    count = 0

    count_before_fillna = df[month_col].isna().sum() 
    df[month_col] = df[month_col].fillna(df[date_col].dt.month)
    count += count_before_fillna - df[month_col].isna().sum()

    df[month_col] = pd.to_numeric(df[month_col], errors='coerce')

    mask_corrupt_month = (df[month_col].isna()) | \
                        (df[month_col] < 1) | \
                        (df[month_col] > 12) # rango [1-12] para meses
    count += mask_corrupt_month.sum()

    df.loc[mask_corrupt_month, month_col] = df.loc[mask_corrupt_month, date_col].dt.month.astype(float)

    print(f'Datos manipulados: {count}')

    return df


def clean_weekday(df, weekday_col, date_col):
    """
    Limpia y corrige la columna de días de la semana en el DataFrame.

    param df: pd.DataFrame
    param weekday_col: str
    param date_col: str
    return: pd.DataFrame
    """
    count = 0

    count_before_fillna = df[weekday_col].isna().sum()
    df[weekday_col] = df[weekday_col].fillna(df[date_col].dt.weekday)
    count += count_before_fillna - df[weekday_col].isna().sum()

    df[weekday_col] = pd.to_numeric(df[weekday_col], errors='coerce')

    mask_corrupt_weekday = (df[weekday_col].isna()) | \
                        (df[weekday_col] < 0) | \
                        (df[weekday_col] > 6) # rango [0-6] para días de la semana
    count += mask_corrupt_weekday.sum()

    imputed_weekday_values = df.loc[mask_corrupt_weekday, date_col].dt.weekday.astype(float)

    df.loc[mask_corrupt_weekday, weekday_col] = imputed_weekday_values

    print(f'Datos manipulados: {count}')

    return df


def clean_holiday(df, holiday_col, date_col):
    """
    Limpia y corrige la columna de días festivos en el DataFrame.

    param df: pd.DataFrame
    param holiday_col: str
    param date_col: str
    return: pd.DataFrame
    """
    count = 0

    us_holidays = holidays.US(years=[2011, 2012])

    def get_us_holiday(date):
        if pd.isna(date):
            return np.nan

        return 1.0 if date in us_holidays else 0.0


    df[holiday_col] = pd.to_numeric(df[holiday_col], errors='coerce')

    mask_corrupt_holiday = (df[holiday_col].isna()) | \
                        (~df[holiday_col].isin([0.0, 1.0]))
    count += mask_corrupt_holiday.sum()

    imputed_holiday_values = df.loc[mask_corrupt_holiday, date_col].apply(get_us_holiday)

    df.loc[mask_corrupt_holiday, holiday_col] = imputed_holiday_values
    df[holiday_col].astype(int)

    print(f'Datos manipulados: {count}')

    return df


def clean_workday(df, workday_col, weekday_col, holiday_col):
    """
    Limpia y corrige la columna de días laborales en el DataFrame.
    param df: pd.DataFrame
    param workday_col: str
    param weekday_col: str
    param holiday_col: str
    return: pd.DataFrame
    """
    count = 0

    df[workday_col] = pd.to_numeric(df[workday_col], errors='coerce')

    mask_corrupt_workingday = (df[workday_col].isna()) | \
                            (~df[workday_col].isin([0.0, 1.0]))
    count += mask_corrupt_workingday.sum()

    imputed_workingday_values = df.loc[mask_corrupt_workingday, weekday_col].apply(
        lambda x: 0.0 if x in [0.0, 6.0] else 1.0
    ) # si 'weekday' está en [0, 6], 'workingday' debe ser 0, sino debe ser 1

    df.loc[mask_corrupt_workingday, workday_col] = imputed_workingday_values
    count += (df.loc[df[holiday_col] == 1.0, workday_col] != 0.0).sum()

    df.loc[df[holiday_col] == 1.0, workday_col] = 0.0 # si es día festivo (holiday=1), NO puede ser día laboral (workingday=0)

    print(f'Datos manipulados: {count}')

    return df


def map_season(current_date):
    """
    Mapea una fecha a su estación correspondiente.

    param current_date: pd.Timestamp
    return: float
    """

    if pd.isna(current_date):
        return np.nan

    month_day = (current_date.month, current_date.day)

    # 1.0 == invierno: 21-Dic al 20-Mar
    if (month_day >= (12, 21)) or (month_day <= (3, 20)):
        return 1.0 

    # 2.0 == primavera: 21-Mar al 20-Jun
    elif (month_day >= (3, 21)) and (month_day <= (6, 20)):
        return 2.0

    # 3.0 == verano: 21-Jun al 22-Sep
    elif (month_day >= (6, 21)) and (month_day <= (9, 22)):
        return 3.0

    # 4.0 == otoño): 23-Sep al 20-Dic
    elif (month_day >= (9, 23)) and (month_day <= (12, 20)):
        return 4.0

    return np.nan

def clean_season(df, season_col, date_col):
    """
    Limpia y corrige la columna de estaciones en el DataFrame.

    param df: pd.DataFrame
    param season_col: str
    param date_col: str
    return: pd.DataFrame
    """
    count = 0

    df[season_col] = pd.to_numeric(df[season_col], errors='coerce')

    mask_corrupt_season = (df[season_col].isna()) | \
                        (df[season_col] < 1.0) | \
                        (df[season_col] > 4.0) | \
                        (df[season_col] % 1 != 0)
    count += mask_corrupt_season.sum()

    imputed_season_values = df.loc[mask_corrupt_season, date_col].apply(map_season)

    df.loc[mask_corrupt_season, season_col] = imputed_season_values

    print(f'Datos manipulados: {count}')

    return df

def impute_instant(df, instant_col, date_col, hour_col):
    """
    Imputa la columna instant

    param df: pd.DataFrame
    param instant_col: str
    param date_col: str
    param hour_col: str
    """
    count = 0

    mask_instant_nan = df[instant_col].isna()
    indexes_to_impute = df[mask_instant_nan].index

    for idx in indexes_to_impute:

        if idx > 0 and idx < len(df) - 1:

            prev_idx = idx - 1
            prev_instant = df.loc[prev_idx, instant_col]
            prev_date = df.loc[prev_idx, date_col]
            prev_hr = df.loc[prev_idx, hour_col]

            next_idx = idx + 1
            next_instant = df.loc[next_idx, instant_col]
            next_date = df.loc[next_idx, date_col]
            next_hr = df.loc[next_idx, hour_col]

            current_date = df.loc[idx, date_col]
            current_hr = df.loc[idx, hour_col]

            if pd.isna(prev_instant) or pd.isna(next_instant):
                continue

            if next_instant - prev_instant != 2:
                continue

            expected_hr_current = (prev_hr + 1) % 24
            sequential_hr_flag = current_hr == expected_hr_current

            if expected_hr_current == 0:
                expected_date = prev_date + pd.Timedelta(days=1)
                sequential_date_flag = current_date == expected_date
            else:
                sequential_date_flag = current_date == prev_date

            if sequential_hr_flag and sequential_date_flag:                
                imputed_value = prev_instant + 1.0
                df.loc[idx, instant_col] = imputed_value
                count += 1

    print(f'Datos manipulados: {count}')

    return df


def get_max_records():
    HOURS_PER_DAY = 24

    days_in_2011 = 366 if calendar.isleap(2011) else 365
    days_in_2012 = 366 if calendar.isleap(2012) else 365    

    total = (days_in_2011 + days_in_2012) * HOURS_PER_DAY

    return total


def crosscheck_duplicates(df): # eliminar registros duplicados con menor cantidad de valores faltantes
    key_columns = ['instant', 'dteday', 'hr']
    rows_before_cleanup = df.shape[0]

    # corección: hay registros con un valor en 'instant' duplicados (1 registro con valores en dteday y otro con valores faltantes)
    df_valid_dates = df.dropna(subset=['dteday']).copy()

    df_valid_dates = df_valid_dates.drop_duplicates(subset=['instant'], keep='first')

    instant_to_dteday_map = df_valid_dates.set_index('instant')['dteday']

    mask_dteday_nan = df['dteday'].isna()
    df.loc[mask_dteday_nan, 'dteday'] = df.loc[mask_dteday_nan, 'instant'].map(instant_to_dteday_map)

    df['tmp_nan_count'] = df.isnull().sum(axis=1)

    df = df.sort_values(
        by=key_columns + ['tmp_nan_count'],
        ascending=[True, True, True, True]
    )

    df = df.drop_duplicates(subset=key_columns, keep='first')

    df = df.drop(columns=['tmp_nan_count'], errors='ignore')

    rows_after_cleanup = df.shape[0]
    total_dropped_rows = rows_before_cleanup - rows_after_cleanup

    print(f'Registros eliminados: {total_dropped_rows}')

    return df



def impute_weather_details(df):
    count = 0

    weather_cols = ['temp', 'atemp', 'hum', 'windspeed']

    mask_hum_outlier = (df['hum'] < 0) | (df['hum'] > 100)
    df.loc[mask_hum_outlier, 'hum'] = np.nan

    mask_windspeed_outlier = (df['windspeed'] == 0) | (df['windspeed'] > 60)
    df.loc[mask_windspeed_outlier, 'windspeed'] = np.nan

    nan_count_knn = df[weather_cols].isna().sum().sum()

    imputer = KNNImputer(n_neighbors=5)

    df[weather_cols] = imputer.fit_transform(df[weather_cols])

    nan_count_weathersit = df['weathersit'].isna().sum()

    df['weathersit'] = df['weathersit'].ffill()
    df['weathersit'] = df['weathersit'].bfill()

    count = nan_count_knn + nan_count_weathersit

    df['weathersit'] = df['weathersit'].astype(int)

    print(f'Datos manipulados: {count}')

    return df


def impute_bikes_total_count(df):
    count = 0

    count_cols = ['cnt', 'registered', 'casual']
    rows_initial = df.shape[0]

    mask_not_nan = df[count_cols].notna().all(axis=1)

    # registered > cnt
    mask_reg_too_high = mask_not_nan & (df['registered'] > df['cnt'])
    count += mask_reg_too_high.sum()
    df.loc[mask_reg_too_high, 'registered'] = df['cnt'] - df['casual']

    # casual > cnt
    mask_casual_too_high = mask_not_nan & (df['casual'] > df['cnt'])
    count += mask_casual_too_high.sum()
    df.loc[mask_casual_too_high, 'casual'] = df['cnt'] - df['registered']

    # correción de 'cnt' donde la suma no cuadra (ej: cnt=13, reg=10, cas=0),
    mask_incorrect_sum = df[count_cols].notna().all(axis=1) & (df['cnt'] != (df['registered'] + df['casual']))
    count += mask_incorrect_sum.sum()
    df.loc[mask_incorrect_sum, 'cnt'] = df['registered'] + df['casual']

    # cnt = registered + casual
    mask_impute_cnt = df['cnt'].isna() & df['registered'].notna() & df['casual'].notna()
    count += mask_impute_cnt.sum()
    df.loc[mask_impute_cnt, 'cnt'] = df['registered'] + df['casual']

    # registered = cnt - casual
    mask_impute_registered = df['registered'].isna() & df['cnt'].notna() & df['casual'].notna()
    count += mask_impute_registered.sum()
    df.loc[mask_impute_registered, 'registered'] = df['cnt'] - df['casual']

    # casual = cnt - registered
    mask_impute_casual = df['casual'].isna() & df['cnt'].notna() & df['registered'].notna()
    count += mask_impute_casual.sum()
    df.loc[mask_impute_casual, 'casual'] = df['cnt'] - df['registered']

    # ajustar valores negativos a NaN (resultado de restas o corrupción)
    mask_negatives = (df['registered'] < 0) | (df['casual'] < 0) | (df['cnt'] < 0)
    count += mask_negatives.sum()
    df.loc[mask_negatives, count_cols] = np.nan

    # eliminar filas que aún tienen NaN (2+ valores NaN, o se volvieron negativos)
    df = df.dropna(subset=count_cols)
    rows_dropped_total = rows_initial - df.shape[0]

    print(f'Datos manipulados (celdas): {count}')
    print(f'Registros eliminados: {rows_dropped_total}')

    return df


def rebuild_instant(df, instant_col):
    count = 0

    sort_key = ['dteday', 'hr']
    df = df.sort_values(by=sort_key, ignore_index=True)

    count = df.shape[0] 

    new_instant = df.index + 1

    df[instant_col] = new_instant.astype(int)

    max_instant = df[instant_col].max()

    print(f'Datos manipulados (celdas): {count}')

    return df


def get_distribution_summary(df, num_cols):
    summary_data = []

    for col in num_cols:
        skew = df[col].skew()

        if skew > 0.5:
            skew_interpret = 'Sesgo Derecho (Positivo)'
        elif skew < -0.5:
            skew_interpret = 'Sesgo Izquierdo (Negativo)'
        else:
            skew_interpret = 'Simétrico/Ligero'


        kurtosis = df[col].kurtosis()

        if kurtosis > 0.5:
            kurtosis_interpret = 'Leptocúrtica (Colas Pesadas)'
        elif kurtosis < -0.5:
            kurtosis_interpret = 'Platicúrtica (Colas Ligeras)'
        else:
            kurtosis_interpret = 'Mesocúrtica (Normal)'

        summary_data.append({
            'Variable': col,
            'Skewness': skew,
            'Kurtosis': kurtosis,
            'Tipo de Sesgo': skew_interpret,
            'Forma de la Distribucion': kurtosis_interpret
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df[['Skewness', 'Kurtosis']] = summary_df[['Skewness', 'Kurtosis']].round(2)

    return summary_df


if __name__ == "__main__":
    app()
