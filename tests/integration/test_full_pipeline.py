import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from bike_sharing.dataset import (
    TypeCastingTransformer,
    ColumnManagementTransformer,
    DateTimeCleaningTransformer,
    FeatureCleaningTransformer,
    InstantImputationTransformer,
    DuplicateHandlingTransformer,
    WeatherImputationTransformer,
    CountValidationTransformer,
    FinalCleanupTransformer
)

from bike_sharing.features import (
    LogTransformer,
    CyclicTransformer,
    OHETransformer,
    CyclicScalerTransformer,
    ColumnDropperTransformer
)


def create_sample_bike_sharing_data():
    """
    Carga una muestra del dataset real de bike sharing desde el archivo de prueba.
    """
    # Cargar el archivo de muestra de prueba (200 filas)
    test_data_dir = Path(__file__).parent.parent / "data"
    data_path = test_data_dir / "bike_sharing_modified_sample.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {data_path}. "
            "Please ensure tests/data/bike_sharing_modified_sample.csv exists."
        )

    df = pd.read_csv(data_path)
    return df


def create_original_reference_data():
    """
    Carga el dataset original de referencia para el ColumnManagementTransformer.
    """
    # Cargar el archivo de muestra de prueba
    test_data_dir = Path(__file__).parent.parent / "data"
    data_path = test_data_dir / "bike_sharing_original_sample.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {data_path}. "
            "Please ensure tests/data/bike_sharing_original_sample.csv exists."
        )

    df = pd.read_csv(data_path)
    return df


@pytest.fixture
def sample_data():
    """Fixture que proporciona datos de muestra para las pruebas."""
    return create_sample_bike_sharing_data()


@pytest.fixture
def original_data():
    """Fixture que proporciona datos originales de referencia."""
    return create_original_reference_data()


def test_full_data_cleaning_pipeline(sample_data, original_data):
    """
    Prueba de integración: Pipeline completo de limpieza de datos.
    """
    # Construir el pipeline de limpieza
    cleaning_pipeline = Pipeline([
        ('type_casting', TypeCastingTransformer()),
        ('column_management', ColumnManagementTransformer(original_data)),
        ('datetime_cleaning', DateTimeCleaningTransformer()),
        ('feature_cleaning', FeatureCleaningTransformer()),
        ('instant_imputation', InstantImputationTransformer()),
        ('duplicate_handling', DuplicateHandlingTransformer()),
        ('weather_imputation', WeatherImputationTransformer()),
        ('count_validation', CountValidationTransformer()),
        ('final_cleanup', FinalCleanupTransformer()),
    ])

    # Ejecutar el pipeline
    cleaned_data = cleaning_pipeline.fit_transform(sample_data)

    # Verificaciones
    assert isinstance(cleaned_data, pd.DataFrame)
    assert len(cleaned_data) > 0
    assert 'dteday' in cleaned_data.columns
    assert 'cnt' in cleaned_data.columns
    assert 'casual' in cleaned_data.columns
    assert 'registered' in cleaned_data.columns

    # Verificar que no hay valores NaN críticos después de la limpieza
    assert cleaned_data['cnt'].isna().sum() == 0
    assert cleaned_data['casual'].isna().sum() == 0
    assert cleaned_data['registered'].isna().sum() == 0

    # Verificar tipos de datos
    assert cleaned_data['instant'].dtype == 'Int64' or cleaned_data['instant'].dtype == 'int64'
    assert cleaned_data['cnt'].dtype == 'Int64' or cleaned_data['cnt'].dtype == 'int64'


def test_full_feature_engineering_pipeline(sample_data, original_data):
    """
    Prueba de integración: Pipeline completo de limpieza + ingeniería de características.
    """
    # Primero limpiar los datos
    cleaning_pipeline = Pipeline([
        ('type_casting', TypeCastingTransformer()),
        ('column_management', ColumnManagementTransformer(original_data)),
        ('datetime_cleaning', DateTimeCleaningTransformer()),
        ('feature_cleaning', FeatureCleaningTransformer()),
        ('instant_imputation', InstantImputationTransformer()),
        ('duplicate_handling', DuplicateHandlingTransformer()),
        ('weather_imputation', WeatherImputationTransformer()),
        ('count_validation', CountValidationTransformer()),
        ('final_cleanup', FinalCleanupTransformer()),
    ])

    cleaned_data = cleaning_pipeline.fit_transform(sample_data)

    # Construir el pipeline de características
    features_pipeline = Pipeline([
        ('log_transformer', LogTransformer()),
        ('cyclic_transformer', CyclicTransformer()),
        ('ohe_transformer', OHETransformer()),
        ('cyclic_scaler', CyclicScalerTransformer()),
        ('column_droper', ColumnDropperTransformer()),
    ])

    # Ejecutar el pipeline de características
    transformed_data = features_pipeline.fit_transform(cleaned_data)

    # Verificaciones
    assert isinstance(transformed_data, pd.DataFrame)
    assert len(transformed_data) > 0

    # Verificar que las columnas logarítmicas fueron creadas
    assert 'cnt_log' in transformed_data.columns
    assert 'casual_log' in transformed_data.columns
    assert 'registered_log' in transformed_data.columns

    # Verificar que las columnas cíclicas fueron creadas
    assert 'hr_sin' in transformed_data.columns
    assert 'hr_cos' in transformed_data.columns
    assert 'mnth_sin' in transformed_data.columns
    assert 'mnth_cos' in transformed_data.columns

    # Verificar que las columnas originales temporales fueron eliminadas
    assert 'hr' not in transformed_data.columns
    assert 'mnth' not in transformed_data.columns
    assert 'weekday' not in transformed_data.columns

    # Verificar que las columnas originales categóricas fueron procesadas
    # Nota: Con drop_first=True, si solo hay una categoría, no se crean columnas nuevas
    assert 'season' not in transformed_data.columns
    assert 'yr' not in transformed_data.columns
    assert 'weathersit' not in transformed_data.columns

    # Verificar que las columnas originales fueron eliminadas por ColumnDropper
    assert 'cnt' not in transformed_data.columns
    assert 'casual' not in transformed_data.columns
    assert 'registered' not in transformed_data.columns


def test_full_pipeline_with_model_training(sample_data, original_data):
    """
    Prueba de integración completa: Limpieza + Características + Entrenamiento.
    """
    # 1. Limpieza de datos
    cleaning_pipeline = Pipeline([
        ('type_casting', TypeCastingTransformer()),
        ('column_management', ColumnManagementTransformer(original_data)),
        ('datetime_cleaning', DateTimeCleaningTransformer()),
        ('feature_cleaning', FeatureCleaningTransformer()),
        ('instant_imputation', InstantImputationTransformer()),
        ('duplicate_handling', DuplicateHandlingTransformer()),
        ('weather_imputation', WeatherImputationTransformer()),
        ('count_validation', CountValidationTransformer()),
        ('final_cleanup', FinalCleanupTransformer()),
    ])

    cleaned_data = cleaning_pipeline.fit_transform(sample_data)

    # 2. Ingeniería de características
    features_pipeline = Pipeline([
        ('log_transformer', LogTransformer()),
        ('cyclic_transformer', CyclicTransformer()),
        ('ohe_transformer', OHETransformer()),
        ('cyclic_scaler', CyclicScalerTransformer()),
        ('column_droper', ColumnDropperTransformer()),
    ])

    transformed_data = features_pipeline.fit_transform(cleaned_data)

    # 3. Preparar datos para entrenamiento
    X = transformed_data.drop(columns=['cnt_log'])
    y_casual = transformed_data['casual_log']
    y_registered = transformed_data['registered_log']

    # Verificar que tenemos features y targets
    assert len(X) > 0
    assert len(y_casual) == len(X)
    assert len(y_registered) == len(X)

    # 4. Entrenar modelos simples (RandomForest con pocos árboles para rapidez)
    model_casual = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    model_registered = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)

    # Split simple (70-30)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_casual_train, y_casual_test = y_casual[:split_idx], y_casual[split_idx:]
    y_registered_train, y_registered_test = y_registered[:split_idx], y_registered[split_idx:]

    # Entrenar
    model_casual.fit(X_train, y_casual_train)
    model_registered.fit(X_train, y_registered_train)

    # 5. Generar predicciones
    y_pred_casual_log = model_casual.predict(X_test)
    y_pred_registered_log = model_registered.predict(X_test)

    # Convertir de log a escala original
    y_pred_casual = np.expm1(y_pred_casual_log)
    y_pred_registered = np.expm1(y_pred_registered_log)
    y_pred_combined = y_pred_casual + y_pred_registered

    # Manejar valores negativos
    y_pred_combined[y_pred_combined < 0] = 0

    # 6. Verificaciones finales
    assert len(y_pred_combined) == len(X_test)
    assert (y_pred_combined >= 0).all()
    assert np.isfinite(y_pred_combined).all()

    # Verificar que las predicciones están en un rango razonable
    assert y_pred_combined.min() >= 0
    assert y_pred_combined.max() < 10000  # Límite razonable para bike counts

    # Calcular métricas básicas
    from sklearn.metrics import mean_squared_error, r2_score

    y_test_combined = np.expm1(y_casual_test) + np.expm1(y_registered_test)

    rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred_combined))
    r2 = r2_score(y_test_combined, y_pred_combined)

    # Verificar que las métricas son razonables
    assert rmse >= 0
    assert -1 <= r2 <= 1  # R2 puede ser negativo si el modelo es muy malo


def test_pipeline_data_flow_consistency(sample_data, original_data):
    """
    Prueba de integración: Verifica la consistencia del flujo de datos a través del pipeline.
    """
    initial_rows = len(sample_data)

    # 1. Limpieza
    cleaning_pipeline = Pipeline([
        ('type_casting', TypeCastingTransformer()),
        ('column_management', ColumnManagementTransformer(original_data)),
        ('datetime_cleaning', DateTimeCleaningTransformer()),
        ('feature_cleaning', FeatureCleaningTransformer()),
        ('instant_imputation', InstantImputationTransformer()),
        ('duplicate_handling', DuplicateHandlingTransformer()),
        ('weather_imputation', WeatherImputationTransformer()),
        ('count_validation', CountValidationTransformer()),
        ('final_cleanup', FinalCleanupTransformer()),
    ])

    cleaned_data = cleaning_pipeline.fit_transform(sample_data)
    cleaned_rows = len(cleaned_data)

    # El número de filas puede cambiar debido a eliminación de duplicados y validación
    # pero no debería aumentar
    assert cleaned_rows <= initial_rows
    assert cleaned_rows > 0

    # 2. Transformación de características
    features_pipeline = Pipeline([
        ('log_transformer', LogTransformer()),
        ('cyclic_transformer', CyclicTransformer()),
        ('ohe_transformer', OHETransformer()),
        ('cyclic_scaler', CyclicScalerTransformer()),
        ('column_droper', ColumnDropperTransformer()),
    ])

    transformed_data = features_pipeline.fit_transform(cleaned_data)
    transformed_rows = len(transformed_data)

    # El número de filas no debe cambiar durante la transformación de características
    assert transformed_rows == cleaned_rows

    # Verificar que el número de columnas cambió (características transformadas)
    assert len(transformed_data.columns) != len(cleaned_data.columns)

    # Verificar que no hay NaNs en columnas críticas después de todo el pipeline
    assert not transformed_data['cnt_log'].isna().any()
    assert not transformed_data['casual_log'].isna().any()
    assert not transformed_data['registered_log'].isna().any()


def test_end_to_end_with_temp_files(sample_data, original_data):
    """
    Prueba de integración end-to-end usando archivos temporales.
    Simula el flujo completo como si se ejecutara desde la línea de comandos.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Guardar datos de muestra
        raw_path = tmpdir_path / "raw_data.csv"
        original_path = tmpdir_path / "original_data.csv"
        cleaned_path = tmpdir_path / "cleaned_data.csv"
        transformed_path = tmpdir_path / "transformed_data.csv"

        sample_data.to_csv(raw_path, index=False)
        original_data.to_csv(original_path, index=False)

        # 1. Pipeline de limpieza
        df = pd.read_csv(raw_path)
        original_df = pd.read_csv(original_path)

        cleaning_pipeline = Pipeline([
            ('type_casting', TypeCastingTransformer()),
            ('column_management', ColumnManagementTransformer(original_df)),
            ('datetime_cleaning', DateTimeCleaningTransformer()),
            ('feature_cleaning', FeatureCleaningTransformer()),
            ('instant_imputation', InstantImputationTransformer()),
            ('duplicate_handling', DuplicateHandlingTransformer()),
            ('weather_imputation', WeatherImputationTransformer()),
            ('count_validation', CountValidationTransformer()),
            ('final_cleanup', FinalCleanupTransformer()),
        ])

        cleaned_df = cleaning_pipeline.fit_transform(df)
        cleaned_df.to_csv(cleaned_path, index=False)

        # Verificar que el archivo fue creado
        assert cleaned_path.exists()

        # 2. Pipeline de características
        cleaned_df = pd.read_csv(cleaned_path)

        features_pipeline = Pipeline([
            ('log_transformer', LogTransformer()),
            ('cyclic_transformer', CyclicTransformer()),
            ('ohe_transformer', OHETransformer()),
            ('cyclic_scaler', CyclicScalerTransformer()),
            ('column_droper', ColumnDropperTransformer()),
        ])

        transformed_df = features_pipeline.fit_transform(cleaned_df)
        transformed_df.to_csv(transformed_path, index=False)

        # Verificar que el archivo fue creado
        assert transformed_path.exists()

        # 3. Leer y verificar archivo final
        final_df = pd.read_csv(transformed_path)

        assert len(final_df) > 0
        assert 'cnt_log' in final_df.columns
        assert 'casual_log' in final_df.columns
        assert 'registered_log' in final_df.columns

        # Verificar que podemos usar estos datos para entrenamiento
        X = final_df.drop(columns=['cnt_log'])
        y = final_df['cnt_log']

        assert len(X) == len(y)
        assert len(X.columns) > 0

        # Entrenar un modelo simple para verificar que los datos son válidos
        model = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=42)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # No debe fallar
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert np.isfinite(predictions).all()
