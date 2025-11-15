from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from bike_sharing.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "bike_sharing_cleaned.csv",
    output_path: Path = PROCESSED_DATA_DIR / "bike_sharing_transformed.csv",
    # -----------------------------------------
):
    """
    Entrada principal para la generación de características.

    Args:
        input_path (Path): Ruta al archivo CSV de entrada.
        output_path (Path): Ruta al archivo CSV de salida.
    """
    logger.info("="*80)
    logger.info("🏗️ Bike Sharing Pipeline de Generación de Características")
    logger.info("="*80)

    logger.info("Cargando datos...")
    bike_sharing_df = pd.read_csv(input_path)

    logger.info("Construyendo pipeline de transformación...")
    pipeline = pipeline()

    # Ejecuta el pipeline de transformación
    logger.info("Ejecutando pipeline de transformación...")
    logger.info("-"*80)
    bike_sharing_transformed = pipeline.fit_transform(bike_sharing_df)
    logger.info("-"*80)

    # Guarda el DataFrame transformado
    logger.info(f"💾 Guardando DataFrame transformado en: {output_path}")
    bike_sharing_transformed.to_csv(output_path, index=False)
    
    logger.success("="*80)
    logger.success("✅ Pipeline completado exitosamente.")
    logger.success("="*80)

def pipeline():
    """
    Construye y devuelve el pipeline de transformación.

    Returns:
        Pipeline: Pipeline de transformación.
    """
    pipeline = Pipeline(steps=[
        ('log_transformer', LogTransformer()),
        ('cyclic_transformer', CyclicTransformer()),
        ('ohe_transformer', OHETransformer()),
        ('cyclic_scaler', CyclicScalerTransformer()),
        ('column_droper', ColumnDropperTransformer()),
    ])
    return pipeline


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para aplicar la transformación logarítmica a 
    las columnas: 'cnt', 'casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed'
    """

    COLS_TO_TRANSFORM = [
        'cnt', 'casual', 'registered', 
        'temp', 'atemp', 'hum', 'windspeed'
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Aplica la transformación logarítmica a las columnas especificadas en COLS_TO_TRANSFORM.

        Parameters:
        X (pd.DataFrame): DataFrame de entrada.
        Returns:
        pd.DataFrame: DataFrame con las columnas transformadas.
        """
        X_transformed = X.copy()
        X_transformed = self.log_transform(X_transformed, self.COLS_TO_TRANSFORM)
        logger.info(f'✓ Transformación logarítmica aplicada a las columnas {self.COLS_TO_TRANSFORM}.')
        return X_transformed


    def log_transform(self, df, log_cols):
        """
        Aplica la transformación logarítmica a las columnas especificadas en log_cols.

        Parameters:
        df (pd.DataFrame): DataFrame de entrada.
        log_cols (list): Lista de nombres de columnas a transformar.    
        """
        for col in log_cols:
            df[f'{col}_log'] = np.log1p(df[col])

        return df


class CyclicTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para aplicar la transformación cíclica a
    las columnas: 'mnth', 'hr', 'weekday'
    """

    CYCLIC_COLS = ['mnth', 'hr', 'weekday']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Aplica la transformación cíclica a las columnas especificadas en CYCLIC_COLS.

        params:
        X (pd.DataFrame): DataFrame de entrada.

        returns:
        pd.DataFrame: DataFrame con las columnas transformadas.
        """
        X = X.copy()
        X = self.cyclic_transform(X, self.CYCLIC_COLS)
        logger.info(f'✓ Transformación cíclica aplicada a las columnas {self.CYCLIC_COLS}.')

        return X

    def cyclic_transform(self, df, cyclic_cols):
        """
        Aplica la transformación cíclica a las columnas especificadas en cyclic_cols.

        params:
        df (pd.DataFrame): DataFrame de entrada.
        cyclic_cols (list): Lista de nombres de columnas a transformar.

        returns:
        pd.DataFrame: DataFrame con las columnas transformadas.
        """
        for col in cyclic_cols:
            if col == 'hr':
                period = 24
            elif col == 'mnth':
                period = 12
            elif col == 'weekday':
                period = 7
            else:
                continue

            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)

        df.drop(columns=cyclic_cols, inplace=True)

        return df

class OHETransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para aplicar la codificación one-hot a
    las columnas categóricas: 'season', 'yr', 'weathersit', 'holiday', 'workingday'
    """

    CATEGORICAL_COLS = [
        'season', 'yr', 'weathersit', 
        'holiday', 'workingday'
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Aplica la codificación one-hot a las columnas especificadas en CATEGORICAL_COLS.

        params:
        X (pd.DataFrame): DataFrame de entrada.

        returns:
        pd.DataFrame: DataFrame con las columnas transformadas.
        """

        X = X.copy()
        X = self.ohe_transform(X, self.CATEGORICAL_COLS)
        logger.info(f'✓ Codificación one-hot aplicada a las columnas {self.CATEGORICAL_COLS}.')

        return X

    def ohe_transform(self, df, cat_cols):
        """
        Aplica la codificación one-hot a las columnas especificadas en cat_cols.

        params:
        df (pd.DataFrame): DataFrame de entrada.
        cat_cols (list): Lista de nombres de columnas a transformar.

        returns:
        pd.DataFrame: DataFrame con las columnas transformadas.
        """
        df = pd.get_dummies(
            df, 
            columns=cat_cols, 
            drop_first=True,
            dtype=int
        )

        return df

class CyclicScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para escalar las columnas cíclicas
    utilizando StandardScaler.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        cyclic_cols = [col for col in X.columns if ('sin' in col) or ('cos' in col)]
        self.scaler.fit(X[cyclic_cols])
        return self

    def transform(self, X):
        cyclic_cols = [col for col in X.columns if ('sin' in col) or ('cos' in col)]
        X_scaled = X.copy()
        X_scaled[cyclic_cols] = self.scaler.transform(X[cyclic_cols])
        logger.info(f'✓ Escalado aplicado a las columnas cíclicas {cyclic_cols}.')
        return X_scaled

class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para eliminar columnas innecesarias
    después de la transformación.
    """

    COLS_TO_DROP = [
        'cnt', 'casual', 'registered', 
        'dteday' #, 'casual_log', 'registered_log'
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_dropped = X.copy()
        X_dropped.drop(columns=self.COLS_TO_DROP, inplace=True)
        logger.info(f'✓ Columnas eliminadas: {self.COLS_TO_DROP}.')
        return X_dropped


if __name__ == "__main__":
    app()
