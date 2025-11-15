import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import typer

from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture 
from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_squared_error, mean_absolute_error

import mlflow
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error, mean_absolute_error

from bike_sharing.config import PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR
from bike_sharing.plots import plot_drift_distribution

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME_CASUAL = 'ElasticNet_Casual_BikeSharing'
MODEL_NAME_REGISTERED = 'ElasticNet_Registered_BikeSharing'
MODEL_ALIAS = 'Staging' # debe coincidir con el alias usado en predict.py
BASELINE_RMSE = 3.07 # RMSE de la Fase 2 (baseline)

DRIFT_FEATURE_NAME = 'temp_log' # columna clave para simular drift
MEANS_SHIFT_AMOUNT = -0.5 
N_COMPONENTS_GMM = 5 
N_SAMPLES_SYNTHETIC = 5000 
ALPHA_THRESHOLD = 0.05 # (p-valor)
ALERT_THRESHOLD_RMSE_FACTOR = 1.5 


app = typer.Typer()


def load_models():
    """
    Cargar los modelos (implementación combinada de cnt = casual + registered) 
    desde MLflow usando el alias Production.
    """
    
    try:
        model_uri_casual = f'models:/{MODEL_NAME_CASUAL}@{MODEL_ALIAS}'
        model_casual = mlflow.pyfunc.load_model(model_uri_casual)

        model_uri_registered = f'models:/{MODEL_NAME_REGISTERED}@{MODEL_ALIAS}'
        model_registered = mlflow.pyfunc.load_model(model_uri_registered)
                
        logger.success(f'Modelos {MODEL_ALIAS} cargados exitosamente.')
        return model_casual, model_registered
    except Exception as e:
        logger.error(f'Error: El modelo no esta registrado.')
        return None, None
    

def predict_and_evaluate(df_drifted: pd.DataFrame, model_casual, model_registered):
    """
    Evaluar métricas de desempeño (RMSE y MAE) en el set con drift.
    """
    
    y_target_cols = ['cnt', 'cnt_log']
    cols_to_drop_for_prediction = ['cnt', 'casual', 'registered', 'cnt_log']

    features_to_predict = df_drifted.drop(
        columns=cols_to_drop_for_prediction,
        errors='ignore'
    )

    # predicción doble
    pred_casual_log = model_casual.predict(features_to_predict)
    pred_registered_log = model_registered.predict(features_to_predict)

    pred_casual_original = np.expm1(pred_casual_log)
    pred_registered_original = np.expm1(pred_registered_log)
    y_predicted = pred_casual_original + pred_registered_original
    y_predicted[y_predicted < 0] = 0

    # obtener el target real para la evaluación (en escala original)
    if 'cnt' in df_drifted.columns:
        y_actual = df_drifted['cnt'].values
    elif 'cnt_log' in df_drifted.columns:
        y_actual = np.expm1(df_drifted['cnt_log']).values
    else:
        logger.error("No se encontró el target real ('cnt' o 'cnt_log') para evaluar.")
        return None, None
        
    y_actual[y_actual < 0] = 0

    rmse_drift = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mae_drift = mean_absolute_error(y_actual, y_predicted)
    
    return rmse_drift, mae_drift


def generate_synthetic_drift(df: pd.DataFrame, feature: str, shift: float):
    """
    Generar datos sintéticos para una característica dada 
    usando un GMM con medias alteradas.
    """
    
    # 1. Ajuste del GMM a la distribución original
    data_original = df[feature].to_numpy().reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=N_COMPONENTS_GMM, random_state=0).fit(data_original)
    
    logger.info(f"Medias GMM Originales:\n{gmm.means_.flatten()}")
    
    # 2. Ajuste de Medias (Aplicación de meanshift)
    modified_means = gmm.means_ + shift
    gmm.means_ = modified_means
    
    logger.warning(f"Medias GMM Modificadas (Shift: {shift}):\n{gmm.means_.flatten()}")
    
    # 3. Generación de datos sintéticos
    synthetic_samples, _ = gmm.sample(n_samples=N_SAMPLES_SYNTHETIC)
    synthetic_values = synthetic_samples.flatten()
    
    return synthetic_values


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "bike_sharing_transformed.csv",
):
    """
    Ejecutar el pipeline de Monitoreo de Drift basado en Simulación GMM y Test Estadístico.
    """
    logger.info("-" * 80)
    logger.info("Iniciando Simulación y Monitoreo de Data Drift")
    logger.info("-" * 80)

    try:
        df_full = pd.read_csv(input_path)
        
        cols_features = df_full.drop(columns=[
            'cnt_log'
        ], errors='ignore').columns
        
        X = df_full[cols_features]
        
        X_train, X_test, _, _ = train_test_split(X, df_full['cnt_log'], test_size=0.3, random_state=42)
        X_train = X_train.reset_index(drop=True)
        
        synthetic_values = generate_synthetic_drift(
            X_train, DRIFT_FEATURE_NAME, MEANS_SHIFT_AMOUNT
        )
        
        X_synthetic = X_test.sample(N_SAMPLES_SYNTHETIC, random_state=42).reset_index(drop=True)
        X_synthetic[DRIFT_FEATURE_NAME] = synthetic_values[:N_SAMPLES_SYNTHETIC] 

        X_ref = X_train[DRIFT_FEATURE_NAME].values
        X_prod = X_synthetic[DRIFT_FEATURE_NAME].values
        
        stat, p_value = mannwhitneyu(X_ref, X_prod, alternative='two-sided')
        drift_detected_statistical = p_value < ALPHA_THRESHOLD
        
        logger.info("-" * 80)
        logger.info(f'Simulación KSDrift en {DRIFT_FEATURE_NAME})')
        logger.info(f'Estadístico (U): {stat:.4f}, P-Value: {p_value:.4f} (Umbral Alfa: {ALPHA_THRESHOLD})')
        
        logger.info(f'SK-Drift detectado: {drift_detected_statistical}')
        
        if drift_detected_statistical:
            logger.error(f'¡DRIFT DETECTADO! (p-value < {ALPHA_THRESHOLD})')
        else:
            logger.success(f'Sin Drift Estadístico significativo.')
            
        logger.info("-" * 80)

        plot_drift_distribution(
            df_base=X_train.sample(N_SAMPLES_SYNTHETIC, random_state=42).reset_index(drop=True),
            df_drifted=X_synthetic,
            feature=DRIFT_FEATURE_NAME,
            output_path = FIGURES_DIR / f"gmm_drift_visualization_{DRIFT_FEATURE_NAME}.png"
        )
        
        model_casual, model_registered = load_models()
        
        if model_casual is not None and model_registered is not None:
            target_cols = df_full[['cnt_log']].iloc[X_synthetic.index].copy().reset_index(drop=True) 
            df_monitor = pd.concat([X_synthetic, target_cols], axis=1)

            rmse_drift, mae_drift = predict_and_evaluate(df_monitor, model_casual, model_registered)
            
            if rmse_drift is not None:
                rmse_increase_factor = rmse_drift / BASELINE_RMSE
            
                logger.info("-" * 80)
                logger.info(f'Análisis de degradación de performance (RMSE)')
                logger.info(f'RMSE con Drift: {rmse_drift:.2f} (vs. Base: {BASELINE_RMSE:.2f})')
                
                if rmse_increase_factor > ALERT_THRESHOLD_RMSE_FACTOR:
                    logger.error(f'¡ALERTA CRÍTICA! Degradación de Performance detectada ({rmse_increase_factor:.2f}x).')
                else:
                    logger.success(f'Performance OK.')

    except Exception as e:
        logger.error(f'Error fatal durante el monitoreo: {e}')
        
    logger.info("=" * 80)


if __name__ == "__main__":
    app()