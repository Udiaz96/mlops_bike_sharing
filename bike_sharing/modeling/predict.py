import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.pyfunc

from loguru import logger
from tqdm import tqdm
import typer

from bike_sharing.config import PROCESSED_DATA_DIR, REPORTS_DIR

warnings.filterwarnings("ignore")

MODEL_NAME_CASUAL = 'ElasticNet_Casual_BikeSharing'
MODEL_NAME_REGISTERED = 'ElasticNet_Registered_BikeSharing'
MODEL_STAGE = 'Staging'

app = typer.Typer()

def predict(features_df: pd.DataFrame, predictions_path: Path):
    """
    Cargar mejor modelo desde una URI de MLFlow y generar predicciones
    """

    logger.info(f'Loading Models: {MODEL_NAME_CASUAL} + {MODEL_NAME_REGISTERED}')
    model_uri_casual = f'models:/{MODEL_NAME_CASUAL}@{MODEL_STAGE}'
    model_casual = mlflow.pyfunc.load_model(model_uri_casual)

    model_uri_registered = f'models:/{MODEL_NAME_REGISTERED}@{MODEL_STAGE}'
    model_registered = mlflow.pyfunc.load_model(model_uri_registered)

    X_test = features_df.copy()

    logger.info(f'Initiate predictions')
    y_pred_log_casual = model_casual.predict(X_test)
    y_pred_log_registered = model_registered.predict(X_test)

    y_pred_original_casual = np.expm1(y_pred_log_casual)
    y_pred_original_registered = np.expm1(y_pred_log_registered)

    y_predictions = y_pred_original_casual + y_pred_original_registered

    y_predictions[y_predictions < 0] = 0

    results_df = pd.DataFrame({
        'predicted_casual': y_pred_original_casual.round(0).astype(int),
        'predicted_registered': y_pred_original_registered.round(0).astype(int),
        'predicted_count': y_predictions.round(0).astype(int)
    })

    logger.info(f'Predictions generated.')

    return results_df


def evaluate_predictions(
    df_actual: pd.DataFrame, 
    df_predicted: pd.DataFrame
):
    """
    Evaluar métricas: comparando los valores reales (datos transformados) vs las predicciones generadas
    
    Args:
        df_actual (pd.DataFrame): DataFrame que contiene la columna del target real ('cnt').
        df_predicted (pd.DataFrame): DataFrame que contiene la columna de la predicción ('predicted_count' = registered + casual).
    """
    
    if 'cnt' in df_actual.columns:
        y_actual = df_actual['cnt'].values
    elif 'cnt_log' in df_actual.columns:
        y_actual = np.expm1(df_actual['cnt_log']).values
    else:
        logger.error("No se encontró la columna de target 'cnt' o 'cnt_log' para evaluar.")
        return

    if 'predicted_count' not in df_predicted.columns:
        logger.error("La columna 'predicted_count' no se encontró en las predicciones.")
        return
        
    y_predicted = df_predicted['predicted_count'].values
    
    y_predicted[y_predicted < 0] = 0
    y_actual[y_actual < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)
    
    logger.info('-' * 80)
    logger.success(f'Evaluation Metrics Results')
    logger.info("-" * 80)
    logger.info(f'RMSE: {rmse:.2f}')
    logger.info(f'MAE : {mae:.2f} bicicletas')
    logger.info(f'R2: {r2:.4f}')
    logger.info('-' * 80)

    if r2 >= 0.9:
        logger.success(f'Results: Above Average. R2 for the model explains most of the variance')
    elif r2 >= 0.7:
        logger.warning(f'Results: Average. Can be optimized')
    else:
        logger.error(f'Results: Below Average')


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "bike_sharing_transformed.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv" 
):

    logger.info('-'*80)
    logger.info(f'Initiate Predictions: Model {MODEL_NAME_CASUAL}, + {MODEL_NAME_REGISTERED}')
    logger.info('-'*80)
    
    try:
        input_df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f'File not found: {input_path}')
        return
    
    cols_to_drop = [
        'cnt_log', 'cnt', 'casual', 'registered'
    ]

    features_to_predict = input_df.drop(
        columns=cols_to_drop,
        errors='ignore'
    )

    results_df = predict(features_to_predict, predictions_path)

    logger.success('Experiments completed successfully.')

    context_cols = [col for col in ['dteday', 'hr'] if col in input_df.columns]
    context_df = input_df[context_cols].reset_index(drop=True)
    
    final_output_df = pd.concat([context_df, results_df], axis=1)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_df.to_csv(predictions_path, index=False)

    logger.info(f'Predictions completed successfully. Output saved to: {predictions_path}')

    try:
        df_actual = pd.read_csv(input_path)        
        df_predicted = pd.read_csv(predictions_path)

        if len(df_actual) != len(df_predicted):
            logger.error(f'Warning: different lengths for datasets.')
            
        evaluate_predictions(df_actual, df_predicted)

    except FileNotFoundError as e:
        logger.error(f'File not found. Error: {e}')
    except Exception as e:
        logger.error(f'Error: {e}')

if __name__ == "__main__":
    app()
