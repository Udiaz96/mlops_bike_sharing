import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

from loguru import logger
from tqdm import tqdm
import typer

from bike_sharing.config import PROCESSED_DATA_DIR

warnings.filterwarnings("ignore")

mlflow.set_experiment("Bike_Sharing_5-Model-Double_Experimentation")

app = typer.Typer()

def train_model(model_type, model_name, params, X_train, X_test, y_casual_train, y_casual_test, y_registered_train, y_registered_test):
    """
    Funcion central para entrenar modelos, evaluar sus métricas y registrarlas en MLFlow
    """

    param_tag = '_'.join(f'{k}-{v}' for k, v in params.items() if k in ['n_estimators', 'max_depth', 'alpha', 'C', 'learning_rate'])
    run_name = f'{model_name}_{param_tag}_(Double-Pred)'

    with mlflow.start_run(run_name=run_name) as run:
        model_casual_total = model_type(**params)
        model_casual_total.fit(X_train, y_casual_train)

        model_registered_total = model_type(**params)
        model_registered_total.fit(X_train, y_registered_train)

        y_pred_casual_log = model_casual_total.predict(X_test)
        y_pred_registered_log = model_registered_total.predict(X_test)

        y_pred_casual_original = np.expm1(y_pred_casual_log)
        y_pred_registered_original = np.expm1(y_pred_registered_log)

        y_pred_combined_original = y_pred_casual_original + y_pred_registered_original

        y_test_combined_original = np.expm1(y_casual_test) + np.expm1(y_registered_test)

        y_pred_combined_original[y_pred_combined_original < 0] = 0

        r2 = r2_score(y_test_combined_original, y_pred_combined_original)
        rmse = np.sqrt(mean_squared_error(y_test_combined_original, y_pred_combined_original))
        mae = mean_absolute_error(y_test_combined_original, y_pred_combined_original)


        mlflow.log_params(params)
        mlflow.log_param('model_name', f'Double_{model_name}')
        mlflow.log_param('strategy', 'Double Pred - Casual+Registered')

        mlflow.log_metric('R2', r2)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)

        mlflow.sklearn.log_model(
            sk_model=model_casual_total,
            name='bikes_casual_model', 
            registered_model_name=f'{model_name}_Casual_BikeSharing'
        )
        
        mlflow.sklearn.log_model(
            sk_model=model_registered_total,
            name='bikes_registered_model', 
            registered_model_name=f'{model_name}_Registered_BikeSharing'
        )

        logger.info(f'Run {run_name} completed. Double-RMSE: {rmse:.2f}')

        return rmse, run.info.run_id, model_name
    

# # CONFIGURACIONES / HIPERPARAMETROS POR MODELO

EXPERIMENT_PARAMS = [
    {
        'model_type': RandomForestRegressor,
        'model_name': 'RandomForest',
        'params': [
            {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
            {'n_estimators': 200, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1},
            {'n_estimators': 300, 'max_depth': 15, 'min_samples_leaf': 2,'random_state': 42, 'n_jobs': -1},
            {'n_estimators': 500, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1},
            {'n_estimators': 500, 'max_depth': 25, 'min_samples_leaf': 1, 'random_state': 42, 'n_jobs': -1},
            {'n_estimators': 300, 'max_depth': 18, 'min_samples_leaf': 2, 'min_samples_split': 5, 'random_state': 42, 'n_jobs': -1}
        ]   
    },
    {
        'model_type': ElasticNet,
        'model_name': 'ElasticNet',
        'params': [
            {'alpha': 0.1, 'l1_ratio': 0.5, 'random_state': 42, 'max_iter': 10000},
            {'alpha': 0.001, 'l1_ratio': 0.8, 'random_state': 42, 'max_iter': 10000},
            {'alpha': 0.1, 'l1_ratio': 0.6, 'random_state': 42, 'max_iter': 10000},
            {'alpha': 0.005, 'l1_ratio': 0.9, 'random_state': 42, 'max_iter': 10000}
        ]   
    },
    {
        'model_type': SVR,
        'model_name': 'SVR',
        'params': [
            {'kernel': 'rbf', 'C': 10, 'epsilon': 0.1},
            {'kernel': 'rbf', 'C': 1, 'epsilon': 0.05}
        ]   
    },
    {
        'model_type': XGBRegressor,
        'model_name': 'XGBoost',
        'params': [
            {'n_estimators': 150, 'max_depth': 5, 'random_state': 42, 'learning_rate': 0.1},
            {'n_estimators': 300, 'max_depth': 7, 'random_state': 42, 'learning_rate': 0.05},
            {'n_estimators': 400, 'max_depth': 6, 'random_state': 42, 'learning_rate': 0.03},
            {'n_estimators': 500, 'max_depth': 5, 'random_state': 42, 'learning_rate': 0.05}
        ]   
    },
    {
        'model_type': LGBMRegressor,
        'model_name': 'LightGBM',
        'params': [
            {'n_estimators': 150, 'max_depth': 10, 'random_state': 42, 'learning_rate': 0.1, 'n_jobs': -1},
            {'n_estimators': 300, 'max_depth': 15, 'random_state': 42, 'learning_rate': 0.05, 'n_jobs': -1},
            {'n_estimators': 400, 'max_depth': 12, 'random_state': 42, 'learning_rate': 0.03, 'n_jobs': -1},
            {'n_estimators': 600, 'max_depth': 15, 'random_state': 42, 'learning_rate': 0.05, 'n_jobs': -1}
        ]   
    }
]


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "bike_sharing_transformed.csv",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Funcion principal para cargar datos, ejecutar el split de los mismos
    y experimantar con 5 modelos de regresión, registrando resultados.
    """
    logger.info('-'*80)
    logger.info('5-Models Double Experimentation with MLFlow')
    logger.info('-'*80)

    try:
        bike_sharing_df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(f'File not found: {input_path}')
        return
    
    X = bike_sharing_df.drop(columns = ['cnt_log'])
    y_casual = bike_sharing_df['casual_log']
    y_registered = bike_sharing_df['registered_log']

    X_train, X_test, _, _ = train_test_split(X, y_casual, test_size = test_size, random_state = random_state)
    
    _, _, y_casual_train, y_casual_test = train_test_split(y_casual, y_casual, test_size = test_size, random_state = random_state)
    
    _, _, y_registered_train, y_registered_test = train_test_split(y_registered, y_registered, test_size = test_size, random_state = random_state)

    logger.info(f'Train Data: {len(X_train)} | Test Data: {len(X_test)}')

    best_rmse = float('inf')
    best_run_id = None
    best_model_name = None

    total_runs = sum(len(c['params']) for c in EXPERIMENT_PARAMS)
    logger.info(f'Initiate {total_runs} runs with MLFlow')

    with tqdm(total = total_runs) as pbar:
        for model_config in EXPERIMENT_PARAMS:
            for params in model_config['params']:

                rmse, run_id, model_name = train_model(
                    model_type = model_config['model_type'],
                    model_name = model_config['model_name'],
                    params = params,
                    X_train = X_train,
                    X_test = X_test,
                    y_casual_train = y_casual_train,
                    y_casual_test = y_casual_test,
                    y_registered_train = y_registered_train,
                    y_registered_test = y_registered_test
                )

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_run_id = run_id
                    best_model_name = model_name

                pbar.update(1)

    logger.success('Experiments completed successfully.')
    logger.info('-'*80)
    logger.info(f'Best Model RMSE: {best_rmse:.2f}')
    logger.info(f'Best Model Name: {best_model_name}')
    logger.info(f'Best Model Run ID: {best_run_id}')
    logger.info('-'*80)

if __name__ == "__main__":
    app()
