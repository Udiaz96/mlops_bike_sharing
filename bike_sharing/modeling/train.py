import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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

# Model registry: maps model names to their classes
MODEL_REGISTRY = {
    'random_forest': RandomForestRegressor,
    'elastic_net': ElasticNet,
    'svr': SVR,
    'xgboost': XGBRegressor,
    'lightgbm': LGBMRegressor
}

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
    

# Removed EXPERIMENT_PARAMS - now using params.yml


@app.command()
def main(
    model_type: str = typer.Argument(..., help="Model type to train (e.g., random_forest, elastic_net, svr, xgboost, lightgbm)"),

    # Batch mode
    batch: bool = typer.Option(False, "--batch", help="Run all configs from params.yaml for this model type"),

    # Training configuration
    params_path: Path = typer.Option(Path("params.yaml"), help="Path to params.yaml file"),
    input_path: Path = typer.Option(None, help="Path to input data (overrides params.yaml if provided)"),
    test_size: float = typer.Option(None, help="Test split size (overrides params.yaml if provided)"),
    random_state: int = typer.Option(None, help="Random state for train/test split (overrides params.yaml if provided)"),

    # Common hyperparameters (tree-based models)
    n_estimators: int = typer.Option(None, help="Number of estimators (RandomForest, XGBoost, LightGBM)"),
    max_depth: int = typer.Option(None, help="Maximum depth (RandomForest, XGBoost, LightGBM)"),
    learning_rate: float = typer.Option(None, help="Learning rate (XGBoost, LightGBM)"),
    n_jobs: int = typer.Option(None, help="Number of parallel jobs (RandomForest, LightGBM)"),

    # RandomForest specific
    min_samples_leaf: int = typer.Option(None, help="Minimum samples per leaf (RandomForest)"),
    min_samples_split: int = typer.Option(None, help="Minimum samples to split (RandomForest)"),

    # ElasticNet specific
    alpha: float = typer.Option(None, help="Regularization strength (ElasticNet)"),
    l1_ratio: float = typer.Option(None, help="L1 ratio (ElasticNet)"),
    max_iter: int = typer.Option(None, help="Maximum iterations (ElasticNet)"),

    # SVR specific
    kernel: str = typer.Option(None, help="Kernel type (SVR)"),
    c: float = typer.Option(None, help="Regularization parameter C (SVR)"),
    epsilon: float = typer.Option(None, help="Epsilon parameter (SVR)"),

    # Model random state (separate from train/test split random_state)
    model_random_state: int = typer.Option(None, help="Random state for model training")
):
    """
    Train a model with specified hyperparameters via CLI or batch mode.
    Designed for DVC pipelines and quick experimentation.

    Example usage:
        # Single experiment with CLI hyperparameters
        python -m bike_sharing.modeling.train random_forest --n-estimators 300 --max-depth 15

        # Batch mode - run all configs from params.yaml
        python -m bike_sharing.modeling.train random_forest --batch
    """
    # Load parameters from params.yaml
    logger.info(f'Loading configuration from {params_path}')
    with open(params_path, 'r') as f:
        params_config = yaml.safe_load(f)

    # Get training configuration with CLI overrides
    train_config = params_config['train']
    input_file = input_path if input_path else Path(train_config['input_path'])
    test_split = test_size if test_size is not None else train_config['test_size']
    rand_state = random_state if random_state is not None else train_config['random_state']

    # Set MLflow experiment
    experiment_name = params_config['mlflow']['experiment_name']
    mlflow.set_experiment(experiment_name)

    # Validate model type
    if model_type not in MODEL_REGISTRY:
        logger.error(f'Invalid model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}')
        raise ValueError(f'Model type must be one of {list(MODEL_REGISTRY.keys())}')

    # Get model class and name
    model_class = MODEL_REGISTRY[model_type]
    model_config = params_config['models'][model_type]
    model_name = model_config['model_class']

    # Determine whether to run in batch mode or single mode
    if batch:
        # Batch mode: run all configs from params.yml
        configs = model_config['configs']
        logger.info('-'*80)
        logger.info(f'Training {model_name} in BATCH MODE')
        logger.info(f'Experiment: {experiment_name}')
        logger.info(f'Number of configurations: {len(configs)}')
        logger.info('-'*80)
    else:
        # Single mode: build hyperparams from CLI arguments
        hyperparams = {}

        if n_estimators is not None:
            hyperparams['n_estimators'] = n_estimators
        if max_depth is not None:
            hyperparams['max_depth'] = max_depth
        if learning_rate is not None:
            hyperparams['learning_rate'] = learning_rate
        if n_jobs is not None:
            hyperparams['n_jobs'] = n_jobs
        if min_samples_leaf is not None:
            hyperparams['min_samples_leaf'] = min_samples_leaf
        if min_samples_split is not None:
            hyperparams['min_samples_split'] = min_samples_split
        if alpha is not None:
            hyperparams['alpha'] = alpha
        if l1_ratio is not None:
            hyperparams['l1_ratio'] = l1_ratio
        if max_iter is not None:
            hyperparams['max_iter'] = max_iter
        if kernel is not None:
            hyperparams['kernel'] = kernel
        if c is not None:
            hyperparams['C'] = c  # Note: CLI uses 'c' but sklearn uses 'C'
        if epsilon is not None:
            hyperparams['epsilon'] = epsilon
        if model_random_state is not None:
            hyperparams['random_state'] = model_random_state

        if not hyperparams:
            logger.error('No hyperparameters provided! Please specify at least one hyperparameter or use --batch.')
            logger.info(f'Example: python -m bike_sharing.modeling.train {model_type} --n-estimators 100 --max-depth 10')
            logger.info(f'Or: python -m bike_sharing.modeling.train {model_type} --batch')
            raise ValueError('At least one hyperparameter must be specified via CLI or use --batch mode')

        # Wrap single config in a list for consistent processing
        configs = [hyperparams]

        logger.info('-'*80)
        logger.info(f'Training {model_name} in SINGLE MODE')
        logger.info(f'Experiment: {experiment_name}')
        logger.info(f'Hyperparameters: {hyperparams}')
        logger.info('-'*80)

    # Load data
    try:
        bike_sharing_df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f'File not found: {input_file}')
        return

    # Prepare features and targets
    X = bike_sharing_df.drop(columns=['cnt_log'])
    y_casual = bike_sharing_df['casual_log']
    y_registered = bike_sharing_df['registered_log']

    # Fixed train_test_split: properly split all data together
    X_train, X_test, y_casual_train, y_casual_test = train_test_split(
        X, y_casual, test_size=test_split, random_state=rand_state
    )

    _, _, y_registered_train, y_registered_test = train_test_split(
        X, y_registered, test_size=test_split, random_state=rand_state
    )

    logger.info(f'Train Data: {len(X_train)} | Test Data: {len(X_test)}')

    # Train all configurations (either batch or single)
    total_runs = len(configs)
    logger.info(f'Initiating {total_runs} training run(s)')

    with tqdm(total=total_runs, desc=f'Training {model_name}') as pbar:
        for params in configs:
            train_model(
                model_type=model_class,
                model_name=model_name,
                params=params,
                X_train=X_train,
                X_test=X_test,
                y_casual_train=y_casual_train,
                y_casual_test=y_casual_test,
                y_registered_train=y_registered_train,
                y_registered_test=y_registered_test
            )
            pbar.update(1)

    logger.success(f'{model_name} training completed successfully.')
    logger.info('-'*80)

if __name__ == "__main__":
    app()
