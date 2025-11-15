from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from tqdm import tqdm
import typer

from bike_sharing.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def plot_drift_distribution(
    df_base: pd.DataFrame, 
    df_drifted: pd.DataFrame, 
    feature: str, 
    output_path: Path
):
    """
    Genera y guarda un gráfico de densidad (KDE) para visualizar el cambio de distribución (Drift).
    
    Args:
        df_base (pd.DataFrame): Datos de referencia (Línea Base).
        df_drifted (pd.DataFrame): Datos con drift (Sintéticos/Monitoreo).
        feature (str): Nombre de la columna a graficar.
        output_path (Path): Ruta donde se guardará el archivo .png.
    """
        
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.kdeplot(
        df_base[feature], 
        fill=True, 
        color='blue',
        label='Datos Originales',
        ax=ax
    )
    
    sns.kdeplot(
        df_drifted[feature], 
        fill=True, 
        color='red', 
        label='Datos Sintéticos',
        ax=ax
    )

    ax.set_title(f'Comparativa: Distribución Original & Sintética ({feature})', fontsize=16)
    ax.set_xlabel(f'{feature}', fontsize=14)
    ax.set_ylabel('Densidad', fontsize=14)
    ax.grid(True)
    
#    base_mean = df_base[feature].mean()
#    drift_mean = df_drifted[feature].mean()
#    ax.axvline(base_mean, color='blue', linestyle=':', linewidth=1.5, label=f'Media Base: {base_mean:.3f}')
#    ax.axvline(drift_mean, color='red', linestyle=':', linewidth=1.5, label=f'Media Drift: {drift_mean:.3f}')
#    ax.legend(fontsize=12, loc='upper right')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)

    logger.success(f'Gráfico de Data Drift generado & guardado en: {output_path.name}')


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):

    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
