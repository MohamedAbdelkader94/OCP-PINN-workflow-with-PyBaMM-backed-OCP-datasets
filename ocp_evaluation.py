
"""
ocp_evaluation.py
=================
Evaluation metrics, comparison tables, and publication-quality figures for 
OCP model benchmarking.

Computes:
- RMSE, MAE, R², max absolute error
- Training and inference time
- Derivative consistency (dU/dx)
- Boundary behavior and extrapolation assessment

Generates:
- OCP prediction vs data plots
- Residual/error plots
- Derivative comparison plots
- Free energy profiles (for Free-Energy PINN)
- Comprehensive comparison table
- Training loss curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 200
matplotlib.rcParams['savefig.bbox'] = 'tight'

import time
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Denormalization helpers (defined inline; ocp_data_processing_pybamm uses
# z-score for U and min-max for x)
# ---------------------------------------------------------------------------
def denormalize_U(U_norm, U_mean_or_min, U_std_or_max):
    """Inverse of z-score normalisation: U = U_norm * std + mean."""
    return U_norm * U_std_or_max + U_mean_or_min

def denormalize_x(x_norm, x_min, x_max):
    """Inverse of [-1,1] min-max normalisation: x = (x_norm+1)/2*(max-min)+min."""
    return (x_norm + 1.0) / 2.0 * (x_max - x_min) + x_min

def unique_color_cycle(n):
    palettes = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Dark2"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palettes.extend(list(cmap.colors))

    unique = []
    seen = set()
    for c in palettes:
        key = tuple(float(x) for x in c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if n > len(unique):
        raise ValueError(f"Requested {n} unique colors, but only {len(unique)} are available.")
    return unique[:n]


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute regression metrics for OCP prediction.
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted OCP values (same units, same scale).
    
    Returns
    -------
    metrics : dict
        RMSE, MAE, R2, max_error.
    """
    residuals = y_true - y_pred
    
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    max_err = np.max(np.abs(residuals))
    
    return {
        'RMSE (V)': rmse,
        'MAE (V)': mae,
        'R²': r2,
        'Max Error (V)': max_err,
    }


def compute_inference_time(model, x: np.ndarray, n_runs: int = 100) -> float:
    """
    Measure average inference time.
    
    Parameters
    ----------
    model : object with predict() method
    x : np.ndarray
        Input stoichiometry values.
    n_runs : int
        Number of repetitions for timing.
    
    Returns
    -------
    avg_time_ms : float
        Average inference time in milliseconds.
    """
    # Warm-up
    _ = model.predict(x)
    
    start = time.time()
    for _ in range(n_runs):
        _ = model.predict(x)
    total = time.time() - start
    
    return (total / n_runs) * 1000  # ms


# ============================================================================
# Comparison table
# ============================================================================

def build_comparison_table(
    models: list,
    model_names: list,
    x_test: np.ndarray,
    U_test: np.ndarray,
    training_times: list,
    data_for_normalize: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Build a comprehensive comparison table of all models.
    
    Parameters
    ----------
    models : list
        Trained models with predict() methods.
    model_names : list of str
        Names for each model.
    x_test : np.ndarray
        Test stoichiometry (normalized or raw, matching model expectations).
    U_test : np.ndarray
        Test OCP values (same normalization as model output).
    training_times : list of float
        Training time for each model in seconds.
    data_for_normalize : dict, optional
        If provided, denormalize predictions before computing metrics.
    
    Returns
    -------
    table : pd.DataFrame
    """
    rows = []
    
    for model, name, t_time in zip(models, model_names, training_times):
        U_pred = model.predict(x_test)
        
        # Denormalize if needed
        if data_for_normalize is not None and data_for_normalize.get('normalize', False):
            U_pred_raw = denormalize_U(U_pred, 
                                        data_for_normalize['U_min'], 
                                        data_for_normalize['U_max'])
            U_test_raw = denormalize_U(U_test.ravel(), 
                                        data_for_normalize['U_min'], 
                                        data_for_normalize['U_max'])
        else:
            U_pred_raw = U_pred
            U_test_raw = U_test.ravel()
        
        metrics = compute_metrics(U_test_raw, U_pred_raw)
        inf_time = compute_inference_time(model, x_test)
        
        row = {
            'Model': name,
            **metrics,
            'Train Time (s)': t_time,
            'Inference (ms)': inf_time,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# Plotting functions
# ============================================================================

def plot_ocp_comparison(
    x_data: np.ndarray,
    U_data: np.ndarray,
    models: list,
    model_names: list,
    x_plot: np.ndarray,
    electrode_name: str = "Electrode",
    save_path: Optional[str] = None,
    data_for_normalize: Optional[Dict] = None,
):
    """
    Plot OCP data and model predictions.
    
    Parameters
    ----------
    x_data, U_data : np.ndarray
        Full dataset for scatter overlay.
    models : list
        Trained models.
    model_names : list
        Model names for legend.
    x_plot : np.ndarray
        Dense x values for smooth prediction curves.
    electrode_name : str
        Title label.
    save_path : str, optional
        File path to save figure.
    data_for_normalize : dict, optional
        Normalization parameters.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Denormalize data if needed
    if data_for_normalize is not None and data_for_normalize.get('normalize', False):
        x_d = denormalize_x(x_data.ravel(), data_for_normalize['x_min'], data_for_normalize['x_max'])
        U_d = denormalize_U(U_data.ravel(), data_for_normalize['U_min'], data_for_normalize['U_max'])
        x_p_raw = denormalize_x(x_plot.ravel(), data_for_normalize['x_min'], data_for_normalize['x_max'])
    else:
        x_d = x_data.ravel()
        U_d = U_data.ravel()
        x_p_raw = x_plot.ravel()
    
    # Data points
    ax.scatter(x_d, U_d, s=8, c='gray', alpha=0.4, label='Literature data', zorder=1)
    
    # Model predictions
    colors = unique_color_cycle(len(models))
    linestyles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]

    for i, (model, name) in enumerate(zip(models, model_names)):
        U_pred = model.predict(x_plot)
        if data_for_normalize is not None and data_for_normalize.get('normalize', False):
            U_pred = denormalize_U(U_pred, data_for_normalize['U_min'], data_for_normalize['U_max'])

        ax.plot(
            x_p_raw,
            U_pred,
            color=colors[i],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.8,
            label=name,
            zorder=2 + i,
        )
    
    ax.set_xlabel(r'Stoichiometry $x$')
    ax.set_ylabel(r'OCP $U$ (V vs Li/Li$^+$)')
    ax.set_title(f'{electrode_name} — OCP Model Comparison')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_residuals(
    x_test: np.ndarray,
    U_test: np.ndarray,
    models: list,
    model_names: list,
    electrode_name: str = "Electrode",
    save_path: Optional[str] = None,
    data_for_normalize: Optional[Dict] = None,
):
    """Plot residual errors for each model on the test set."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    if data_for_normalize is not None and data_for_normalize.get('normalize', False):
        x_raw = denormalize_x(x_test.ravel(), data_for_normalize['x_min'], data_for_normalize['x_max'])
        U_raw = denormalize_U(U_test.ravel(), data_for_normalize['U_min'], data_for_normalize['U_max'])
    else:
        x_raw = x_test.ravel()
        U_raw = U_test.ravel()
    
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#264653', '#F4A261']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        U_pred = model.predict(x_test)
        if data_for_normalize is not None and data_for_normalize.get('normalize', False):
            U_pred = denormalize_U(U_pred, data_for_normalize['U_min'], data_for_normalize['U_max'])
        
        residuals = U_raw - U_pred
        ax.scatter(x_raw, residuals * 1000, s=15, alpha=0.6,
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   label=name)
    
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel(r'Stoichiometry $x$')
    ax.set_ylabel('Residual (mV)')
    ax.set_title(f'{electrode_name} — Prediction Residuals')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_derivative_comparison(
    x_plot: np.ndarray,
    pinn_model,
    electrode_name: str = "Electrode",
    true_derivative_func=None,
    save_path: Optional[str] = None,
    data_for_normalize: Optional[Dict] = None,
):
    """
    Plot dU/dx from the PINN and compare with numerical derivative of data.
    
    Parameters
    ----------
    x_plot : np.ndarray
        Stoichiometry values for plotting.
    pinn_model : object with predict_with_derivatives() method
    electrode_name : str
    true_derivative_func : callable, optional
        If provided, compute the true analytical derivative for comparison.
    save_path : str, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    U_pred, dU_dx = pinn_model.predict_with_derivatives(x_plot)
    
    if data_for_normalize is not None and data_for_normalize.get('normalize', False):
        x_raw = denormalize_x(x_plot.ravel(), data_for_normalize['x_min'], data_for_normalize['x_max'])
        # Scale derivative: dU/dx_raw = dU_norm/dx_norm * (U_max-U_min)/(x_max-x_min)
        scale = (data_for_normalize['U_max'] - data_for_normalize['U_min']) / \
                (data_for_normalize['x_max'] - data_for_normalize['x_min'] + 1e-10)
        dU_dx_raw = dU_dx * scale
    else:
        x_raw = x_plot.ravel()
        dU_dx_raw = dU_dx
    
    ax.plot(x_raw, dU_dx_raw, 'r-', linewidth=1.5, label='PINN dU/dx')
    
    ax.set_xlabel(r'Stoichiometry $x$')
    ax.set_ylabel(r'$dU/dx$ (V)')
    ax.set_title(f'{electrode_name} — OCP Derivative from PINN')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_training_loss(
    train_losses: list,
    val_losses: list,
    model_name: str = "PINN",
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, 'b-', alpha=0.7, linewidth=0.8, label='Train loss')
    ax.semilogy(epochs, val_losses, 'r-', alpha=0.7, linewidth=0.8, label='Validation loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} — Training Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_free_energy(
    x_plot: np.ndarray,
    fe_pinn_model,
    electrode_name: str = "Electrode",
    save_path: Optional[str] = None,
):
    """Plot the learned free energy profile G(x) from FreeEnergyPINN."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    G = fe_pinn_model.predict_free_energy(x_plot)
    
    ax.plot(x_plot.ravel(), G, 'b-', linewidth=1.5)
    ax.set_xlabel(r'Stoichiometry $x$')
    ax.set_ylabel(r'$G(x)$ (learned, a.u.)')
    ax.set_title(f'{electrode_name} — Learned Free Energy Profile')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


# ============================================================================
# Main evaluation pipeline
# ============================================================================

def run_full_evaluation(
    electrode_name: str,
    data: Dict,
    pinn_direct,
    pinn_fe,
    baselines: list,
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run the complete evaluation pipeline.
    
    Parameters
    ----------
    electrode_name : str
        Name of the electrode for labeling.
    data : dict
        Preprocessed data.
    pinn_direct : DirectOCPPINN
        Trained direct PINN model.
    pinn_fe : FreeEnergyPINN
        Trained free-energy PINN model.
    baselines : list
        Trained baseline models.
    output_dir : str
        Directory for saving figures.
    
    Returns
    -------
    comparison : pd.DataFrame
        Comparison table with metrics for all models.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    x_test = data['x_test']
    U_test = data['U_test']
    
    # Dense x for plotting
    x_plot = np.linspace(0.0, 1.0, 500).reshape(-1, 1)
    
    # Collect all models
    all_models = [pinn_direct, pinn_fe] + baselines
    all_names = ['PINN (Direct)', 'PINN (Free-Energy)'] + [b.name for b in baselines]
    all_times = [pinn_direct.training_time, pinn_fe.training_time] + \
                [b.training_time for b in baselines]
    
    # Build comparison table
    table = build_comparison_table(
        all_models, all_names, x_test, U_test, all_times,
        data_for_normalize=data
    )
    
    print(f"\n{'='*70}")
    print(f"  {electrode_name} — Model Comparison")
    print(f"{'='*70}")
    print(table.to_string(index=False))
    
    # Save table
    table.to_csv(f"{output_dir}/{electrode_name.lower().replace(' ', '_')}_comparison.csv", 
                 index=False)
    
    # Generate figures
    # 1. OCP comparison
    plot_ocp_comparison(
        data['x_train'], data['U_train'],
        all_models, all_names, x_plot,
        electrode_name=electrode_name,
        save_path=f"{output_dir}/{electrode_name.lower().replace(' ', '_')}_ocp_comparison.png",
        data_for_normalize=data
    )
    
    # 2. Residuals
    plot_residuals(
        x_test, U_test, all_models, all_names,
        electrode_name=electrode_name,
        save_path=f"{output_dir}/{electrode_name.lower().replace(' ', '_')}_residuals.png",
        data_for_normalize=data
    )
    
    # 3. Derivative plot (PINN direct only)
    plot_derivative_comparison(
        x_plot, pinn_direct,
        electrode_name=electrode_name,
        save_path=f"{output_dir}/{electrode_name.lower().replace(' ', '_')}_derivative.png",
        data_for_normalize=data
    )
    
    # 4. Training loss
    plot_training_loss(
        pinn_direct.train_losses, pinn_direct.val_losses,
        model_name=f"{electrode_name} — Direct PINN",
        save_path=f"{output_dir}/{electrode_name.lower().replace(' ', '_')}_training_loss.png"
    )
    
    # 5. Free energy profile
    plot_free_energy(
        x_plot, pinn_fe,
        electrode_name=electrode_name,
        save_path=f"{output_dir}/{electrode_name.lower().replace(' ', '_')}_free_energy.png"
    )
    
    return table


if __name__ == "__main__":
    print("Evaluation module loaded. Use run_full_evaluation() for the full pipeline.")

