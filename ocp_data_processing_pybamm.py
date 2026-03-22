"""
ocp_data_processing_pybamm.py
=============================
Data loading, preprocessing, and dataset generation for OCP/OCV modeling,
using PyBaMM's built-in parameter sets as the source of OCP functions.

This keeps the same external logic as the original ocp_data_processing.py:
- generate_ocp_dataset(...)
- preprocess_data(...)

But instead of hard-coded analytical surrogate expressions, the clean OCP curve is
sampled directly from the OCP functions bundled in PyBaMM parameter sets.

Supported datasets
------------------
- Graphite anode half-cell OCP from the Chen2020 parameter set
- NMC811 cathode half-cell OCP from the Chen2020 parameter set
- LFP cathode half-cell OCP from the Prada2013 parameter set

Notes
-----
- PyBaMM exposes built-in parameter sets through ``pybamm.parameter_sets`` and
  loads them via ``pybamm.ParameterValues(<name>)``.
- In parameter dictionaries, OCP functions are stored under keys such as
  ``"Negative electrode OCP [V]"`` and ``"Positive electrode OCP [V]"``.
- This file assumes the returned OCP functions accept array-like stoichiometry
  inputs on (0, 1) and return voltage values in volts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Callable

try:
    import pybamm
except ImportError as exc:
    raise ImportError(
        "PyBaMM is required for ocp_data_processing_pybamm.py. "
        "Install it with: pip install pybamm"
    ) from exc


# ============================================================================
# PyBaMM-backed OCP accessors
# ============================================================================

def _get_pybamm_ocp_function(electrode: str) -> Tuple[Callable, str, str, str]:
    """
    Return the PyBaMM OCP function and metadata for a requested electrode.

    Returns
    -------
    ocp_fn : callable
        OCP function from the PyBaMM parameter set.
    parameter_set : str
        Name of the PyBaMM parameter set.
    parameter_key : str
        Dictionary key used in the parameter set.
    electrode_role : str
        Either 'anode' or 'cathode'.
    """
    electrode = electrode.lower()

    if electrode == "graphite":
        parameter_set = "Chen2020"
        parameter_key = "Negative electrode OCP [V]"
        electrode_role = "anode"
    elif electrode == "nmc811":
        parameter_set = "Chen2020"
        parameter_key = "Positive electrode OCP [V]"
        electrode_role = "cathode"
    elif electrode == "lfp":
        parameter_set = "Prada2013"
        parameter_key = "Positive electrode OCP [V]"
        electrode_role = "cathode"
    else:
        raise ValueError(
            f"Unknown electrode '{electrode}'. Expected one of: graphite, nmc811, lfp"
        )

    params = pybamm.ParameterValues(parameter_set)
    ocp_fn = params[parameter_key]
    return ocp_fn, parameter_set, parameter_key, electrode_role



def pybamm_ocp(electrode: str, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the PyBaMM OCP function for the selected electrode.

    Parameters
    ----------
    electrode : str
        'graphite', 'nmc811', or 'lfp'.
    x : np.ndarray
        Stoichiometry values in the open interval (0, 1).

    Returns
    -------
    np.ndarray
        OCP values in volts.
    """
    ocp_fn, _, _, _ = _get_pybamm_ocp_function(electrode)
    x = np.asarray(x, dtype=float)
    x_safe = np.clip(x, 1e-6, 1 - 1e-6)

    # PyBaMM bundled OCP functions are ordinary Python-callable functions in the
    # parameter dictionary. For numpy-array input they typically return ndarray-like
    # output; cast explicitly for downstream stability.
    U = ocp_fn(x_safe)
    return np.asarray(U, dtype=float)


# ============================================================================
# Dataset generation
# ============================================================================

def generate_ocp_dataset(
    electrode: str = "graphite",
    n_points: int = 200,
    noise_std: float = 0.003,
    seed: int = 42,
    x_range: Tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Generate an OCP dataset using PyBaMM's built-in OCP functions.

    The returned dataframe keeps the same columns as the original code so the
    rest of the pipeline can remain unchanged.

    Parameters
    ----------
    electrode : str
        One of 'graphite', 'nmc811', or 'lfp'.
    n_points : int
        Number of data points.
    noise_std : float
        Standard deviation of additive Gaussian noise [V].
    seed : int
        Random seed.
    x_range : tuple, optional
        Stoichiometry range. If None, uses practical defaults for each electrode.

    Returns
    -------
    pd.DataFrame
        Columns:
        - x
        - U_measured
        - U_true
        - source
        - parameter_set
        - parameter_key
        - electrode_role
    """
    rng = np.random.RandomState(seed)
    electrode = electrode.lower()

    if electrode == "graphite":
        if x_range is None:
            x_range = (0.01, 0.99)
    elif electrode == "nmc811":
        if x_range is None:
            x_range = (0.25, 0.95)
    elif electrode == "lfp":
        if x_range is None:
            x_range = (0.01, 0.99)
    else:
        raise ValueError(
            f"Unknown electrode '{electrode}'. Expected one of: graphite, nmc811, lfp"
        )

    ocp_fn, parameter_set, parameter_key, electrode_role = _get_pybamm_ocp_function(electrode)

    x = np.linspace(x_range[0], x_range[1], n_points, dtype=float)
    U_true = np.asarray(ocp_fn(np.clip(x, 1e-6, 1 - 1e-6)), dtype=float)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_points)
    U_measured = U_true + noise

    df = pd.DataFrame(
        {
            "x": x,
            "U_measured": U_measured,
            "U_true": U_true,
            "source": "PyBaMM bundled parameter set",
            "parameter_set": parameter_set,
            "parameter_key": parameter_key,
            "electrode_role": electrode_role,
        }
    )
    return df


# ============================================================================
# Data preprocessing (kept same logic / API)
# ============================================================================

def preprocess_data(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    normalize: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Preprocess OCP data for model training.

    Keeps the original output structure used by the existing PINN / baseline code.
    """
    rng = np.random.RandomState(seed)

    x = df["x"].values.reshape(-1, 1)
    U = df["U_measured"].values.reshape(-1, 1)
    U_true = df["U_true"].values.reshape(-1, 1)

    n = len(df)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    x_train, U_train = x[train_idx], U[train_idx]
    x_val, U_val = x[val_idx], U[val_idx]
    x_test, U_test = x[test_idx], U[test_idx]
    U_true_test = U_true[test_idx]

    out = {
        "x_train": x_train,
        "U_train": U_train,
        "x_val": x_val,
        "U_val": U_val,
        "x_test": x_test,
        "U_test": U_test,
        "U_true_test": U_true_test,
        "x_all": x,
        "U_all": U,
        "U_true_all": U_true,
        "df": df.copy(),
    }

    if normalize:
        x_min, x_max = x_train.min(), x_train.max()
        U_mean, U_std = U_train.mean(), U_train.std()
        if U_std < 1e-12:
            U_std = 1.0

        def x_norm(arr):
            return 2.0 * (arr - x_min) / (x_max - x_min) - 1.0

        def U_norm(arr):
            return (arr - U_mean) / U_std

        out.update(
            {
                "x_train_n": x_norm(x_train),
                "U_train_n": U_norm(U_train),
                "x_val_n": x_norm(x_val),
                "U_val_n": U_norm(U_val),
                "x_test_n": x_norm(x_test),
                "U_test_n": U_norm(U_test),
                "x_all_n": x_norm(x),
                "U_all_n": U_norm(U),
                "U_true_all_n": U_norm(U_true),
                "norm": {
                    "x_min": x_min,
                    "x_max": x_max,
                    "U_mean": U_mean,
                    "U_std": U_std,
                },
            }
        )
    else:
        out["norm"] = None

    return out
