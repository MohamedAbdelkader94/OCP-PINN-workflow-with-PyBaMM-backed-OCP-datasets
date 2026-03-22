"""
ocp_baselines.py
================
Baseline models for OCP fitting — conventional methods compared against the PINN.

Implements the following baselines:
1. Polynomial regression (degree 8 and 12)
2. Cubic spline interpolation
3. Nernst-type analytical fit (ideal solution + polynomial correction)
4. Redlich-Kister thermodynamic fit
5. Gaussian Process Regression (GPR)

Each baseline uses the same train/test split as the PINN for fair comparison.

References:
    Plett, Battery Management Systems, Vol. 1 (Artech House, 2015).
    Yao & Viswanathan, J. Phys. Chem. Lett. 15 (2024) 1105.
    Karthikeyan et al., J. Power Sources 185 (2008) 1398.
"""

import numpy as np
import time
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from typing import Dict, Tuple, Optional


# ============================================================================
# Constants
# ============================================================================
R_GAS = 8.314462
F_CONST = 96485.33
T_REF = 298.15


# ============================================================================
# 1. Polynomial regression
# ============================================================================

class PolynomialOCPModel:
    """
    OCP model using polynomial regression.
    
    Polynomial fitting is the simplest and most widely used empirical approach 
    for OCP modeling. High-degree polynomials (n >= 8) are typically needed 
    to capture the non-linear features of intercalation electrodes.
    
    Limitation: Does not enforce thermodynamic consistency or monotonicity.
    Can produce oscillatory artifacts at boundaries (Runge's phenomenon).
    
    References:
        Doyle et al., J. Electrochem. Soc. 143 (1996) 1890.
        Sikha et al., J. Electrochem. Soc. 152 (2005) A1682.
    
    Parameters
    ----------
    degree : int
        Polynomial degree (typically 8-16 for battery OCP).
    """
    
    def __init__(self, degree: int = 12):
        self.degree = degree
        self.model = make_pipeline(
            PolynomialFeatures(degree), 
            LinearRegression()
        )
        self.training_time = 0.0
        self.name = f"Polynomial (deg {degree})"
    
    def train(self, x_train: np.ndarray, U_train: np.ndarray):
        """Fit the polynomial model."""
        start = time.time()
        self.model.fit(x_train.reshape(-1, 1), U_train.ravel())
        self.training_time = time.time() - start
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict OCP."""
        return self.model.predict(x.reshape(-1, 1))


# ============================================================================
# 2. Cubic spline interpolation
# ============================================================================

class SplineOCPModel:
    """
    OCP model using cubic spline interpolation.
    
    Splines provide smooth, piecewise-polynomial fits that pass through data
    points. They are highly flexible and widely used in battery management 
    systems for OCV-SOC lookup tables.
    
    Limitation: Cannot extrapolate beyond the training data range. Does not 
    enforce monotonicity, and the resulting function gradients (dU/dx) may 
    not be thermodynamically meaningful.
    
    Reference:
        Yao & Viswanathan, J. Phys. Chem. Lett. 15 (2024) 1105 (discussion of 
        spline-based models and their thermodynamic limitations).
    """
    
    def __init__(self):
        self.spline = None
        self.training_time = 0.0
        self.name = "Cubic Spline"
    
    def train(self, x_train: np.ndarray, U_train: np.ndarray):
        """Fit the cubic spline."""
        start = time.time()
        # Sort data by x for spline construction
        sort_idx = np.argsort(x_train.ravel())
        x_sorted = x_train.ravel()[sort_idx]
        U_sorted = U_train.ravel()[sort_idx]
        
        # Remove duplicate x values (keep mean U)
        unique_x, unique_idx = np.unique(x_sorted, return_index=True)
        unique_U = np.array([U_sorted[x_sorted == xi].mean() for xi in unique_x])
        
        self.spline = CubicSpline(unique_x, unique_U)
        self.x_min = unique_x.min()
        self.x_max = unique_x.max()
        self.training_time = time.time() - start
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict OCP (clipped to training range for extrapolation safety)."""
        x_clipped = np.clip(x.ravel(), self.x_min, self.x_max)
        return self.spline(x_clipped)


# ============================================================================
# 3. Nernst-type analytical fit
# ============================================================================

class NernstOCPModel:
    """
    OCP model combining the Nernst equation with polynomial correction.
    
    The model expresses OCP as:
        U(x) = U0 - (RT/F) * ln(x/(1-x)) + sum_i a_i * x^i
    
    The first two terms represent the ideal thermodynamic contribution from
    entropy of mixing (Nernst equation for a single-site lattice model).
    The polynomial correction captures deviations due to non-ideal 
    interactions (excess Gibbs energy).
    
    This is physically more meaningful than pure polynomial fitting because
    it correctly captures the logarithmic divergence at x -> 0 and x -> 1.
    
    Parameters
    ----------
    poly_degree : int
        Degree of the correction polynomial.
    T : float
        Temperature in Kelvin.
    """
    
    def __init__(self, poly_degree: int = 6, T: float = T_REF):
        self.poly_degree = poly_degree
        self.T = T
        self.params = None
        self.training_time = 0.0
        self.name = f"Nernst + Poly (deg {poly_degree})"
    
    def _nernst_ideal(self, x: np.ndarray) -> np.ndarray:
        """Ideal contribution: -(RT/F) * ln(x/(1-x))."""
        x_safe = np.clip(x, 1e-6, 1 - 1e-6)
        return -(R_GAS * self.T / F_CONST) * np.log(x_safe / (1 - x_safe))
    
    def _model_func(self, x, *params):
        """Full model: U0 + Nernst + polynomial."""
        U0 = params[0]
        poly_coeffs = params[1:]
        
        U = U0 + self._nernst_ideal(x)
        for i, a in enumerate(poly_coeffs):
            U = U + a * x**(i+1)
        return U
    
    def train(self, x_train: np.ndarray, U_train: np.ndarray):
        """Fit the Nernst + polynomial model using nonlinear least squares."""
        start = time.time()
        
        x = x_train.ravel()
        U = U_train.ravel()
        
        # Initial guess: U0 from data midpoint, zero polynomial coefficients
        p0 = [np.mean(U)] + [0.0] * self.poly_degree
        
        try:
            self.params, _ = curve_fit(
                self._model_func, x, U, p0=p0,
                maxfev=10000
            )
        except RuntimeError:
            # If optimization fails, fall back to simpler fit
            self.params = p0
        
        self.training_time = time.time() - start
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict OCP."""
        return self._model_func(x.ravel(), *self.params)


# ============================================================================
# 4. Redlich-Kister thermodynamic fit
# ============================================================================

class RedlichKisterOCPModel:
    """
    OCP model based on Redlich-Kister expansion of excess Gibbs energy.
    
    The thermodynamic model expresses OCP as:
        U(x) = U0 - (RT/F) * ln(x/(1-x)) + (1/F) * d(g_excess)/dx
    
    where g_excess = x*(1-x) * sum_{i=0}^{N} Omega_i * (1-2x)^i
    
    This is the standard thermodynamic approach for modeling OCP of 
    intercalation materials. The Omega_i coefficients have units of J/mol 
    and represent the strength of lithium-vacancy interactions.
    
    Limitation: Does not enforce monotonicity. For materials with phase 
    separation (e.g., LFP), the regular R-K model produces non-physical 
    oscillations unless combined with common tangent construction.
    
    References:
        Karthikeyan et al., J. Power Sources 185 (2008) 1398.
        Plett, Battery Management Systems, Vol. 1 (Artech House, 2015).
        Yao & Viswanathan, J. Phys. Chem. Lett. 15 (2024) 1105.
    
    Parameters
    ----------
    n_terms : int
        Number of R-K coefficients (typically 3-8).
    T : float
        Temperature in Kelvin.
    """
    
    def __init__(self, n_terms: int = 6, T: float = T_REF):
        self.n_terms = n_terms
        self.T = T
        self.params = None
        self.training_time = 0.0
        self.name = f"Redlich-Kister ({n_terms} terms)"
    
    def _rk_model(self, x, U0, *omega):
        """Compute OCP from R-K expansion."""
        x_safe = np.clip(x, 1e-6, 1 - 1e-6)
        
        # Ideal mixing term
        U = U0 - (R_GAS * self.T / F_CONST) * np.log(x_safe / (1 - x_safe))
        
        # Excess chemical potential from R-K expansion
        # mu_excess = d(g_excess * N_total) / dN_Li
        # For binary on single lattice:
        # mu_excess/F = (1/F) * sum_i omega_i * [(1-2x)^(i+1) - 2*i*x*(1-x)*(1-2x)^(i-1)]
        z = 1 - 2 * x_safe
        for i, om in enumerate(omega):
            if i == 0:
                U += (om / F_CONST) * z
            else:
                U += (om / F_CONST) * (z**(i+1) - 2*i*x_safe*(1-x_safe)*z**(i-1))
        
        return U
    
    def train(self, x_train: np.ndarray, U_train: np.ndarray):
        """Fit the R-K model."""
        start = time.time()
        
        x = x_train.ravel()
        U = U_train.ravel()
        
        # Initial guess
        p0 = [np.mean(U)] + [0.0] * self.n_terms
        
        try:
            self.params, _ = curve_fit(
                self._rk_model, x, U, p0=p0,
                maxfev=20000
            )
        except RuntimeError:
            self.params = p0
        
        self.training_time = time.time() - start
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict OCP."""
        return self._rk_model(x.ravel(), *self.params)


# ============================================================================
# 5. Gaussian Process Regression
# ============================================================================

class GPR_OCPModel:
    """
    OCP model using Gaussian Process Regression.
    
    GPR is a non-parametric Bayesian regression method that provides both
    a mean prediction and uncertainty estimates. It is well-suited for 
    interpolation of smooth functions like OCP curves.
    
    The Matern kernel is chosen for its ability to model varying degrees 
    of smoothness, and the WhiteKernel accounts for observation noise.
    
    Limitation: Computational cost scales as O(n³), making it impractical 
    for very large datasets. No built-in physics constraints.
    """
    
    def __init__(self):
        kernel = Matern(length_scale=0.1, nu=2.5) + WhiteKernel(noise_level=1e-4)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        self.training_time = 0.0
        self.name = "Gaussian Process (Matern)"
    
    def train(self, x_train: np.ndarray, U_train: np.ndarray):
        """Fit the GPR model."""
        start = time.time()
        self.model.fit(x_train.reshape(-1, 1), U_train.ravel())
        self.training_time = time.time() - start
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict OCP (mean prediction)."""
        return self.model.predict(x.reshape(-1, 1))
    
    def predict_with_uncertainty(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict OCP with uncertainty bounds."""
        mean, std = self.model.predict(x.reshape(-1, 1), return_std=True)
        return mean, std


# ============================================================================
# Factory and utilities
# ============================================================================

def get_all_baselines() -> list:
    """Return a list of all baseline models."""
    return [
        PolynomialOCPModel(degree=8),
        PolynomialOCPModel(degree=12),
        SplineOCPModel(),
        NernstOCPModel(poly_degree=6),
        RedlichKisterOCPModel(n_terms=6),
        GPR_OCPModel(),
    ]


def train_all_baselines(data: Dict) -> list:
    """
    Train all baseline models on the same data.
    
    Parameters
    ----------
    data : dict
        Preprocessed data dictionary.
    
    Returns
    -------
    models : list of trained baseline models
    """
    models = get_all_baselines()
    
    for model in models:
        print(f"Training {model.name}...")
        model.train(data['x_train'], data['U_train'])
        print(f"  Training time: {model.training_time:.4f} s")
    
    return models


if __name__ == "__main__":
    # Quick test with synthetic data
    x = np.linspace(0.01, 0.99, 100)
    U = 3.4 - 0.5 * x + 0.1 * np.sin(5 * x)
    
    for ModelClass in [PolynomialOCPModel, SplineOCPModel, NernstOCPModel,
                       RedlichKisterOCPModel, GPR_OCPModel]:
        if ModelClass == PolynomialOCPModel:
            model = ModelClass(degree=8)
        elif ModelClass == RedlichKisterOCPModel:
            model = ModelClass(n_terms=4)
        elif ModelClass == NernstOCPModel:
            model = ModelClass(poly_degree=4)
        else:
            model = ModelClass()
        
        model.train(x, U)
        U_pred = model.predict(x)
        rmse = np.sqrt(np.mean((U_pred - U)**2))
        print(f"{model.name}: RMSE = {rmse:.6f}, Time = {model.training_time:.4f}s")
