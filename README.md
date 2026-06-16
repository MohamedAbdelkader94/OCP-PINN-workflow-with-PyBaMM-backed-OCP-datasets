# OCP-PINN: Physics-Informed Neural Networks for Battery Open-Circuit Potential Modeling

> Learn thermodynamically consistent OCP curves for Li-ion battery electrodes using PINNs, benchmarked against classical baselines and driven by real PyBaMM parameter sets.

---

## Overview

This repository implements a research framework for modeling the **Open-Circuit Potential (OCP)** of Li-ion battery electrodes as a function of lithium stoichiometry *x*. Rather than relying on hand-tuned empirical expressions, two **Physics-Informed Neural Network (PINN)** formulations are trained on data sampled directly from [PyBaMM](https://www.pybamm.org/) built-in parameter sets and evaluated against six classical baseline methods.

The framework supports three electrode materials out-of-the-box:

| Electrode | Chemistry | PyBaMM source |
|-----------|-----------|---------------|
| Graphite | Li$_x$C$_6$ | `Chen2020` |
| NMC811 | Li$_x$Ni$_{0.8}$Mn$_{0.1}$Co$_{0.1}$O$_2$ | `Chen2020` |
| LFP | Li$_x$FePO$_4$ | `Prada2013` |

---

## Key Features

- **Two PINN formulations**
  - *Direct OCP PINN* — learns U(x) directly with physics regularisation terms (monotonicity, smoothness, boundary behaviour)
  - *Free-Energy PINN* — learns the Gibbs free energy G(x) and derives U(x) = −(1/F) dG/dx via automatic differentiation, inherently enforcing thermodynamic consistency

- **Six classical baselines** for fair benchmarking
  - Polynomial regression (degree 8 & 12)
  - Cubic spline interpolation
  - Nernst + polynomial correction
  - Redlich-Kister thermodynamic expansion
  - Gaussian Process Regression (Matérn kernel)

- **PyBaMM-backed data generation** — clean OCP curves are sampled directly from validated literature parameter sets; configurable Gaussian noise simulates experimental scatter

- **Comprehensive evaluation** — RMSE, MAE, R², max absolute error, training time, and inference latency for every model, per electrode

- **Publication-quality figures** — OCP comparison, residual plots, dU/dx derivatives, free-energy profiles, and training loss curves

---

## Repository Structure

```
ocp-pinn/
├── ocp_pinn_model.py              # PINN architectures (Direct & Free-Energy)
├── ocp_baselines.py               # All six classical baseline models
├── ocp_data_processing_pybamm.py  # PyBaMM dataset generation & preprocessing
├── ocp_evaluation.py              # Metrics, plots, and evaluation pipeline
├── run_ocp_pinn_pybamm.py         # Main execution script
├── ocp_pybamm_workflow.ipynb      # Interactive Jupyter notebook
├── results_pybamm/                # Output directory (auto-created)
│   ├── data/                      # Generated CSVs per electrode
│   ├── figures/                   # All plots
│   └── combined_comparison.csv    # Aggregated metrics table
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

**Requirements:** Python ≥ 3.9

```bash
git clone https://github.com/<your-username>/ocp-pinn.git
cd ocp-pinn
pip install -r requirements.txt
```

**Core dependencies:**

```
torch>=2.0
pybamm>=23.0
numpy
pandas
scipy
scikit-learn
matplotlib
jupyter
```

---

## Quick Start

### Run the full pipeline (all three electrodes)

```bash
python run_ocp_pinn_pybamm.py
```

Results are saved to `results_pybamm/`. Training 3 electrodes × 2 PINNs at 5 000 epochs takes roughly 10–20 minutes on CPU; significantly faster with a GPU.

### Interactive notebook

```bash
jupyter notebook ocp_pybamm_workflow.ipynb
```

### Use a single electrode programmatically

```python
from ocp_data_processing_pybamm import generate_ocp_dataset, preprocess_data
from ocp_pinn_model import DirectOCPPINN

df   = generate_ocp_dataset(electrode='nmc811', n_points=200, noise_std=0.003)
data = preprocess_data(df, train_frac=0.7, val_frac=0.15, normalize=True)

pinn = DirectOCPPINN(hidden_layers=[64, 64, 64, 64], electrode_type='cathode')
pinn.train(data, n_epochs=5000)

import numpy as np
x = np.linspace(0.25, 0.95, 300)
U = pinn.predict(x)
```

---

## Physics Background

The governing equation for OCP of an intercalation electrode is:

$$U(x) = U_0 - \frac{RT}{F}\ln\frac{x}{1-x} + U_\text{excess}(x)$$

where the ideal mixing term comes from the Nernst equation and $U_\text{excess}$ captures non-ideal lithium–vacancy interactions via the excess Gibbs energy:

$$U_\text{excess}(x) = \frac{1}{F}\frac{\partial g_\text{excess}}{\partial x}$$

The **Free-Energy PINN** encodes this structure directly — the network outputs G(x), and U(x) is obtained by automatic differentiation. This guarantees thermodynamic self-consistency by construction.

---

## Results (example)

All metrics are computed on a held-out test set (15% of data) in physical units (V):

| Model | RMSE (V) | MAE (V) | R² | Train Time (s) |
|-------|----------|---------|----|----------------|
| PINN (Direct) | — | — | — | — |
| PINN (Free-Energy) | — | — | — | — |
| Polynomial (deg 12) | — | — | — | — |
| Cubic Spline | — | — | — | — |
| Nernst + Poly | — | — | — | — |
| Redlich-Kister | — | — | — | — |
| Gaussian Process | — | — | — | — |

*Run the pipeline to populate this table with your results.*

---

## References

- Raissi et al., *J. Comput. Phys.* **378** (2019) 686 — original PINN formulation
- Yao & Viswanathan, *J. Phys. Chem. Lett.* **15** (2024) 1105 — thermodynamic OCP modeling
- Ferguson & Bazant, *J. Electrochem. Soc.* **159** (2012) A1967 — free energy of intercalation
- Karthikeyan et al., *J. Power Sources* **185** (2008) 1398 — Redlich-Kister for batteries
- Chen et al., *J. Electrochem. Soc.* **167** (2020) 080534 — Chen2020 parameter set
- Prada et al., *J. Electrochem. Soc.* **160** (2013) A616 — Prada2013 (LFP) parameter set

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
