"""
run_ocp_pinn_pybamm.py
======================
Main execution script for the OCP-PINN research workflow.

This version keeps the same pipeline logic as run_ocp_pinn.py, but replaces the
synthetic analytical dataset generator with a PyBaMM-backed generator that samples
OCP functions directly from built-in parameter sets.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from ocp_data_processing_pybamm import generate_ocp_dataset, preprocess_data
from ocp_pinn_model import DirectOCPPINN, FreeEnergyPINN
from ocp_baselines import train_all_baselines
from ocp_evaluation import run_full_evaluation


def run_electrode(electrode: str, electrode_label: str, electrode_type: str,
                  output_dir: str, n_epochs: int = 5000):
    print(f"\n{'#'*70}")
    print(f"  Processing: {electrode_label}")
    print(f"{'#'*70}")

    print("\n[1/5] Generating OCP dataset from PyBaMM...")
    df = generate_ocp_dataset(electrode=electrode, n_points=200, noise_std=0.003)
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    df.to_csv(f"{output_dir}/data/{electrode}_ocp.csv", index=False)
    print(f"  Data: {len(df)} points, U in [{df['U_true'].min():.3f}, {df['U_true'].max():.3f}] V")
    print(f"  Source: {df['parameter_set'].iloc[0]} / {df['parameter_key'].iloc[0]}")

    print("\n[2/5] Preprocessing data...")
    data = preprocess_data(df, train_frac=0.7, val_frac=0.15, normalize=True)
    print(f"  Train: {len(data['x_train'])}, Val: {len(data['x_val'])}, Test: {len(data['x_test'])}")

    print(f"\n[3/5] Training PINNs ({n_epochs} epochs)...")

    print("\n  --- Direct OCP PINN ---")
    pinn_direct = DirectOCPPINN(
        hidden_layers=[64, 64, 64, 64],
        lr=1e-3,
        w_data=1.0,
        w_physics=0.05,
        w_mono=0.005 if electrode_type == 'cathode' else 0.001,
        w_smooth=0.001,
        electrode_type=electrode_type,
        activation='tanh',
    )
    pinn_direct.train(data, n_epochs=n_epochs, n_colloc=300, verbose=True)
    print(f"  Direct PINN training time: {pinn_direct.training_time:.2f} s")

    print("\n  --- Free-Energy PINN ---")
    pinn_fe = FreeEnergyPINN(
        hidden_layers=[64, 64, 64, 64],
        lr=1e-3,
        w_data=1.0,
        w_convex=0.01,
        activation='tanh',
    )
    pinn_fe.train(data, n_epochs=n_epochs, n_colloc=300, verbose=True)
    print(f"  Free-Energy PINN training time: {pinn_fe.training_time:.2f} s")

    print(f"\n[4/5] Training baseline models...")
    baselines = train_all_baselines(data)

    print(f"\n[5/5] Evaluating all models...")
    table = run_full_evaluation(
        electrode_label,
        data,
        pinn_direct,
        pinn_fe,
        baselines,
        output_dir=f"{output_dir}/figures",
    )
    return table


def main():
    output_dir = "results_pybamm"
    os.makedirs(output_dir, exist_ok=True)

    n_epochs = 5000
    electrodes = [
        ("graphite", "Graphite (Li_xC_6)", "anode"),
        ("nmc811", "NMC811 (Li_xNi_{0.8}Mn_{0.1}Co_{0.1}O_2)", "cathode"),
        ("lfp", "LFP (Li_xFePO_4)", "cathode"),
    ]

    all_tables = []
    for electrode, label, etype in electrodes:
        table = run_electrode(electrode, label, etype, output_dir, n_epochs)
        table["Electrode"] = label
        all_tables.append(table)

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(f"{output_dir}/combined_comparison.csv", index=False)

    print(f"\n\n{'='*70}")
    print("  COMBINED RESULTS — ALL ELECTRODES")
    print(f"{'='*70}")
    print(combined.to_string(index=False))
    print(f"\n\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
