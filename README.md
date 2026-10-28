# TCAS-DVFS: Thermal-Crosstalk-Aware Scheduling and Model Predictive DVFS for Spaceborne Multicore AI Accelerators

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](requirements.txt)
[![Target: Elsevier CEP](https://img.shields.io/badge/Target-Control%20Engineering%20Practice-orange.svg)](https://www.sciencedirect.com/journal/control-engineering-practice)

This repository contains the complete simulation infrastructure, predictive control algorithms (TCAS-DVFS), experimental testbenches, sensitivity evaluations, and reproducibility artifacts accompanying the manuscript:

> **"Thermal-Crosstalk-Aware Scheduling and Model Predictive DVFS for Spaceborne Multicore AI Accelerators"**  
> *Submitted to Elsevier Control Engineering Practice (CEP).*

---

## 1. Repository Structure

```text
.
├── configs/
│   └── nominal.json                 # Nominal parameters (orbit, thermal, MPC, SDP)
├── experiments/
│   ├── run_experimentos.py          # Primary experimental campaign (Table 1, ablation, multi-seed)
│   ├── run_incerteza.py             # Robustness & uncertainty campaign (LHS ±20%, n=30)
│   ├── run_convergencia_temporal.py # Numerical convergence analysis (dt in [1.0, 0.5, 0.25, 0.125] s)
│   └── validacao_rainflow.py        # Rainflow algorithm validation against ASTM E1049-85
├── results/
│   ├── tabela1.csv                  # Baseline comparisons (3 orbits, seed 0)
│   ├── tabela_sementes.csv          # 10-seed stochastic noise evaluation
│   ├── ablacao.csv                  # Architectural ablation study
│   ├── amostras.csv                 # LHS parameter perturbation matrix (n=30)
│   ├── convergencia_temporal.csv    # Integrator error vs time step (dt)
│   ├── convergencia_temporal.json   # Full temporal convergence metrics
│   ├── metadata.json                # Complete computational environment metadata
│   ├── resultados.json              # Primary campaign JSON output
│   └── incerteza.json               # Uncertainty campaign JSON output
├── src/
│   ├── tcas_core.py                 # Thermal RC network plant, orbit model, baseline governors
│   └── tcas_dvfs.py                 # TCAS-DVFS: SDP spatial scheduler + warm-started ADMM QP-MPC
├── requirements.txt                 # Pinned dependencies
├── LICENSE                          # MIT License
└── README.md                        # This documentation
```

---

## 2. Requirements and Installation

The code requires Python 3.9 or newer and standard scientific libraries (`numpy`, `matplotlib`, `scipy`).

```bash
# Clone the repository (or extract local archive)
cd "Artigo 01"

# Create a virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Running Experiments

### A. Primary Experimental Campaign (Table 1, Ablation, Multi-seed)
To reproduce the full 3-orbit baseline comparisons, ablation configurations, and 10-seed sensor noise evaluations:
```bash
python experiments/run_experimentos.py --orbitas 3 --orbitas-ablacao 3 --sementes 10 --saida results
```

### B. Uncertainty & Robustness Evaluation (LHS ±20%, n=30)
To run the 30-sample Latin Hypercube Sampling campaign evaluating parameter mismatches in thermal capacitances, conductances, optical absorptivity, and emissivity:
```bash
python experiments/run_incerteza.py --amostras 30 --orbitas 3 --saida results
```

### C. Rainflow Algorithm ASTM E1049-85 Validation
To formally verify the 3-point Rainflow counting implementation and cyclic range ($\Delta T$) definition:
```bash
python experiments/validacao_rainflow.py
```

### D. Numerical Integrator Temporal Convergence Study
To evaluate the convergence of the second-order Runge-Kutta / Heun scheme across $\Delta t_{\text{plant}} \in \{1.0, 0.5, 0.25, 0.125\}\text{ s}$:
```bash
python experiments/run_convergencia_temporal.py
```

---

## 4. Methodological Demarcation

As detailed in Table 1 of the manuscript, this repository implements a **control-theoretic numerical simulation** under the following scope:
- **Directly Simulated / Estimated:** 3D RC thermal network dynamics, orbital solar/albedo/Earth-IR radiative heat transfer, ADMM predictive frequency modulation, SDP spatial allocation, cyclic thermal range ($\Delta T$).
- **Proxies:** Coffin-Manson cycle damage metric ($\sum n_i \Delta T_i^q$), Black's electromigration acceleration ratio ($MTTF_{\text{rel}}$).
- **Explicit Exclusions:** This software does not measure plastic strain in solder ball joints, does not perform radiation single-event upset (SEU) injection, and does not claim space flight qualification or absolute hardware MTTF in operating hours.

---

## 5. Citation and License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use this codebase or model in your research, please cite:
```bibtex
@article{barros2026tcas,
  author    = {Barros, Dafnhe T.},
  title     = {Thermal-Crosstalk-Aware Scheduling and Model Predictive {DVFS} for Spaceborne Multicore {AI} Accelerators},
  journal   = {Control Engineering Practice},
  year      = {2026},
  note      = {Submitted}
}
```
