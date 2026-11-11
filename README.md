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
│   ├── validacao_rainflow.py        # Rainflow algorithm validation against ASTM E1049-85
│   ├── validacao_jacobiano.py       # Analytical Jacobian 4th-order finite difference verification
│   └── validar_pinn.py              # Physics-Informed Neural Network (PINN) out-of-sample validation
├── results/
│   ├── tabela1.csv                  # Baseline comparisons (3 orbits, seed 0)
│   ├── tabela_sementes.csv          # 10-seed stochastic noise evaluation
│   ├── ablacao.csv                  # Architectural ablation study (1 orbit, seed 0)
│   ├── amostras.csv                 # LHS parameter perturbation matrix (n=30)
│   ├── convergencia_temporal.csv    # Integrator error vs time step (dt)
│   ├── convergencia_temporal.json   # Full temporal convergence metrics
│   ├── fig_validacao_pinn.png       # Out-of-sample PINN validation curves
│   ├── metadata.json                # Complete computational environment metadata
│   ├── resultados.json              # Primary campaign JSON output
│   └── incerteza.json               # Uncertainty campaign JSON output
├── src/
│   ├── tcas_core.py                 # Thermal RC network plant, orbit model, baseline governors
│   ├── tcas_dvfs.py                 # TCAS-DVFS: SDP spatial scheduler + warm-started ADMM QP-MPC (152 vars)
│   └── pinn_thermal.py              # Physics-Informed Neural Network (PINN) surrogate model
├── requirements.txt                 # Pinned dependencies
├── LICENSE                          # MIT License
└── README.md                        # This documentation
```

---

## 2. Mathematical and Physical Formulations

### A. Radiative Thermal Budget (Fourier Conduction Bound)
The radiator budget is derived from the first principles of Stefan-Boltzmann radiation and Fourier heat conduction:
- **Maximum Radiator Temperature:** $T_{\text{rad,max}} = T_{\text{cap}} - \frac{P_{\text{min,cores}} \cdot R_r}{4}$, ensuring that core temperatures remain within $T_{\text{cap}}$ even at the minimum operational power floor $P_{\text{min,cores}}$.
- **Transient Thermal Buffer:** $P_{\text{buffer}} = \frac{C_{\text{rad}} (T_{\text{rad,max}} - T_{\text{rad}})}{\tau_{\text{horiz}}} \left(1 - e^{-\tau_{\text{horiz}} / \tau_{\text{rad}}}\right)$, where $\tau_{\text{rad}} = C_{\text{rad}} R_r / 4$.
- **Physical Infeasibility Contingency:** If $P_{\text{budget\_phys}} < P_{\text{min,cores}}$, the governor immediately triggers a deterministic hardware protection contingency ($F = F_{\text{min}}$) with status `"physically_infeasible"`, eliminating any artificial floor masking in the quadratic program.

### B. Decoupled Spatio-Temporal MPC (152 Decision Variables)
The predictive controller solves an ADMM quadratic program with:
- **32 frequency variables** ($hc \times N = 8 \times 4$);
- **120 individual spatio-temporal slack variables** ($s_{i,k} \ge 0$ for each die $i \in \{1..4\}$ and prediction step $k \in \{1..30\}$);
- **Total decision variables:** $n_{\text{vars}} = 152$;
- **Stopping Criteria:** Scaled OSQP tolerances ($\epsilon_{\text{pri}}$, $\epsilon_{\text{dua}}$) with rigorous dual residual projection on $A^T$.

### C. Semidefinite Relaxation (SDP) Consensus Residual
The topological task mapper computes the exact dual consensus residual projected onto the affine constraint manifold:
$$r_{\text{dua}} = \rho \|\mathcal{P}_{\text{affine}}((Z_1^k - Z_1^{k-1}) + (Z_2^k - Z_2^{k-1}))\|_F$$

---

## 3. Running Experiments

### A. Primary Experimental Campaign (Table 1, Ablation, Multi-seed)
To reproduce the full 3-orbit baseline comparisons, 1-orbit ablation configurations, and 10-seed sensor noise evaluations:
```bash
python experiments/run_experimentos.py --orbitas 3 --orbitas-ablacao 1 --sementes 10 --saida results
```

### B. Analytical Jacobian Verification
To verify the analytical Jacobian against 4th-order central finite differences across 100 stochastic operational states:
```bash
python experiments/validacao_jacobiano.py
```

### C. Rainflow Algorithm ASTM E1049-85 Validation
To formally verify the 3-point Rainflow counting implementation and complete multiset cycle equality (Section 5.4.4):
```bash
python experiments/validacao_rainflow.py
```

### D. Numerical Integrator Temporal Convergence Study
To evaluate the convergence of the second-order Runge-Kutta / Heun scheme across $\Delta t_{\text{plant}} \in \{1.0, 0.5, 0.25, 0.125\}\text{ s}$:
```bash
python experiments/run_convergencia_temporal.py
```

### E. Physics-Informed Neural Network (PINN) Out-of-Sample Validation
To train the PINN surrogate on Orbit 1 and evaluate out-of-sample generalization on Orbit 2:
```bash
python experiments/validar_pinn.py
```

### F. Uncertainty & Robustness Evaluation (LHS ±20%, n=30)
To run the 30-sample Latin Hypercube Sampling campaign:
```bash
python experiments/run_incerteza.py --amostras 30 --orbitas 3 --saida results
```

---

## 4. Citation and License

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
