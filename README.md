# TCAS-DVFS: Thermal-Crosstalk-Aware Scheduling and Predictive DVFS for High-Vacuum Space AI Clusters

Approximate Multiphysics Modeling, Predictive Control, and Numerical Simulation for 2.5D/3D Chiplet Accelerators in Low Earth Orbit (LEO).

## Repository Overview

- `controller/`: Predictive Thermal MPC (`mpc_tcas_controller.m`, QP formulation).
- `models/`: Multiphysics thermal network, orbital radiation model ($Q_{sol}$, $Q_{alb}$, $Q_{ir}$), and silicon leakage kinetics.
- `scheduler/`: Thermal-crosstalk-aware and priority-based task dispatchers.
- `baselines/`: Reference strategies (Static Safe, Reactive Throttling, Blind Round-Robin).
- `plots/`: `gerar_todas_figuras_artigo1.m` — Automated generation of publication figures (Figures 1, 5, 6, 7, 8 and Table 1) directly from `.mat` simulation logs.
- `docs/`: 
  - `PROMPTS_ILUSTRACOES_GEMINI.md`: Full text (~1,300 words each) of prompts used to generate the conceptual architectural diagrams (Figures 1 to 4) for scientific reproducibility.

## Requirements & Reproducibility

### Numerical Simulation & Plot Generation
- **GNU Octave** (v8.0+ or v11+) or **MATLAB** (R2022b+).
- Run the simulation scripts in sequence or directly generate paper plots:
  ```matlab
  cd plots
  gerar_todas_figuras_artigo1
  ```
  The script automatically reads datasets from `../data/` and outputs publication-quality 300 DPI PNG figures.

### Conceptual Illustrations
The system architecture, orbital thermal geometry, and MPC control-flow diagrams (Figures 1–4 in Section I–III) were synthesized using generative visual tools. The full prompt formulations are detailed in `docs/PROMPTS_ILUSTRACOES_GEMINI.md`.
