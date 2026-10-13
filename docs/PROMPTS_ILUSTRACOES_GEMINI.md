# Prompts para Geração e Tradução das Figuras Científicas (PT -> EN)
**Projeto de Mestrado — PROPECAUT / IFES**  
**Artigo:** *Dynamic Voltage and Frequency Scaling and Thermal-Crosstalk-Aware Scheduling for Orbital AI Clusters in Space Vacuum*  
**Autor:** Dafnhê Thor Castro de Martins Barros | **Orientador:** Flávio Barcelos Braz da Silva  

Este documento reúne os prompts técnicos avançados prontos para uso em ferramentas de geração de imagens (Google Gemini / Imagen), calibrados para produzir figuras de nível de publicação científica (padrão IEEE Transactions on Aerospace and Electronic Systems), com estilo vetorial, tipografia em inglês estritamente legível e fidelidade aos dados de co-simulação do artigo.

---

## Diretrizes Gerais para Todas as Gerações
1. **Formato:** Widescreen 16:9 (`--ar 16:9`).
2. **Estilo:** Ilustração técnica vetorial científica / engenharia de sistemas aeroespaciais, fundo branco limpo, alto contraste, paleta sóbria (azul marinho, ciano, ardósia, detalhes em dourado para raios solares).
3. **Tipografia:** Fonte sem serifa técnica (estilo Helvetica / Arial / LaTeX), apenas em inglês, sem erros de ortografia, com fórmulas matemáticas bem delineadas.

---

### Figura 01: Geometria Orbital e Balanço Radiativo (500 km LEO SSO)
```text
Scientific 3D technical illustration for an IEEE aerospace journal, 16:9 widescreen format, vector diagram aesthetic with clean lines and high contrast.
Topic: 500-km Sun-Synchronous Orbit (SSO) low Earth orbit geometry and radiative heat balance of a small satellite (12U CubeSat).
Visual elements:
- A realistic Earth globe showing atmospheric limb and orbital path.
- 12U satellite chassis with an external passive thermal radiator panel (Area = 0.12 m²).
- Vector arrows illustrating external radiative thermal fluxes:
  * "Direct Solar Radiation: S_solar = 1361 W/m² (alpha_s = 0.12)"
  * "Earth Albedo Flux: a_E = 0.30, Q_albedo"
  * "Earth Infrared Emission: T_earth = 255 K, Q_earth_IR"
  * "Deep Space Heat Sink: T_space = 3 K (-270.15 °C)"
  * "Radiator Quartic Emission: q_rad = epsilon * sigma * T_rad^4 (epsilon = 0.88)"
- Orbital eclipse shadow cone clearly separating "Sunlit Phase (59.4 min)" and "Eclipse / Umbra Phase (35.2 min)".
- Clean, crisp English labels only, sans-serif scientific typography, white background with professional engineering color palette (navy blue, gold solar rays, cyan radiation lines). No text errors or misspelled words.
```

---

### Figura 02: Fluxo de Co-simulação Multifísica e Controle
```text
High-resolution scientific systems engineering flow diagram for an IEEE Transactions paper, 16:9 aspect ratio, clean white background.
Topic: Integrated discrete-time multiphysics co-simulation architecture (Delta t_c = 100 ms).
Diagram layout (left to right interconnected modular blocks with professional aerospace iconography):
- Block 1: "Orbital Mechanics & Dynamics (500-km SSO, J2 perturbation, beta angle)"
- Block 2: "External Radiative Environment (Solar flux, Earth albedo, Earth IR, eclipse shadow function)"
- Block 3: "Compact Thermal Network (5x5 CTM state-space model, AlN cold plate, inter-die conductive crosstalk)"
- Block 4: "Semiconductor Power Dissipation (Dynamic power + exponential subthreshold leakage P_leak(T))"
- Block 5: "Supervisory Control Layer: TCAS-DVFS (SDP task allocator + Multivariable MPC-DVFS active-set solver)"
- Feedback loop: "Real-time Telemetry (TMS junction diodes, current shunt rail sensing at 100 Hz)".
All text strictly in English, sharp typography, professional blue/dark grey color scheme, crisp vector arrows and mathematical symbols.
```

---

### Figura 03: Empacotamento Multi-Die e Diafonia Térmica Condutiva 3D
```text
Engineering 3D exploded isometric diagram and equivalent thermal circuit schematic for a space electronics research paper, 16:9 widescreen, clean scientific aesthetic.
Topic: High-density multi-die System-on-Module packaging and inter-die conductive thermal crosstalk.
Visual layout:
- Left: 3D layered mechanical packaging stack:
  * "Silicon Die (FinFET nodes, 8 nm, 15-25 W)"
  * "TIM-1 (Thermal Interface Material)"
  * "Nickel-plated Copper Heat Spreader (IHS)"
  * "TIM-2"
  * "Aluminum Nitride Carrier (AlN Substrate, k = 180 W/(m·K))"
  * "BGA Solder Balls (SAC305 lead-free alloy)"
  * "Structural Cold Plate Chassis (Al 6061-T6)"
- Right: Thermal RC circuit schematic showing 4 compute nodes in a 2x2 grid:
  * Heat sources P_1(t) through P_4(t)
  * Lumped thermal capacitances c_i = 14.8 J/K
  * Lateral coupling mutual resistances R_12 = R_34 = 1.45 K/W, R_13 = R_24 = 1.52 K/W, diagonal R_14 = 3.20 K/W
  * Conductive path to radiator R_ri = 2.10 K/W rejecting heat to T_space = 3 K.
All labels in English, clean white background, razor-sharp technical vector lines.
```

---

### Figura 04: Não Linearidade de Fuga Estática e Feedback Eletrotérmico
```text
Precise scientific 2D data visualization plot for an IEEE microelectronics journal, 16:9 aspect ratio, clean white background.
Topic: Subthreshold leakage power feedback and thermal runaway in space vacuum.
Plot details:
- Main Graph: Leakage Current I_leak and Leakage Power P_leak vs. Junction Temperature T_j (from 20 °C to 120 °C).
- Show exponential upward curves for core voltages: V_core = 0.72 V (optimized TCAS-DVFS), 0.85 V, and 0.98 V (saturated Ondemand).
- Clearly mark:
  * "Subthreshold Leakage: I_leak proportional to T^2 * exp(-q*V_th / (k_B*T))"
  * "Target Safety Boundary: T_safe = 85.00 °C (358.15 K)"
  * "Critical Throttling Trip Point: T_trip = 95.00 °C (368.15 K)"
  * Shaded red danger zone: "Positive Thermal Feedback / Thermal Runaway Zone (Vacuum: h = 0)"
  * Shaded green safe zone: "TCAS-DVFS Operating Envelope (Strictly <= 83.89 °C)"
High readability, flawless English labels, precise LaTeX-style mathematical symbols, crisp coordinate axes with grid lines.
```

---

### Figura 05: Dinâmica Orbital de Iluminação, Penumbra e Eclipse
```text
Scientific orbit mechanics visualization and timing graph for an aerospace publication, 16:9 aspect ratio, clean white background.
Topic: LEO solar illumination, penumbra transitions, and umbra eclipse dynamics (500 km SSO, period = 94.6 min).
Visual panels:
- Top: 3D orbital diagram showing satellite moving from "Direct Sunlit Phase (59.4 min)" into "Atmospheric Penumbra Transition Zone" and then "Total Umbra Eclipse (35.2 min)".
- Bottom: Time-series curve of solar visibility indicator function I_sunlit(t) from 0.0 to 1.0.
- Highlight steep flux slew rate during penumbra entry and exit: "|dQ_ext/dt| > 45 W/(m²·s)".
- Annotate geometric parameters: Earth radius R_E = 6371 km, orbit altitude H_orb = 500 km, solar half-angle theta_sun_app = 0.267°.
All labels strictly in English, clear technical annotations, elegant navy/gold/cyan scientific color palette.
```

---

### Figura 06: Malha de Elementos Finitos 3D (FEM) de Referência
```text
3D finite element engineering simulation rendering and thermal mesh for a satellite chassis, 16:9 widescreen, professional CAD/CAE aesthetic.
Topic: 3D FEM thermal structural analysis of a 12U CubeSat chassis.
Visual elements:
- 3D structured hexahedral mesh of satellite structure (482,000 cells) with fine mesh refinement (0.25 mm) around 4 compute dies and TIM layers.
- Cross-sectional temperature contour gradient map displayed in Kelvin (color bar from 293 K to 365 K).
- Callout annotations:
  * "Al 6061-T6 Chassis Frame (Alodine 1200 finish)"
  * "AlN High-Conductivity Cold Plate (180 mm x 120 mm x 3.5 mm, k = 180 W/(m·K))"
  * "4x System-on-Module Inference Dies (2x2 Grid)"
  * "Multivariable Subspace Identification (N4SID): Compact CTM calibrated with RMSE < 0.38 K vs full 3D conductive domain"
Crisp English annotations, clean background, high-resolution engineering presentation.
```

---

### Figura 07: Arquitetura de Controle em Malha Fechada do TCAS-DVFS
```text
Control systems engineering block diagram for an IEEE Transactions journal, 16:9 aspect ratio, clean white background.
Topic: Hierarchical closed-loop control architecture of TCAS-DVFS.
Diagram blocks and signals (clearly labeled with transfer functions and discrete equations):
- "Inference Task Queue W = {w_1, ..., w_M}"
- "Upper-Level Scheduler: SDP-TCAS (Convex Semidefinite Relaxation & Randomized Rounding, delta t_sched = 2.0 s)"
- "Lower-Level Controller: Multivariable MPC-DVFS (Receding Horizon H_p = 30, H_c = 10, Active-Set QP Solver, delta t_c = 100 ms)"
- "EPS Orbital Energy Budget Feedback: P_orbital_budget(t)"
- "State Estimator: Real-time Extended Kalman Filter (EKF)"
- "Plant: 5x5 Compact Thermal Network (CTM) in Space Vacuum"
- Output telemetry: "Junction Temperatures T_j(k), Core Voltages V_core(k), Clock Frequencies f_clk(k)".
Sharp vector lines, strict English text, no spelling errors, elegant corporate IEEE engineering color palette.
```

---

### Figura 08: Painel Comparativo Consolidado de Métricas (Tabela 1)
```text
Comprehensive scientific dashboard and multi-panel performance comparison chart for an IEEE publication, 16:9 widescreen format, clean white background.
Topic: Quantitative performance benchmarking of DVFS governors in 500-km LEO orbit (3 full orbits, 34,009 time steps).
Panels (display exact values clearly in clean bar charts and data tables):
- Panel A: "Peak Junction Temperature (T_max)"
  * Linux Ondemand: 412.33 °C (Thermal Runaway)
  * Global Power Cap (65 W): 142.74 °C (Runaway)
  * Thermal Throttling: 85.00 °C
  * Space-PID: 85.47 °C
  * TCAS-DVFS: 83.89 °C (Strictly Safe, < 85.00 °C)
- Panel B: "Inter-Die Spatial Gradient (Delta T_spatial)"
  * Linux Ondemand: 12.87 K
  * Thermal Throttling: 9.99 K
  * Space-PID: 4.78 K
  * TCAS-DVFS: 1.63 K (83.7% reduction)
- Panel C: "Thermal Violations (> 85 °C)"
  * Linux Ondemand: 16,038 s (94.26%)
  * Thermal Throttling: 4 s (0.02%)
  * TCAS-DVFS: 0 s (0.00% violations)
- Panel D: "Total Energy & Sustained Throughput"
  * TCAS-DVFS: 110.82 FPS sustained, 932.6 kJ total energy, EDP = 8.41 kJ·s
All numbers exactly matching Table 1, 100% English text, crisp scientific styling.
```

---

### Figura 09: Trajetórias Térmicas Transitórias em 3 Órbitas LEO
```text
High-precision multi-orbit time-series plot for an IEEE aerospace paper, 16:9 aspect ratio, clean white background.
Topic: Transient junction temperatures and inference throughput across 3 consecutive LEO orbits (284 minutes, 34,009 steps).
Plot sections:
- Top Graph: "Peak Junction Temperature T_max (°C) vs Mission Time (min)"
  * Red curve: "Linux Ondemand (Severe thermal runaway, peaking at 412.33 °C)"
  * Orange dashed curve: "Global Power Cap (Unstable runaway, peaking at 142.74 °C)"
  * Blue dashed curve: "Thermal Throttling (Sawtooth limit-cycle oscillations between 70 °C and 85 °C)"
  * Solid Dark Blue curve: "TCAS-DVFS (Proposed) - Smooth isothermal stability capped at 83.89 °C"
  * Horizontal red dashed line: "Safety Limit: T_trip = 85.00 °C"
  * Shaded grey regions: "Eclipse Phases (35.2 min each)"
- Bottom Graph: "Sustained Inference Throughput (FPS)" showing stable 110.82 FPS under TCAS-DVFS vs erratic stalls under Throttling.
- Inset box: "Spatial Gradient: Delta T_spatial = 1.63 K (83.7% attenuation)".
Flawless scientific data rendering, pure English annotations, high legibility.
```

---

### Figura 10: Curvas de Modulação e P-states do TCAS-DVFS
```text
Detailed 2D engineering graph showing Dynamic Voltage and Frequency Scaling (DVFS) trajectories, 16:9 widescreen, clean white background.
Topic: Adaptive P-state transitions and proactive frequency modulation under TCAS-DVFS.
Graph elements:
- Top Axis: "Clock Frequency f_clk (GHz) from 0.30 GHz to 1.30 GHz"
- Bottom Axis: "Core Supply Voltage V_core (V) from 0.65 V to 1.05 V"
- Curve: Discrete P-state operating points along the FinFET alpha-power delay curve (tau_prop <= 1 / (2*f_clk)).
- Callout annotations:
  * "Proactive frequency scaling during sunlit thermal accumulation"
  * "Optimal power point: V_core = 0.72 V at f_clk = 672 MHz (Mean operational state)"
  * "Slew rate limiter: |Delta f_clk| <= 50 MHz per 100 ms step"
  * "Rapid frequency boost during orbital eclipse cold recovery"
Clean vector axes, all text in English, professional blue and slate color palette.
```

---

### Figura 11: Mapeamento Espacial de Tarefas via Relaxação SDP
```text
Scientific schematic diagram showing spatial workload distribution on a 2x2 multi-die processor cluster, 16:9 widescreen, clean white background.
Topic: Topology-Crosstalk-Aware Scheduling (TCAS) via Semidefinite Programming (SDP) relaxation.
Visual contrast:
- Left: "Greedy / Linux Ondemand Allocation (Unaware of Crosstalk)"
  * Heavy neural tasks concentrated on adjacent Die 1 and Die 2
  * Red thermal hotspot overlapping on cold plate, resulting in severe spatial gradient Delta T = 12.87 K
- Right: "Proposed SDP-TCAS Topological Allocation"
  * Heavy neural tasks steered to diagonally separated Die 1 and Die 4
  * Dispersed heat flux across AlN carrier plate, minimizing mutual coupling cost Tr(M_Phi * Y)
  * Uniform thermal profile with inter-die gradient suppressed to Delta T = 1.63 K.
Clear mathematical labels, arrows showing heat conduction pathways, 100% English labels, professional IEEE graphic style.
```

---

### Figura 12: Cargas de Trabalho de IA de Borda Orbital (Workloads)
```text
Scientific overview diagram of spaceborne deep learning benchmark models, 16:9 widescreen format, modern vector illustration, white background.
Topic: Edge AI inference workloads for LEO Earth observation.
Three distinct workflow panels:
- Panel 1: "Cloud-Seg: Optical Cloud Screening"
  * Architecture: DeepLabV3+ with ResNet-50 backbone
  * Input: 512x512 multispectral optical tiles
  * Metrics: 55.4 FPS sustained, 41.0 ms latency, 37.1% EDP reduction
- Panel 2: "SAR-YOLO: Maritime Vessel Detection"
  * Architecture: YOLOv8x on Synthetic Aperture Radar complex matrices
  * Metrics: 88.6 FPS sustained, 28.4 ms latency, 41.2% EDP reduction
- Panel 3: "Hyper-ViT: Hyperspectral Feature Extractor"
  * Architecture: Vision Transformer (ViT-Tiny) across 204 spectral bands
  * Metrics: 40.2 FPS sustained, 58.2 ms latency, 38.4% EDP reduction.
Crisp icons, pure English typography, elegant scientific color scheme.
```

---

### Figura 13: Diagrama Conceitual da Plataforma de Co-simulação HIL
```text
Laboratory systems engineering architecture diagram for Hardware-in-the-Loop (HIL) space validation, 16:9 aspect ratio, clean white background.
Topic: Real-time HIL orbital thermal-vacuum emulation testbed.
System blocks:
- Left: "Host Workstation & Telemetry Logging (100 Hz sampling)"
- Center: "Real-Time Target (RT-Linux executing 500-km SSO orbit & 5x5 CTM thermal differential equations)"
- Right: "Physical Compute Cluster: 4x NVIDIA Jetson Orin SoM on AlN Carrier Plate"
  * Shunt current amplifiers (TI INA226, 1 kHz rail power monitoring)
  * Programmable DC power supplies (12 V rail)
  * On-chip TMS thermal diode telemetry acquisition.
Clean arrows showing bi-directional signals: "Voltage/Current commands", "Thermal telemetry T_j", "Inference frame triggers". All text strictly in English, high-resolution vector aesthetic.
```

---

### Figura 14: Comparativo Conceitual dos Governadores de Linha de Base
```text
Conceptual comparative infographic for an IEEE Transactions paper, 16:9 widescreen format, clean white background.
Topic: Baseline thermal governors vs. the proposed TCAS-DVFS framework.
Five comparison columns with icons and metric badges:
1. "Linux Ondemand: Reactive, blind to vacuum physics -> Thermal Runaway (412.33 °C), 16,038 s violations"
2. "Thermal Throttling: Reactive trip point -> Violent Sawtooth Hysteresis, 9.99 K spatial gradient"
3. "Space-PID: Decentralized SISO -> High-frequency hunting, steady-state offset, 4.78 K gradient"
4. "Global Power Cap: Uniform power scaling -> Cannot prevent localized hotspot runaway (142.74 °C)"
5. "TCAS-DVFS (Proposed): Model-predictive + Crosstalk-aware -> Zero violations (0.00%), 83.89 °C peak, 1.63 K gradient, 110.82 FPS sustained".
Clear checkmarks, cross marks, badges, strictly English text, high readability.
```

---

### Figura 15: Projeções de Confiabilidade Física Estrutural (Black e Coffin-Manson)
```text
Precise scientific bar chart and reliability projection diagram for an IEEE semiconductor reliability journal, 16:9 aspect ratio, clean white background.
Topic: Structural and physical reliability enhancement under TCAS-DVFS.
Visual layout (Two primary comparison charts):
- Chart A: "Interconnect Electromigration Lifetime (Black's Equation)"
  * Formula: MTTF_EM = A_EM * J^-n * exp(E_a / (k_B * T_j))
  * Bar 1: "Baseline Thermal Throttling: 1.00x (Reference)"
  * Bar 2: "TCAS-DVFS (Nominal): 2.53x (+153% Lifetime Extension)"
  * Bar 3: "TCAS-DVFS (Conservative): 2.34x (+134% Lifetime Extension)"
  * Callout: "Driven by peak junction containment at 83.89 °C and reduced leakage current"
- Chart B: "BGA Solder Joint Thermal Fatigue Life (Coffin-Manson Relation)"
  * Formula: N_f = B * (Delta T_spatial)^-q (q = 2.1 for SAC305 solder)
  * Bar 1: "Baseline Thermal Throttling (Delta T = 9.99 K): 1.00x (Reference)"
  * Bar 2: "TCAS-DVFS (Delta T = 1.63 K): 45.1x (+4410% Fatigue Endurance Improvement)"
  * Callout: "45.1-fold increase in solder thermal cycling endurance due to 83.7% spatial gradient reduction"
Numbers strictly matching the article text (+4410% and 45.1x), flawless English typography, professional blue/teal engineering color palette.
```
