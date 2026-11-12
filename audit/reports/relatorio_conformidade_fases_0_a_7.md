# Relatório Técnico de Conformidade Forense — TCAS-DVFS (Fases 0 a 7: A01–A18)

Este documento consolida a execução integral do **Plano de Ação Mandatória — Correção do TCAS-DVFS** após a reauditoria do commit `a8027ee`. Para cada uma das **18 ações mandatórias (A01–A18)** distribuídas nas **Fases 0 a 7**, registram-se os **4 entregáveis exigidos**:
1. **Correção de código**: arquivos alterados, descrição técnica e commit SHA;
2. **Teste automatizado**: script executável que verifica a propriedade exigida;
3. **Resultado numérico**: saída versionada em JSON ou CSV;
4. **Interpretação científica**: o que o resultado comprova e o seu domínio de validade (o que ele não reivindica).

---

## Fase 0 — Congelamento, Rastreabilidade e Isolamento

- **Tag de Congelamento**: `audit-baseline-a8027ee` apontando para `a8027ee943659c61a09f3abb7043e22b89e74941`.
- **Arquivo de Congelamento**: `audit/reproducibility/baseline_freeze.json` e cópia integral dos resultados pré-correção em `audit/reproducibility/historical_a8027ee/`.
- **Estrutura de Auditoria**: `audit/{reports,tests,numerical_validation,pinn,mpc,sdp,reproducibility}/`.

---

## Fase 1 — Reconstrução Matemática da PINN (A01 a A05)

### A01 — Retropropagação Analítica Completa da Perda Física ($L_{\text{phys}}$)
1. **Correção de código**: `src/pinn_thermal.py` (Commit `658a992`). Implementado o método `_backprop_physics(t_col, f_col, al_col, qext_col)` que calcula o resíduo da EDO de 5 estados $R = \dot{\hat{T}} - f_{\text{CTM}}(\hat{T}, f, \alpha, Q_{\text{ext}})$ e retropropaga analiticamente:
   - O gradiente em relação à predição de temperatura $\frac{\partial L_{\text{phys}}}{\partial \hat{T}} = -\frac{2}{5 N_c} R \left(\frac{\partial f_{\text{CTM}}}{\partial \hat{T}}\right)^T$ usando o Jacobiano físico exato;
   - O gradiente em relação à derivada temporal $\frac{\partial L_{\text{phys}}}{\partial \dot{\hat{T}}} = \frac{2}{5 N_c} R$ através de todas as camadas ocultas via primeira derivada $\sigma'(z) = 1 - a^2$ e segunda derivada $\sigma''(z) = -2 a (1 - a^2)$ da ativação $\tanh$.
2. **Teste automatizado**: `audit/tests/test_pinn_gradients.py` (Testes `T01`, `T02`, `T03`, `T05`).
3. **Resultado numérico**: `audit/pinn/pinn_gradient_tests.json`:
   - `T01 (w_phys = 0)`: contribuição física no gradiente total $= 0.000 \times 10^0$;
   - `T02 (w_phys > 0)`: norma de Frobenius do gradiente físico por camada $\|g_{W,\text{phys}}\|_F = [1.9179, 1.7486, 2.2616]$;
   - `T03 (sensibilidade a w_phys)`: $\|\Delta g_{W,\text{total}}\|_F = [4.7947, 4.3715, 5.6539]$ entre $w_{\text{phys}}=1$ e $w_{\text{phys}}=10$;
   - `T05 (diferenças finitas centrais)`: erro relativo máximo de Frobenius entre gradiente analítico e numérico em todas as camadas $= 2.463 \times 10^{-9}$ para $L_{\text{phys}}$ e $4.183 \times 10^{-9}$ para $L_{\text{total}}$ (critério $< 10^{-7}$).
4. **Interpretação científica**: Comprova matematicamente que $w_{\text{phys}}$ altera diretamente os gradientes de todos os pesos e vieses de todas as camadas da rede, com exatidão de $O(10^{-9})$ frente a diferenças finitas centrais.

### A02 — Retropropagação Completa da Condição Inicial ($L_{\text{IC}}$)
1. **Correção de código**: `src/pinn_thermal.py` (Commit `658a992`). Substituído o gradiente truncado na última camada por `_backprop_mse(t_ic, f_ic, al_ic, q_ic, T_ic)` percorrendo todas as camadas ocultas e de saída.
2. **Teste automatizado**: `audit/tests/test_pinn_gradients.py` (Teste `T04`).
3. **Resultado numérico**: `audit/pinn/pinn_gradient_tests.json`:
   - `T04`: normas de gradiente de $L_{\text{IC}}$ por camada $\|g_{W,\text{ic}}\|_F = [549.87, 375.13, 646.84]$ e $\|g_{b,\text{ic}}\|_2 = [139.20, 297.10, 444.89]$, todas estritamente positivas.
4. **Interpretação científica**: Comprova que o vínculo físico de ancoragem em $t=0$ atualiza as representações internas de todas as camadas ocultas, não apenas o viés de saída.

### A03 — Tratamento de Lotes Incompletos (Mini-batches)
1. **Correção de código**: `src/pinn_thermal.py` (Commit `658a992`). Substituído `n_batches = max(1, N_data // batch_size)` por `starts_d = list(range(0, N_data, batch_size))`, processando 100% das amostras supervisionadas e de colocação em todas as épocas.
2. **Teste automatizado**: `audit/tests/test_pinn_gradients.py` (Teste `T06`).
3. **Resultado numérico**: `audit/pinn/pinn_gradient_tests.json`:
   - `T06`: para $N_{\text{data}} = 103$ e `batch_size = 32` (resto 7), `samples_processed_per_epoch = [103, 103, 103]`.
4. **Interpretação científica**: Elimina o descarte silencioso das amostras finais de cada época quando $N_{\text{data}}$ não é múltiplo de `batch_size`.

### A04 — Pontos de Colocação Físicos Independentes dos Dados Supervisionados
1. **Correção de código**: `src/pinn_thermal.py` (`gerar_pontos_colocacao_independentes`, Commit `658a992`) e `experiments/validar_pinn.py` (Commit `561dc9a`).
2. **Teste automatizado**: `audit/tests/test_pinn_gradients.py` (Teste `T07`) e `experiments/validar_pinn.py`.
3. **Resultado numérico**: `audit/pinn/pinn_gradient_tests.json` e `audit/pinn/pinn_vs_mlp_comparison.json`:
   - Interseção de índices entre supervisão ($N_{\text{sup}} = 60$) e colocação ($N_{\text{col}} = 500$): `0` pontos comuns;
   - Distância temporal mínima entre conjuntos no teste `T07`: $2.2132\text{ s} > 0$.
4. **Interpretação científica**: Garante que a perda física $L_{\text{phys}}$ e a validação do resíduo da EDO ocorrem em instantes temporais estritamente disjuntos das observações ruidosas de temperatura.

### A05 — Comparação Controlada PINN ($w_{\text{phys}} > 0$) vs MLP Supervisionada Pura ($w_{\text{phys}} = 0$)
1. **Correção de código**: `experiments/validar_pinn.py` (Commit `561dc9a`). Comparação sob mesmos pesos iniciais (`seed=42`), mesmos dados esparsos ruidosos da Órbita 1 ($60$ pontos, $\sigma = 0.60\text{ K}$), mesmos $500$ pontos de colocação disjuntos e mesma época/taxa de aprendizado.
2. **Teste automatizado**: `experiments/validar_pinn.py`.
3. **Resultado numérico**: `audit/pinn/pinn_vs_mlp_comparison.json` e `audit/pinn/pinn_training_curves.csv`:
   - **MLP Supervisionada ($w_{\text{phys}} = 0$)**: MSE físico nos pontos de colocação $= 7.9777 \times 10^{-2}\text{ K}^2/\text{s}^2$; resíduo físico RMS na Órbita 2 (out-of-sample) $= 0.5259\text{ K/s}$; RMSE de temperatura (Órbita 2) $= 1.503\text{ K}$.
   - **PINN Equilibrada ($w_{\text{phys}} = 20\text{ s}^2$)**: MSE físico na colocação $= 5.8614 \times 10^{-2}\text{ K}^2/\text{s}^2$ (**redução de 26.53%** no resíduo da EDO); resíduo físico RMS na Órbita 2 $= 0.4655\text{ K/s}$; RMSE de temperatura $= 1.659\text{ K}$; deslocamento paramétrico $\|\theta_{\text{PINN}} - \theta_{\text{MLP}}\|_2 = 0.2334$.
   - **PINN Fortemente Física ($w_{\text{phys}} = 80\text{ s}^2$)**: MSE físico na colocação $= 3.4085 \times 10^{-2}\text{ K}^2/\text{s}^2$ (**redução de 57.28%** no resíduo da EDO); resíduo físico RMS na Órbita 2 $= 0.3783\text{ K/s}$; RMSE de temperatura $= 2.120\text{ K}$; deslocamento paramétrico $\|\theta_{\text{PINN}} - \theta_{\text{MLP}}\|_2 = 0.5419$.
4. **Interpretação científica**: Comprova que a regularização física atua genuinamente sobre os pesos aprendidos, reduzindo de forma monotônica ($26.5\% \to 57.3\%$) a violação da equação diferencial do balanço térmico nos pontos de colocação e na órbita futura, exibindo o clássico *trade-off* de Pareto entre fidelidade à EDO contínua e ajuste aos dados pontuais ruidosos. Delimitação: a PINN é um modelo substituto (*surrogate*) de avaliação física offline e não substitui o integrador Heun da planta nem o filtro EKF em malha fechada.

---

## Fase 2 — Validação Numérica do MPC / ADMM (A06, A07, A08)

### A06 — Validação do Solver ADMM contra Solver QP de Referência Independente (`SLSQP`)
1. **Correção de código**: `src/tcas_dvfs.py` e `audit/mpc/validate_mpc_admm.py` (Commit `917d2f3`).
2. **Teste automatizado**: `audit/mpc/validate_mpc_admm.py`.
3. **Resultado numérico**: `audit/mpc/mpc_validation_report.json`:
   - `C1_factivel_nominal` ($T_0 \approx 55^\circ\text{C}$, $P_{\text{bud}} = 115\text{ W}$): converge em $35$ iterações; $\|f_{\text{ADMM}} - f_{\text{ref}}\|_\infty = 1.40 \times 10^{-2}\text{ GHz}$; $\|s_{\text{ADMM}} - s_{\text{ref}}\|_\infty = 7.16 \times 10^{-7}\text{ K}$; erro relativo no objetivo $|\Delta J|/|J_{\text{ref}}| = 1.49 \times 10^{-5}$.
   - `C2_restricoes_ativas` ($T_0 \approx 79.5^\circ\text{C}$, $P_{\text{bud}} = 75\text{ W}$): converge em $20$ iterações; $\|f_{\text{ADMM}} - f_{\text{ref}}\|_\infty = 1.05 \times 10^{-3}\text{ GHz}$; $\|s_{\text{ADMM}} - s_{\text{ref}}\|_\infty = 1.99 \times 10^{-4}\text{ K}$; erro relativo no objetivo $= 1.82 \times 10^{-4}$.
   - `C3_quase_inviavel_soft_constraint` ($T_0 = 83.4^\circ\text{C} > T_{\text{cap}}$, slacks ativos $s^* \approx 0.42\text{ K}$): converge em $50$ iterações; $\|f_{\text{ADMM}} - f_{\text{ref}}\|_\infty = 4.62 \times 10^{-3}\text{ GHz}$; $\|s_{\text{ADMM}} - s_{\text{ref}}\|_\infty = 1.38 \times 10^{-4}\text{ K}$; erro relativo no objetivo $= 5.00 \times 10^{-4}$.
   - `C4_inviavel_fisico_orcamento` ($P_{\text{bud}} = 12\text{ W} < P_{\min,\text{cores}} = 19.34\text{ W}$): identificado e bloqueado preventivamente pelo cheque físico de `gov_tcas_dvfs` (`bloqueado_por_contingencia_em_gov_tcas_dvfs = True`), acionando `F_MIN` sem mascarar a inviabilidade.
   - **Warm Start vs Cold Start** ($15$ passos consecutivos): diferença máxima de frequência entre warm e cold start $= 3.91 \times 10^{-3}\text{ GHz}$.
4. **Interpretação científica**: Comprova que o ADMM em 60 iterações reproduz a solução ótima de referência do subproblema QP com erro de frequência da ordem de $1\text{ a }14\text{ MHz}$ ($0.1\%\text{ a }1\%$) e erro relativo de custo $< 0.05\%$.

### A07 — Condicionamento Numérico e Sensibilidade Paramétrica do QP
1. **Correção de código**: Equilibração de linhas de $A$ (`rn = np.abs(A).max(1)`) e análise espectral em `audit/mpc/validate_mpc_admm.py` (Commit `917d2f3`).
2. **Teste automatizado**: `audit/mpc/validate_mpc_admm.py`.
3. **Resultado numérico**: `audit/mpc/mpc_validation_report.json`:
   - $\kappa_2(P_m) = 198.63$;
   - $\kappa_2(A_{\text{raw}}) = 207.10 \to \kappa_2(A_{\text{norm}}) = 9.64$ (**melhoria de condicionamento de 21.5x** pela equilibração de linhas);
   - Sensibilidade avaliada para $\rho_a \in \{0.2, 0.5, 1.0, 2.0, 5.0, 10.0\}$, $\rho \in \{0.005, 0.02, 0.05, 0.10\}$ e $\lambda \in \{10, 50, 200, 500\}$, mantendo erro $\|f_{\text{ADMM}} - f_{\text{ref}}\|_\infty \le 2.1 \times 10^{-3}\text{ GHz}$ na faixa nominal $\rho_a \in [0.5, 2.0]$.
4. **Interpretação científica**: Demonstra que a matriz do sistema KKT/ADMM é bem-condicionada ($\kappa_2(K_{\text{ADMM}}) \approx 399$) graças à normalização de linhas de $A$.

### A08 — Erro da Predição Linearizada de 30 Passos contra a Planta Não Linear (Heun)
1. **Correção de código**: `audit/mpc/validate_mpc_admm.py` (Commit `917d2f3`).
2. **Teste automatizado**: `audit/mpc/validate_mpc_admm.py`.
3. **Resultado numérico**: `audit/mpc/mpc_nonlinear_prediction_error.csv` e `audit/mpc/mpc_validation_report.json`:
   - No passo de atuação imediata ($m=1$, $\Delta t_c = 2\text{ s}$): erro máximo entre predição linear e EDO não linear é de **$0.0422\text{ K}$** (`R1`), **$0.1411\text{ K}$** (`R2`) e **$0.3074\text{ K}$** (`R3`);
   - Em $10\text{ s}$ ($m=5$): erro máximo de $0.229\text{ K}$ (`R1`), $0.613\text{ K}$ (`R2`) e $1.490\text{ K}$ (`R3`);
   - Ao final de $60\text{ s}$ ($m=30$, malha aberta): o erro acumula até $1.63\text{ K}$ (`R1`), $4.56\text{ K}$ (`R2`) e $6.54\text{ K}$ (`R3`) devido à não-linearidade cúbica $P_{\text{dyn}} \propto V(f)^2 f$ e exponencial de $P_{\text{leak}}(T)$.
4. **Interpretação científica**: Comprova que a linearização sucessiva local (*Successive Local Linearization*) tem alta precisão no horizonte curto de atuação ($< 0.31\text{ K}$ em $2\text{ s}$, onde o controle $F[0]$ é efetivamente aplicado em horizonte móvel), e justifica por que o MPC realimenta o estado pelo EKF a cada $2\text{ s}$ e adota folga de segurança de $2.0\text{ K}$ ($T_{\text{cap}} = 83^\circ\text{C}$ frente ao limite de $85^\circ\text{C}$).

---

## Fase 3 — Fechamento Físico do Orçamento Térmico (A09, A10)

### A09 — Derivação Física Formal, Unidades e Sinais de $T_{\text{rad,max}}$ e $P_{\text{buffer}}$
1. **Correção de código**: `src/tcas_dvfs.py` e `audit/numerical_validation/validate_thermal_budget.py` (Commit `6807c40`). Incorporado o acoplamento condutivo-entálpico no horizonte:
   $$P_{\text{cond,coupled}} = \frac{P_{\text{cond,inst}} + (\tau_{\text{horiz}}/\tau_{\text{rad}}) Q_{\text{rad,liq}}}{1 + \tau_{\text{horiz}}/\tau_{\text{rad}}}, \quad P_{\text{cond,max}} = \min(P_{\text{cond,inst}}, P_{\text{cond,coupled}})$$
2. **Teste automatizado**: `audit/numerical_validation/validate_thermal_budget.py`.
3. **Resultado numérico**: `audit/numerical_validation/thermal_budget_validation.json`.
4. **Interpretação científica**: Garante consistência dimensional estrita ($\text{K}$, $\text{W}$, $\text{J/K}$, $\text{K/W}$, $\text{s}$) e contabiliza o aquecimento do radiador durante o horizonte de $240\text{ s}$ no limite de condução de Fourier.

### A10 — Comparação contra Integração Numérica Completa da EDO em 7 Regimes
1. **Correção de código**: `audit/numerical_validation/validate_thermal_budget.py` (Commit `6807c40`).
2. **Teste automatizado**: `audit/numerical_validation/validate_thermal_budget.py`.
3. **Resultado numérico**: `audit/numerical_validation/thermal_budget_validation.json`:
   - `R1_regime_estacionario`: $P_{\text{ana}} = 56.59\text{ W}$ vs $P_{\text{ODE}} = 58.15\text{ W}$ ($T_{\max,\text{sim}} = 82.01^\circ\text{C} \le 83^\circ\text{C}$);
   - `R2_aquecimento_transitorio`: $P_{\text{ana}} = 46.00\text{ W}$ vs $P_{\text{ODE}} = 47.67\text{ W}$ ($T_{\max,\text{sim}} = 81.93^\circ\text{C} \le 83^\circ\text{C}$);
   - `R3_resfriamento`: $P_{\text{ana}} = 38.38\text{ W}$ vs $P_{\text{ODE}} = 42.64\text{ W}$ ($T_{\max,\text{sim}} = 80.74^\circ\text{C} \le 83^\circ\text{C}$);
   - `R4_degrau_abrupto_carga`: $P_{\text{ana}} = 49.81\text{ W}$ vs $P_{\text{ODE}} = 51.00\text{ W}$ ($T_{\max,\text{sim}} = 82.24^\circ\text{C} \le 83^\circ\text{C}$);
   - `R5_eclipse_orbital`: $P_{\text{ana}} = 71.55\text{ W}$ vs $P_{\text{ODE}} = 74.30\text{ W}$ ($T_{\max,\text{sim}} = 81.25^\circ\text{C} \le 83^\circ\text{C}$);
   - `R6_incidencia_solar_zenite`: $P_{\text{ana}} = 30.76\text{ W}$ vs $P_{\text{ODE}} = 33.58\text{ W}$ ($T_{\max,\text{sim}} = 81.56^\circ\text{C} \le 83^\circ\text{C}$);
   - `R7_contingencia_energetica_saturacao`: $P_{\text{ana}} = 12.67\text{ W} < P_{\min,\text{cores}} = 22.86\text{ W}$ (`inviavel_fisico = True`).
4. **Interpretação científica**: Comprova que a aproximação analítica fechada é ligeiramente conservadora ($1.2\text{ a }4.2\text{ W}$ abaixo do limite numérico exato da EDO) e preserva $T_{\max} \le 83.0^\circ\text{C}$ em todos os 6 regimes fisicamente viáveis, enquanto detecta sem mascaramento a inviabilidade física no regime `R7`.

---

## Fase 4 — Auditoria do Escalonador SDP (A11, A12)

### A11 — Formulação Matemática, Resíduos e Correção de Posto da Projeção Afim (`pinv`)
1. **Correção de código**: `src/tcas_dvfs.py` (Commit `ef9b6ba`). Identificado e corrigido defeito numérico crítico: a matriz de restrições afins $E \in \mathbb{R}^{289 \times 289}$ possui posto $251$ (menor valor singular verdadeiro $\sigma_{251} = 0.3480$) e $38$ valores singulares algebricamente nulos oriundos de linhas redundantes de soma de linhas/colunas ($\sigma_{252} = 1.50 \times 10^{-14}$). O uso anterior de `np.linalg.pinv(E)` com `rcond=1e-15` invertia $\sigma_{252}$, gerando $\|E^+\|_2 = 2.27 \times 10^{14}$. Com `Ep = np.linalg.pinv(E, rcond=1e-10)`, obteve-se $\|E^+\|_2 = 2.873$ e erro de projeção $\|E(I - E^+ E)\|_2 = 5.83 \times 10^{-14}$.
2. **Teste automatizado**: `audit/sdp/validate_sdp_scheduler.py`.
3. **Resultado numérico**: `audit/sdp/sdp_validation_report.json`.
4. **Interpretação científica**: Restabelece a exatidão em precisão de máquina da projeção afim de Douglas-Rachford/ADMM sobre o politopo levantado de atribuição.

### A12 — Viabilidade Semidefinida, Restrições Afins e Comparação contra Enumeração Exata (24 Permutações)
1. **Correção de código**: `src/tcas_dvfs.py` e `audit/sdp/validate_sdp_scheduler.py` (Commit `ef9b6ba`).
2. **Teste automatizado**: `audit/sdp/validate_sdp_scheduler.py` (50 instâncias estocásticas em Warm Start e Cold Start).
3. **Resultado numérico**: `audit/sdp/sdp_validation_report.json`:
   - **Viabilidade PSD ($Z_1 \succeq 0$)**: $\min \lambda_{\min}(Z_1) = -1.08 \times 10^{-15}$ (`psd_strictly_satisfied_tol_1e_14 = True`);
   - **Viabilidade Afim ($E \operatorname{vec}(Y) = e$)**: violação máxima $\|E \operatorname{vec}(Y) - e\|_\infty = 5.22 \times 10^{-15}$ (`affine_satisfied_tol_1e_12 = True`);
   - **Não-negatividade ($Z_2 \ge 0$)**: $\min(Z_2) = 0.0$;
   - **Qualidade pós-arredondamento contra as 24 permutações**: taxa de acerto do ótimo global exato `exact_hit_rate_pct = 100.0%` (tanto em Warm Start quanto em Cold Start nas 50 instâncias, e `142/142 = 100.0%` na campanha orbital de 3 órbitas!), com `mean_heuristic_suboptimality_ratio = 5.74e-16`.
4. **Interpretação científica**: Separa rigorosamente viabilidade cônica/afim, convergência do ADMM e qualidade heurística pós-arredondamento, demonstrando que após a regularização de `pinv(E, rcond=1e-10)` a relaxação SDP seguida de arredondamento encontra a permutação globalmente ótima em 100% dos casos testados, sem necessitar reivindicar certificado dual aproximado.

---

## Fase 5 — Auditoria do EKF, Jacobiano e Barreira Térmica DSO (A13, A14, A17)

### A13 — Jacobiano Analítico nas Fronteiras de Saturação ($200\text{ K}, 500\text{ K}$, $F_{\min}, F_{\max}$)
1. **Correção de código**: `audit/numerical_validation/validate_ekf_and_dso.py` e `experiments/validacao_jacobiano.py` (Commit `0f53538`).
2. **Teste automatizado**: `audit/numerical_validation/validate_ekf_and_dso.py` e `experiments/validacao_jacobiano.py`.
3. **Resultado numérico**: `audit/numerical_validation/ekf_jacobian_dso_report.json`:
   - 100 pontos interiores: erro relativo máximo de Frobenius $= 5.069 \times 10^{-9} < 10^{-7}$;
   - Estados extremos fora do domínio ($160\text{ K} < 200\text{ K}$ e $540\text{ K} > 500\text{ K}$): erro relativo $< 2.5 \times 10^{-10}$;
   - Vizinhanças internas das fronteiras ($200.001\text{ K}$ e $499.999\text{ K}$) e extremos de frequência ($F_{\min}, F_{\max}$): erro relativo $< 5.1 \times 10^{-9}$;
   - Quinas exatas $T = 200.0\text{ K}$ e $T = 500.0\text{ K}$: derivada lateral saturada coincide com `jacobiano()` com erro $< 10^{-10}$, e o salto de subgradiente lateral é documentado e quantificado.
4. **Interpretação científica**: Comprova a exatidão do Jacobiano analítico em todo o domínio operacional e fora dele, documentando a descontinuidade de derivada lateral induzida por `np.clip(T, 200, 500)` nas quinas exatas $200\text{ K}$ e $500\text{ K}$.

### A14 — Observabilidade e Consistência Estatística do EKF (Forma de Joseph)
1. **Correção de código**: `audit/numerical_validation/validate_ekf_and_dso.py` (Commit `0f53538`).
2. **Teste automatizado**: `audit/numerical_validation/validate_ekf_and_dso.py`.
3. **Resultado numérico**: `audit/numerical_validation/ekf_jacobian_dso_report.json`:
   - **Observabilidade**: $\operatorname{rank}(\mathcal{O}_{\text{full}}) = 5/5$ e $\operatorname{rank}(\mathcal{O}_{\text{dies\_only}}) = 5/5$ (comprovando que mesmo sem medição direta no radiador, o 5º estado é estruturalmente observável a partir dos 4 dies pelo acoplamento condutivo $1/(R_r C_{\text{die}})$);
   - **Filtragem em 1 órbita ($11\,337$ passos)**: erro médio cai de $\text{RMSE}_{\text{bruto}} = 0.2493\text{ K}$ para $\text{RMSE}_{\text{EKF}} = 0.1070\text{ K}$ (**redução de 57.1% no ruído de medição**);
   - **Simetria e Positividade de Joseph**: $\max_k \|P_k - P_k^T\|_F = 1.00 \times 10^{-18}$ e $\min_k \lambda_{\min}(P_k) = 1.79 \times 10^{-2} > 0$;
   - **Inovação**: média próxima de zero ($|\bar{\nu}_i| < 2.5 \times 10^{-3}\text{ K}$) e $\text{NIS}_{\text{médio}} = 3.885$ (consistente e ligeiramente conservador frente a $\mathbb{E}[\chi^2_5] = 5.0$).
4. **Interpretação científica**: Comprova estabilidade numérica incondicional da atualização de covariância de Joseph e eficácia estatística da filtragem de estado.

### A17 — Validação Empírica da Barreira Térmica DSO (*Deterministic Safety Override*)
1. **Correção de código**: `audit/numerical_validation/validate_ekf_and_dso.py`.
2. **Teste automatizado**: `audit/numerical_validation/validate_ekf_and_dso.py` (Cenários `S1` a `S5`).
3. **Resultado numérico**: `audit/numerical_validation/ekf_jacobian_dso_report.json`:
   - `S4_descasamento_parametrico_inter_amostras_mpc` ($R_r \times 1.50$, $C_{\text{die}} \times 0.60$, atualização MPC a cada $10\text{ s}$): Sem DSO, $T_{\max} = 86.69^\circ\text{C}$ e tempo de violação $>85^\circ\text{C}$ é de **$449.5\text{ s}$**; Com DSO ($84.5^\circ\text{C}$), $T_{\max} = 85.09^\circ\text{C}$ e tempo $>85^\circ\text{C}$ cai para **$7.0\text{ s}$** (**redução de 98.4%** na duração de violação, com $982$ intervenções da barreira).
   - `S5_contingencia_falha_mpc_defesa_em_profundidade` (MPC inoperante): Sem DSO, ocorre disparo térmico até $365.01^\circ\text{C}$ ($597.0\text{ s}$ acima de $85^\circ\text{C}$); Com DSO, o pico é contido em $85.45^\circ\text{C}$.
   - `S2_radiador_saturado_77C_atuador_no_piso` e `S3_condicao_inicial_adversa_acima_84_5C`: quando o próprio MPC já saturou a frequência no piso físico $F_{\min} = 0.30\text{ GHz}$ ou a condição inicial parte acima da barreira ($84.9^\circ\text{C}$), ocorrem violações residuais ($22.0\text{ s}$ em `S3`), pois nenhum controlador por DVFS pode extrair calor abaixo da potência mínima $P_{\min,\text{cores}}$.
4. **Interpretação científica**: Comprova que a DSO atua efetivamente como mecanismo de defesa em profundidade contra distúrbios inter-amostras, erro de modelo e falha do MPC, ao mesmo tempo em que delimita honestamente que ela não constitui um conjunto positivamente invariante formal quando o fluxo térmico externo ou a condição inicial excedem a capacidade física de rejeição a $F_{\min}$.

---

## Fase 6 — Validação de Rainflow e Delimitação de Confiabilidade (A15, A16)

### A15 — Conformidade Normativa ASTM E1049-85 do Algoritmo Rainflow
1. **Correção de código**: `experiments/validacao_rainflow.py` e `audit/numerical_validation/validate_reliability_metrics.py` (Commit `ae73125`).
2. **Teste automatizado**: `audit/numerical_validation/validate_reliability_metrics.py` e `experiments/validacao_rainflow.py`.
3. **Resultado numérico**: `audit/numerical_validation/reliability_validation.json`:
   - Onda triangular $[0, 10, 0, 10, 0]$: $4$ meios-ciclos de $\Delta T = 10.0\text{ K}$ (soma $= 2.0$ ciclos);
   - Onda senoidal (amplitude $5\text{ K}$): ciclos fechados extraídos com range completo $\Delta T = 10.0\text{ K}$;
   - Sequência padrão ASTM E1049-85 Seção 5.4.4 ($[0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]$): multiconjunto extraído $\{\Delta T=2.0: 1.0,\; 3.0: 0.5,\; 4.0: 1.5,\; 6.0: 0.5,\; 8.0: 1.0,\; 9.0: 0.5\}$, soma $= 5.0$ ciclos (`conformidade_exata_astm = True`).
4. **Interpretação científica**: Comprova que `rainflow()` implementa fielmente o método de 3 pontos da ASTM E1049-85 retornando a excursão pico-a-vale $\Delta T$.

### A16 — Delimitação Física de Black e Coffin-Manson como Índices Relativos Adimensionais
1. **Correção de código**: `audit/numerical_validation/validate_reliability_metrics.py` (Commit `ae73125`).
2. **Teste automatizado**: `audit/numerical_validation/validate_reliability_metrics.py`.
3. **Resultado numérico**: `audit/numerical_validation/reliability_validation.json`:
   - Invariância algébrica à área transversal efetiva $A_{\text{eff}}$ e corrente de referência $J_{\text{ref}}$ na razão de Black: erro $= 4.44 \times 10^{-16}$;
   - Invariância algébrica entre uso de range ($\Delta T^q$) e semi-amplitude ($(\Delta T/2)^q$) na razão de Coffin-Manson: erro $= 8.88 \times 10^{-16}$.
4. **Interpretação científica**: Documenta que o proxy $I = P/V$ e os expoentes de Black ($n=1.8, E_a=0.7\text{ eV}$) e Coffin-Manson ($q \in \{1.9, 2.1\}$) produzem **índices comparativos relativos adimensionais** ($D_{\text{base}}/D_{\text{prop}}$) onde constantes geométricas e metalúrgicas do encapsulamento se cancelam, não devendo ser interpretados como predição absoluta de MTTF em horas.

---

## Fase 7 — Reprodutibilidade e Nova Campanha Experimental Validada (A18)

### A18 — Consolidação de Proveniência (`metadata.json`) e Regeneração Integral dos Resultados
1. **Correção de código**: `audit/reproducibility/generate_provenance.py`, `results/metadata.json` e `audit/reproducibility/campaign_provenance.json`.
2. **Teste automatizado e scripts executados**:
   - `experiments/run_convergencia_temporal.py`
   - `experiments/run_experimentos.py --orbitas 3 --orbitas-ablacao 1 --sementes 10 --saida results --workers 8`
   - `experiments/run_incerteza.py --amostras 30 --orbitas 3 --saida results --workers 6`
   - `experiments/gerar_figuras.py`
3. **Resultado numérico**:
   - **Convergência Temporal (`results/convergencia_temporal.json`)**: mantendo $\Delta t_{\text{MPC}} = 2.0\text{ s}$ e $\Delta t_{\text{SDP}} = 120.0\text{ s}$ invariantes, o erro absoluto em $T_{\max}$ frente à malha fina ($\Delta t = 0.125\text{ s}$, $T_{\max} = 83.177^\circ\text{C}$) cai monotonicamente: $0.062\text{ K}$ ($\Delta t=1.0\text{ s}$) $\to 0.034\text{ K}$ ($\Delta t=0.5\text{ s}$) $\to 0.005\text{ K}$ ($\Delta t=0.25\text{ s}$), e o erro relativo de energia cai de $0.0174\% \to 0.0114\% \to 0.0033\%$.
   - **Campanha Principal (`results/tabela1.csv` e `results/tabela_sementes.csv`)**: em 3 órbitas ($17\,004\text{ s}$, 10 sementes), o `TCAS-DVFS` obtém $T_{\max} = 83.24 \pm 0.02^\circ\text{C}$, tempo de violação acima de $85^\circ\text{C}$ igual a $0.0 \pm 0.0\text{ s}$, gradiente espacial médio $\Delta T_{\text{med}} = 0.885 \pm 0.001\text{ K}$ (redução de $83.2\%$ frente aos $5.257\text{ K}$ do `Throttling 85/75`), throughput de $124.62\text{ FPS}$, menor $\text{EDP} = 7.452\text{ kJ/FPS}$ e $100.0\%$ de acerto exato no escalonador SDP ($142/142$ chamadas).
   - **Campanha de Incerteza Paramétrica LHS (`results/incerteza.json` e `results/amostras.csv`)**: em 30 amostras de Hipercubo Latino ($\pm 20\%$ em $R_{ij}, R_r, C_{\text{die}}, C_{\text{rad}}$, 3 órbitas cada), o `TCAS-DVFS` registra **zero violações térmicas** ($0/30$ amostras acima de $85^\circ\text{C}$, limite superior unilateral Clopper-Pearson de $95\%$ igual a $9.50\%$), $T_{\max} = 82.36 \pm 1.58^\circ\text{C}$, redução pareada de $-82.61\%$ em $\Delta T_{\text{med}}$ ($\text{IC}_{95\%}: [-85.30\%, -79.92\%]$), redução pareada de $-78.74\%$ em $\Delta T_{\text{pico}}$ ($\text{IC}_{95\%}: [-79.20\%, -78.28\%]$), ganho pareado de $+4.08\%$ em FPS frente ao `Throttling 85/75` ($+4.95\%$ frente ao `Throttling 83/73`) e redução pareada de $-2.62\%$ no EDP.
   - **Rastreabilidade de Commit (`results/metadata.json`)**: `code_commit_hash` e `git_commit_hash` sincronizados com o commit do código validado, acompanhados dos hashes SHA-256 de todos os artefatos gerados.
4. **Interpretação científica**: Comprova que todas as conclusões experimentais do artigo permanecem válidas e numericamente mais sólidas após a correção integral das pendências físicas, matemáticas e algorítmicas identificadas na reauditoria.
