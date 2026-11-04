# REVISTA MULTIDISCIPLINAR DO NORDESTE MINEIRO — REMUNOM
**ISSN: 2178-6925 | Qualis CAPES B1 (Engenharias / Interdisciplinar)**

## CARTA DE RESPOSTA PONTO A PONTO AOS REVISORES E RELATÓRIO DE MODIFICAÇÕES
*(RESPONSE TO REVIEWERS & REVISION MANIFEST)*

---
- **Identificação da Submissão:** Artigo 01 — Submissão ID 8278 (Decisão: ACEITO - REVISÕES SÃO NECESSÁRIAS)
- **Título do Artigo:** *Thermal-Crosstalk-Aware Scheduling and Model Predictive Dynamic Voltage and Frequency Scaling for Spaceborne Multi-Chip AI Clusters in Deep Vacuum: Lumped Multiphysics Modeling, Predictive Control, and Numerical Simulation*
- **Autores:** Dafnhê Thor Castro de Martins Barros (Autor Correspondente, ProPECAut/IFES) e Flávio Barcelos Braz da Silva (Orientador / Professor Permanente, ProPECAut/IFES)
- **Instituição:** Instituto Federal do Espírito Santo (IFES) — Programa de Pós-Graduação em Engenharia de Controle e Automação (ProPECAut)
- **Data:** Novembro de 2026

---
### Preâmbulo

Prezados Editores e Ilustres Revisores da REMUNOM,

Agradecemos imensamente pelas valiosas contribuições e pelo parecer favorável com necessidade de revisões. Realizamos uma revisão exaustiva e integral do artigo, atendendo rigorosamente a todos os 15 apontamentos formulados pela banca examinadora.

### 1. Quadro-Síntese de Atendimento aos 15 Itens da Revisão por Pares

| Item | Requisito Editorial / Apontamento dos Revisores | Status | Seção do Artigo Atualizada |
| :---: | :--- | :---: | :--- |
| **1** | Formatação de equações nativas OMML, sem artefatos de escape e com unidades e dimensões explicadas. | **CONCLUÍDO** | Seções 1.2, 2.1, 2.2, 3.1, 3.2, 4.1-4.4, 5, 7.1-7.2 (Eqs. 1 a 29) |
| **2** | Detalhamento das cargas de trabalho sintéticas e hipótese de overhead nulo de migração / DVFS. | **CONCLUÍDO** | Seções 4.3 e 5, Tabela 2 |
| **3** | Restrição da incerteza LHS (n=30) estritamente aos parâmetros da planta física (Rij, Rr, ci, Crad). | **CONCLUÍDO** | Seção 6.5, Seção 9.2 |
| **4** | Interpretação conservadora de Clopper-Pearson (IC 95% = 11,57%) e intervalos pareados via teste t. | **CONCLUÍDO** | Seção 6.5 |
| **5** | Explicitação do mecanismo de segurança reativo (backstop a 84,5 °C) e hierarquia de saturação. | **CONCLUÍDO** | Seção 4.4, Seção 6.4 |
| **6** | Documentação da convergência do ADMM (18,4 iterações, folga s=0,00 K) e subotimalidade do SDP (34,5%). | **CONCLUÍDO** | Seção 4.3, Seção 4.4, Seção 6.4 |
| **7** | Reconciliação do pico de temperatura (83,21 °C vs 83,27 °C) e estudo formal de convergência em 4 passos. | **CONCLUÍDO** | Seção 5 |
| **8** | Definição formal da janela temporal de confiabilidade (descarte do warm-up de 2 h; 9.804 s estáveis). | **CONCLUÍDO** | Seção 7, Seção 7.2, Tabela 6 |
| **9** | Disponibilidade integral do código-fonte no GitHub e proveniência transparente das fontes de dados. | **CONCLUÍDO** | Data Availability, Seção 10, Tabela 2, Referências |
| **10** | Reenquadramento como trade-off multi-objetivo (zero violações e alto throughput) sem alegações de superioridade irrestrita. | **CONCLUÍDO** | Abstract, Resumo, Resumen, Seções 1.4, 6.1, 6.4, 10 |
| **11** | Delimitação estrita do escopo de sensibilidade (planta térmica vs exclusão de variações ambientais). | **CONCLUÍDO** | Seção 6.5, Seção 9.2 (Ameaça 5) |
| **12** | Justificativa dos limites de temperatura (85 °C como norma ECSS de carga útil; 125 °C como teto de validade FinFET). | **CONCLUÍDO** | Seção 3.1, Seção 5, Tabela 3 |
| **13** | Padronização do termo 'Relative MTTF proxy ratio' e explicitação de ausência de extensão de vida útil agregada (0,96 e 0,81). | **CONCLUÍDO** | Abstract, Resumo, Resumen, Seção 7.1, Seção 10 |
| **14** | Substituição de 'eliminação' por 'redução substancial' de fadiga térmica, reportando os 20 ciclos > 2,0 K restantes. | **CONCLUÍDO** | Abstract, Resumo, Resumen, Seção 7.2, Tabela 6, Figura 7 |
| **15** | Explicitação do significado operacional de FPS e EDP como proxies computacionais agregados da co-simulação. | **CONCLUÍDO** | Seção 5 (Eqs. 26 e 27) |

---

### 2. Respostas Detalhadas Ponto a Ponto (Point-by-Point Responses)

#### Item 1: Formatação Nativa de Equações Matemáticas e Clareza Dimensional

> **Apontamento dos Revisores / Checklist:**
> *O parecer editorial apontou a presença de artefatos de texto em equações (como '$h \≈ 0$' e espaços em branco indevidos em equações OMath), solicitando padronização e definição estrita de todos os símbolos, unidades físicas e dimensões matriciais.*

**Resposta dos Autores:**
Todas as 29 equações do manuscrito foram rigorosamente convertidas e geradas no formato nativo Office OpenXML / OMML (Word Equation Tool), eliminando 100% de quaisquer marcadores LaTeX residuais ou caracteres de controle indevidos no texto corrido. Todos os símbolos físicos, vetores de estado (ex: x ∈ ℝ⁵), matrizes de condutância (G_crosstalk ∈ ℝ⁴ˣ⁴), constantes de Stefan-Boltzmann e unidades no Sistema Internacional (W, J/K, K/W, GHz, V, eV) foram exaustivamente definidos logo após cada formulação.

**Modificação Textual Inserida no Manuscrito:**
> "Seções 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4, 5, 7.1 e 7.2 (Equações 1 a 29)."

---

#### Item 2: Detalhamento dos Modelos de Carga de Trabalho e Hipóteses de Migração

> **Apontamento dos Revisores / Checklist:**
> *Os revisores solicitaram que fossem detalhados os modelos de carga de inferência sintética (SAR-YOLO, Cloud-Seg, Hyper-ViT, Telemetry), as taxas de atividade α_i, e que fosse declarada expressamente a hipótese adotada para o overhead de migração de tarefas e chaveamento de tensão/frequência no DVFS.*

**Resposta dos Autores:**
As 4 cargas de trabalho sintéticas foram explicitadas na Seção 5 e Tabela 2, detalhando os volumes nominais de inferência (45, 60, 30 e 80 FPS) e a modelagem estocástica de rajadas via fatores de atividade contínuos α_i ∈ [0,4; 1,0] em processo Markoviano por partes (chaveamento a cada 30–120 s). Foi adicionada declaração expressa de que, no modelo concentrado de referência, os atrasos de transição elétrica de DVFS e o overhead de troca de contexto de tarefas (da ordem de microsegundos a dezenas de milissegundos) são assumidos negligenciáveis frente às constantes de tempo térmicas e aos períodos de controle (Δt_plant = 0,5 s; Δt_MPC = 2,0 s).

**Modificação Textual Inserida no Manuscrito:**
> ""Incoming neural inference workloads are mapped to dies at discrete allocation epochs (Δt_sched = 120 s). In this baseline lumped multi-chip module formulation, task migration overhead (context switching latency) and voltage/frequency transition settling times (typically microseconds to tens of milliseconds) are assumed negligible relative to the multi-second thermal time constants and control intervals (Δt_plant = 0.5 s, Δt_MPC = 2.0 s)." (Seção 4.3, Parágrafo 1; Seção 5, Parágrafo 3)."

---

#### Item 3: Restrição Estrita da Incerteza Paramétrica (LHS, n=30) à Planta Física

> **Apontamento dos Revisores / Checklist:**
> *Foi requerido que a campanha de sensibilidade por Hipercubo Latino (LHS, n=30) esclarecesse categoricamente se a incerteza de ±20% incidia apenas sobre a planta física ou também sobre o modelo interno do observador EKF e do controlador MPC.*

**Resposta dos Autores:**
Esclarecemos e documentamos na Seção 6.5 que as perturbações de ±20% foram aplicadas exclusivamente aos parâmetros físicos da planta térmica real simulada (resistências condutivas Rij, resistência radiador Rr, capacitâncias dos dies ci e do radiador Crad). O estimador de estados (EKF), o modelo interno do preditor (LTV MPC) e o cálculo do orçamento radiativo mantiveram seus parâmetros estritamente nominais. Essa escolha metodológica avalia precisamente a robustez do controle e a estabilidade do observador sob descasamento estrutural planta-modelo (model mismatch). A simetria geométrica planar foi preservada nas matrizes perturbadas.

**Modificação Textual Inserida no Manuscrito:**
> ""Crucially, perturbations were restricted strictly to the physical plant parameters, while the EKF observer, predictive control internal model, and radiative budget calculator retained their nominal parameter values. This setup specifically evaluates control robustness and observer stability under plant-model mismatch. Planar symmetry of the MCM packaging is preserved in the perturbed parameter matrices, and orbital environmental radiative forcing was held nominal to isolate structural thermodynamic sensitivities." (Seção 6.5, Parágrafo 1)."

---

#### Item 4: Interpretação Estatística Exata de Clopper-Pearson e Testes Pareados

> **Apontamento dos Revisores / Checklist:**
> *O relatório destacou que a ausência de violações térmicas (0/30) não autoriza a conclusão de 'probabilidade zero de falha', demandando a interpretação estrita do intervalo de Clopper-Pearson e a caracterização dos testes de hipótese pareados.*

**Resposta dos Autores:**
A redação foi integralmente ajustada para refletir modéstia epistêmica rigorosa: explicitou-se que o resultado de k=0 violações em n=30 realizações independentes estabelece um limite superior de confiança conservador e exato de 11,57% a 95% de nível de confiança bilateral, não implicando probabilidade real nula. Além disso, foram reportados os intervalos de confiança de 95% obtidos via teste t de Student pareado para as diferenças sob parâmetros físicos idênticos: contra Throttling 83/73, a redução de temperatura foi de 0,63 K (IC 95%: [-0,75; -0,50] K) com ganho de vazão de +5,26% (IC 95%: [+4,60%; +5,91%]); como o delta de EDP cruzou o zero (-0,36%, IC 95%: [-0,89%; +0,18%]), os autores abstêm-se de alegar superioridade estrita de EDP sob incerteza contra esse baseline conservador.

**Modificação Textual Inserida no Manuscrito:**
> ""No thermal-threshold violations were observed in the 30 sampled uncertainty realizations (k = 0, n = 30 samples > 85 °C). Under the hypothesis of 30 independent simulated trial realizations, the exact, conservative Clopper-Pearson binomial method yields a two-sided 95% confidence upper bound of 11.57% for the violation probability... Paired differences across identical physical parameter samples were evaluated using a two-sided paired Student's t-test at 95% confidence level..." (Seção 6.5, Parágrafos 2 e 3)."

---

#### Item 5: Mecanismo de Segurança Reativo (Backstop a 84,5 °C) e Saturação

> **Apontamento dos Revisores / Checklist:**
> *Os revisores solicitaram descrever a sequência operacional hierárquica entre a otimização MPC e o mecanismo de saturação/backstop de segurança, quantificando sua atuação durante as simulações nominais e ablativas.*

**Resposta dos Autores:**
A hierarquia foi formalmente detalhada na Seção 4.4: a cada época de controle (2,0 s), o MPC computa as frequências ótimas fi(k) limitadas à faixa admissível [0,30; 1,30] GHz. A cada passo de integração da planta (0,5 s), um algoritmo de segurança reativo rápido monitora a temperatura estimada pelo EKF (T̂i): caso T̂i > 84,5 °C, o backstop intervém, sobrepondo-se ao comando do MPC e impondo teto fi(t) ≤ max[fmin, fnom,i max(0,15; (85,0 - T̂i)/10,0)], travando em fmin caso T̂i ≥ 85,0 °C. No TCAS-DVFS nominal, o backstop interveio 0 vezes (duração de 0 s), pois o controle preditivo proativo manteve a temperatura máxima em 83,27 °C < 84,5 °C. Na variante ablativa 'Without MPC', o backstop atuou repetidamente, saturando os dies em fmin e acumulando 1.732,0 s de violações acima de 85 °C.

**Modificação Textual Inserida no Manuscrito:**
> ""The control execution follows a strict hierarchical sequence: at each control epoch k, the discrete LTV MPC-DVFS solves the condensed QP to produce unconstrained/box-clamped optimal frequencies f_i(k) ∈ [f_min, f_max]. At each physical plant integration step (Δt_plant = 0.5 s), a fast reactive safety backstop monitors the EKF estimated junction temperatures T̂_i(t)... In the nominal 3-orbit co-simulation of TCAS-DVFS, this fast safety backstop intervened 0 times (0 s duration)... Conversely, in the 'Without MPC' ablation variant, the safety backstop intervened repeatedly, forcing frequencies to f_min and accumulating 1,732.0 s above 85 °C." (Seção 4.4, Parágrafo 2)."

---

#### Item 6: Métricas de Convergência do ADMM, Variável de Folga e Subotimalidade do SDP

> **Apontamento dos Revisores / Checklist:**
> *Foi requerido documentar as estatísticas de convergência do algoritmo de divisão ADMM (iterações, resíduos primal/dual e relaxação de folga térmica) e a qualidade das soluções do alocador topológico SDP.*

**Resposta dos Autores:**
Incluímos na Seção 4.4 as estatísticas completas das 8.502 atualizações de controle executadas: o resolvedor ADMM convergiu em média em 18,4 iterações (máximo de 34, confortavelmente abaixo do teto de 60 iterações), com resíduo primal médio ||Ax - z||_∞ < 1,4 × 10⁻³ e resíduo dual < 8,2 × 10⁻⁴. A variável de folga térmica s convergiu para 0,00 K em todos os passos, comprovando que a restrição de 83,0 °C foi satisfeita sem necessidade de relaxação. Para o alocador SDP (Seção 4.3 e 6.4), documentamos que nas 142 épocas de alocação, o arredondamento por hiperplanos aleatórios coincidiu com o ótimo combinatório global exato (dentre as 24 permutações de 4 dies) em 49 chamadas (34,5%), apresentando um gap médio de subotimalidade de apenas 3,8% ± 2,1%.

**Modificação Textual Inserida no Manuscrito:**
> ""Across the 8,502 MPC control epochs in the 3-orbit co-simulation, the warm-started ADMM solver converged in an average of 18.4 iterations (maximum 34 iterations, well below the 60-iteration limit), with average primal residual ||Ax - z||_∞ < 1.4 × 10⁻³ and dual residual < 8.2 × 10⁻⁴. The thermal slack variable converged to s = 0.00 K throughout the nominal run... Across the 142 allocation epochs in the 3-orbit run, randomized hyperplane rounding of the SDP relaxation matched the true combinatorial minimum in 49 of 142 calls (34.5%), with an average suboptimality cost gap of 3.8% ± 2.1%." (Seções 4.3 e 4.4)."

---

#### Item 7: Reconciliação do Pico de Temperatura e Convergência Temporal Discreta

> **Apontamento dos Revisores / Checklist:**
> *O parecer solicitou esclarecer a aparente divergência entre os valores de temperatura máxima citados (83,21 °C vs 83,27 °C) e apresentar a convergência da malha de simulação.*

**Resposta dos Autores:**
A distinção foi plenamente esclarecida na Seção 5: o valor de 83,21 °C decorre do ensaio preliminar restrito a 1 única órbita, ao passo que 83,27 °C é o pico termicamente estabilizado ao longo da co-simulação multifísica completa de 3 órbitas LEO (17.004,0 s), induzido pelo ligeiro acúmulo cíclico de fluxo solar e albedo no núcleo do radiador. Apresentamos ainda os resultados da análise de convergência temporal formal em 4 passos: Δt = 1,0 s (83,38 °C); Δt = 0,5 s (83,27 °C); Δt = 0,25 s (83,18 °C); e Δt = 0,125 s (83,12 °C). A diferença entre o passo nominal adotado (0,5 s) e o passo mais refinado (0,125 s) é de apenas 0,15 K — patamar inferior ao próprio ruído intrínseco dos sensores (σ_s = 0,25 K) —, garantindo estabilidade numérica estrita.

**Modificação Textual Inserida no Manuscrito:**
> ""We clarify that the single-orbit preliminary convergence test reached a peak temperature of 83.21 °C, whereas the full 3-orbit co-simulation at nominal Δt_plant = 0.5 s reached 83.27 °C due to slight cumulative multi-orbit solar and albedo heat accumulation in the radiator core. A formal temporal convergence study across Δt_plant ∈ {1.0, 0.5, 0.25, 0.125} s (with invariant control step Δt_MPC = 2.0 s) confirmed asymptotic monotonic convergence: Δt = 1.0 s yields T_max = 83.38 °C; Δt = 0.5 s yields T_max = 83.27 °C; Δt = 0.25 s yields T_max = 83.18 °C; and Δt = 0.125 s yields T_max = 83.12 °C." (Seção 5, Parágrafo 1)."

---

#### Item 8: Janela Temporal de Confiabilidade e Descarte de Transiente Inicial (Warm-up)

> **Apontamento dos Revisores / Checklist:**
> *Os revisores questionaram a inclusão do transiente inicial nas análises de confiabilidade física e solicitaram a definição inequívoca da janela temporal estável utilizada.*

**Resposta dos Autores:**
Definiu-se rigorosamente na Seção 7 que o tempo total de simulação é de 17.004,0 s (34.009 passos, ≈ 4,72 horas, correspondendo a 3 órbitas completas). As primeiras 2 horas (7.200 s, 14.400 passos, ≈ 1,27 órbitas) foram estritamente descartadas como fase de aquecimento e estabilização orbital (warm-up), período necessário para que a estrutura do radiador atinja seu ciclo térmico quase-estacionário entre 265 K e 282 K. As projeções de eletromigração e a contagem de ciclos por Rainflow (ASTM E1049-85 com emparelhamento de meios-ciclos) incidem estritamente sobre os 9.804,0 s subsequentes (≈ 2,72 horas, 1,73 órbitas estáveis).

**Modificação Textual Inserida no Manuscrito:**
> ""The total simulation duration is 17,004.0 s (34,009 steps, ≈ 4.72 h, spanning 3 complete orbits). The first 2 hours (7,200 s, 14,400 steps, ≈ 1.27 orbits) are discarded as the initial transient warm-up phase, during which the cold-launched radiator core warms up from 270 K and stabilizes into cyclical orbital thermal equilibrium between 265 K (eclipse) and 282 K (sunlit). All reliability projections (Black's equation and Rainflow cycle counting) are evaluated strictly over the remaining steady-state window of 9,804.0 s (≈ 2.72 h, 1.73 orbits), and are presented strictly as indicative comparative proxies." (Seção 7, Parágrafo 1)."

---

#### Item 9: Disponibilidade do Código e Segregação Transparente de Fontes de Dados

> **Apontamento dos Revisores / Checklist:**
> *Foi requerido fornecer o link do repositório público de reprodutibilidade e segregar explicitamente a origem dos parâmetros entre normas espaciais, literatura empírica de semicondutores e dados sintéticos.*

**Resposta dos Autores:**
O repositório público de acesso aberto no GitHub foi devidamente referenciado na seção 'Data Availability': https://github.com/dafnhe-thor-barros/TCAS-DVFS. Na Seção 1.4 (Tabela 1) e Seção 4.4 (Tabela 2), foi feita a segregação transparente entre: (i) normas consolidadas de engenharia térmica espacial (ECSS-E-ST-31C e NASA-HDBK-7009B); (ii) literatura empírica de FinFET para dimensões e correntes de fuga (Kang et al., Skadron et al., Wang et al.); e (iii) cargas de trabalho e perfis radiativos orbitais sintetizados numericamente.

**Modificação Textual Inserida no Manuscrito:**
> ""All simulation source code, execution pipelines, nominal parameter manifests, co-simulation logs, and computational reproducibility metadata are openly available in the GitHub repository at: https://github.com/dafnhe-thor-barros/TCAS-DVFS." (Data Availability; Seção 1.4, Tabela 1; Seção 4.4, Tabela 2)."

---

#### Item 10: Reenquadramento como Trade-Off Multi-Objetivo vs. Superioridade Uniforme

> **Apontamento dos Revisores / Checklist:**
> *O relatório editorial orientou eliminar alegações de 'superioridade absoluta' ou 'domínio irrestrito', recontextualizando a proposta sob a ótica de um trade-off multi-objetivo equilibrado.*

**Resposta dos Autores:**
Todas as passagens do resumo, introdução, resultados e conclusões foram reescritas com rigorosa modéstia epistêmica: o TCAS-DVFS é agora caracterizado como uma solução de compromisso multi-objetivo que prioriza o confinamento térmico estrito (zero violações do limiar de 85 °C) aliado a alta vazão computacional (124,38 FPS). Reconheceu-se explicitamente que o baseline TCA simples atinge um gradiente espacial de pico ligeiramente menor (1,58 K vs 2,11 K) ao custo de reduzir a vazão (116,16 FPS), que o Space-PID entrega menor gradiente médio (0,62 K vs 0,80 K) com violações térmicas temporárias (14,0 s), e que a variante 'Without SDP' obtém vazão marginalmente superior (125,16 FPS) em matrizes 2x2 simétricas por não incorrer em reorganizações dinâmicas.

**Modificação Textual Inserida no Manuscrito:**
> ""Importantly, these empirical results characterize TCAS-DVFS as a multi-objective trade-off: it prioritizes zero thermal-threshold violations (0 s > 85 °C) and high inference throughput (124.38 FPS), while acknowledging that simple TCA achieves a slightly lower peak spatial gradient (1.58 K vs 2.11 K) at the cost of lower throughput (116.16 FPS), Space-PID achieves a lower mean gradient (0.62 K vs 0.80 K) with transient overshoots, and the 'Without SDP' variant achieves marginally higher nominal throughput (125.16 FPS) in the symmetric N=4 topology." (Abstract; Seção 6.1, Parágrafo 2; Seção 10)."

---

#### Item 11: Delimitação Estrita do Escopo de Robustez Paramétrica

> **Apontamento dos Revisores / Checklist:**
> *Solicitou-se explicitar os limites de validade da amostragem LHS, esclarecendo quais incertezas físicas foram modeladas e quais foram intencionalmente excluídas.*

**Resposta dos Autores:**
A Seção 6.5 e a Seção 9.2 (Ameaça à Validade 5) especificam textualmente que a campanha de incerteza por LHS limitou-se estritamente à variação simultânea de ±20% nos 4 parâmetros termo-estruturais da planta (Rij, Rr, ci, Crad). Fatores ambientais e radiativos externos — tais como variações sazonais do fluxo solar, flutuações de albedo terrestre, degradação óptica da emissividade do radiador e anomalias de atitude do satélite — foram deliberadamente mantidos nominais para isolar a sensibilidade estrutural do conjunto, ficando reservados para investigações multifísicas subsequentes.

**Modificação Textual Inserida no Manuscrito:**
> ""5. Parametric Robustness Scope: The LHS sensitivity campaign evaluates variations across the 4 internal thermal parameters (R_ij, R_r, c_i, C_rad), reserving orbital, solar, and optical degradation uncertainties for future multi-parameter investigations." (Seção 6.5; Seção 9.2, Ameaça 5)."

---

#### Item 12: Fundamentação Física dos Limites Térmicos Operacionais (85 °C e 125 °C)

> **Apontamento dos Revisores / Checklist:**
> *Foi apontada a necessidade de justificar tecnicamente a escolha de 85 °C como teto térmico e explicar por que regimes acima de 125 °C foram marcados como não físicos.*

**Resposta dos Autores:**
Fundamentou-se na Seção 3.1 e Seção 5 que o limite de 85,0 °C é a especificação clássica de qualificação operacional para eletrônica espacial COTS embarcada sob as diretrizes da norma ECSS-E-ST-31C, assegurando margem de confiabilidade em missões de pequeno porte. Por outro lado, o teto de 125,0 °C representa a fronteira máxima de calibração física dos modelos analíticos de corrente de fuga subthreshold em processos FinFET de 16/12 nm encontrados na literatura. Regimes onde baselines sem controle (como o Ondemand desregulado) atingem temperaturas teóricas extremas (412,30 °C) representam extrapolações matemáticas de fuga exponencial positiva, devendo ser estritamente excluídos de comparações quantitativas diretas.

**Modificação Textual Inserida no Manuscrito:**
> ""The thermal threshold T_cap = 85.0 °C is adopted as a typical operational qualification limit in small-satellite payload engineering guidelines (ECSS-E-ST-31C) to maintain safety margins for commercial off-the-shelf electronic components in space missions, whereas 125.0 °C represents the physical upper boundary of validity for standard FinFET subthreshold leakage models documented in solid-state literature (Kang et al. 1; Skadron et al. 94)." (Seção 3.1; Seção 5; Tabela 3, nota de rodapé)."

---

#### Item 13: Padronização Terminológica de Eletromigração e Reconhecimento de Não Extensão de Vida Útil

> **Apontamento dos Revisores / Checklist:**
> *Os revisores alertaram para a incorreção de sugerir 'extensão global de vida útil por eletromigração' diante de razões de dano relativo próximas ou inferiores a 1,0.*

**Resposta dos Autores:**
A terminologia foi padronizada em todo o texto para 'Relative Mean Time to Failure (MTTF) proxy ratio' (MTTF_prop / MTTF_ref). Explicitou-se no Abstract, na Seção 7.1 e nas Conclusões que, sob a equação de Black integrada no regime permanente, o TCAS-DVFS apresenta razões relativas agregadas de 0,96 (contra Throttling 85/75) e 0,81 (contra Throttling 83/73). Esse comportamento decorre do fato de a proposta sustentar maior execução útil (+7,6% e +9,9% de corrente média consumida), concluindo-se formalmente que não há extensão de vida útil agregada frente à eletromigração. O benefício do controle reside na proteção seletiva do nó individual mais crítico (razão relativa de 1,52 e 1,33), evitando picos pontuais de envelhecimento acelerado.

**Modificação Textual Inserida no Manuscrito:**
> ""Because TCAS-DVFS sustains higher inference execution rates, drawing +7.6% and +9.9% higher average current across all dies, we formally conclude that the framework provides no net electromigration life extension on an aggregate basis, while mitigating localized accelerated aging on the hottest individual die by eliminating thermal spikes." (Abstract; Resumo; Resumen; Seção 7.1, Parágrafo 1; Seção 10)."

---

#### Item 14: Adequação Terminológica sobre Fadiga Térmica e Detalhamento de Ciclos Residuais

> **Apontamento dos Revisores / Checklist:**
> *Foi requerido suavizar afirmações de 'eliminação total de ciclos térmicos' e explicitar a existência de ciclos residuais sob a norma ASTM E1049-85.*

**Resposta dos Autores:**
A linguagem foi devidamente moderada: substituiu-se o termo 'eliminação' por 'redução substancial' de excursões térmicas de grande amplitude. Esclareceu-se na Seção 7.2 e Tabela 6 que, após o período de warm-up, o número de ciclos térmicos com amplitude ΔT > 2,0 K foi reduzido de 612 no Throttling para 20 no TCAS-DVFS (queda de 96,7%). Reforçou-se enfaticamente que a variação de temperatura de junção ΔT funciona como um indicador proxy de fadiga em solda SAC305, e não como uma medição direta de deformação plástica por cisalhamento em esferas de BGA.

**Modificação Textual Inserida no Manuscrito:**
> ""Rather than eliminating all thermal excursions, TCAS-DVFS substantially reduces large thermal excursions: cycles with ΔT > 2.0 K are reduced from 612 in Throttling to 20 under TCAS-DVFS (a 96.7% reduction). We emphasize that junction temperature range ΔT serves as an indicative thermal fatigue proxy, rather than a direct calculation of inelastic shear strain in SAC305 solder balls." (Abstract; Resumo; Seção 7.2, Parágrafo 2; Tabela 6)."

---

#### Item 15: Significado Operacional de FPS e do Produto Energia-Atraso (EDP)

> **Apontamento dos Revisores / Checklist:**
> *Solicitou-se explicitar formalmente o significado físico das métricas agregadas de desempenho e eficiência computacional adotadas.*

**Resposta dos Autores:**
A Seção 5 (Equações 26 e 27) detalha minuciosamente que a métrica FPS representa um proxy de vazão ponderado linearmente pela frequência operacional dos aceleradores (FPS = 1/N_steps ∑_t ∑_i (fi(t)/fmax) FPSnom,i), traduzindo a capacidade de processamento instantânea média disponibilizada e não a latência de execução no nível de instruções ou ciclos de clock. Da mesma forma, o produto energia-atraso Et_frame = E · (1/FPS) [kJ·s/quadro] atua como um proxy global de eficiência, relacionando a energia elétrica total dissipada durante os 17.004,0 s de co-simulação à taxa agregada de entrega de quadros.

**Modificação Textual Inserida no Manuscrito:**
> ""Aggregate inference throughput is evaluated via the frequency-scaled performance proxy... which reflects aggregate computational capacity rather than cycle-accurate kernel instruction latency. Similarly, aggregate energy efficiency is quantified by the throughput-normalized energy-delay product proxy E_t_frame = E · (1/FPS) [kJ·s/frame], representing cumulative energy expenditure weighted by inverse throughput across the complete co-simulation duration..." (Seção 5, Parágrafo 3; Equações 26 e 27)."

---

### 3. Considerações Finais e Assinatura

Reiteramos nosso compromisso com o rigor científico e colocamo-nos à disposição para quaisquer esclarecimentos adicionais.

**Dafnhê Thor Castro de Martins Barros** (dafnhe.barros@msn.com)

**Prof. Dr. Flávio Barcelos Braz da Silva** (flavio.silva@ifes.edu.br)

Programa de Pós-Graduação em Engenharia de Controle e Automação (ProPECAut) — IFES