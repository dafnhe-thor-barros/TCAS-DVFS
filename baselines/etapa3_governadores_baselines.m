% =========================================================================
% PROJETO DE MESTRADO: CONTROLE ESTATÍSTICO/PREDITIVO DE CLUSTERS DE IA
% ETAPA 3 (BLINDADA): Implementação e Auto-Tuning dos Governadores Baselines
%          1. Linux Ondemand (Sem consciência térmica)
%          2. Thermal Throttling Puro Reativo (Histerese 85°C / 75°C)
%          3. Space-PID Descentralizado com Auto-Tuning Sistemático (ITAE)
%          4. Global Power Cap Estático (Cluster P_cap = 65 W)
%
% Métricas Expandidas: Throughput (FPS), Latência, EDP, t_viol, E_total, E_leak
% Compatibilidade: GNU Octave >= 7.0 e MATLAB >= R2020b
% =========================================================================
function etapa3_governadores_baselines()

close all; clc;

fprintf('===================================================================\n');
fprintf(' ETAPA 3 (BLINDADA): Governadores Baselines com FPS e EDP\n');
fprintf('===================================================================\n');

%% 1. CARREGAMENTO DOS DADOS DA PLANTA (ETAPA 2)
script_dir = '.';
try
    [s_dir, ~, ~] = fileparts(mfilename('fullpath'));
    if ~isempty(s_dir)
        script_dir = s_dir;
    end
catch
    script_dir = '.';
end

mat_etapa2 = fullfile(script_dir, 'etapa2_dados_planta.mat');
if ~exist(mat_etapa2, 'file')
    if exist('etapa2_dados_planta.mat', 'file')
        mat_etapa2 = 'etapa2_dados_planta.mat';
    else
        error('Arquivo "%s" não encontrado! Execute a Etapa 2 primeiro.', mat_etapa2);
    end
end

fprintf('-> Carregando parâmetros e planta de: %s\n', mat_etapa2);
dados_p = load(mat_etapa2);

t              = dados_p.t;
dt             = dados_p.dt;
Nt             = dados_p.Nt;
N_nodes        = dados_p.N_nodes;
alpha_workload = dados_p.alpha_workload;
f_nominal      = dados_p.f_nominal;

% Fluxos térmicos orbitais separados (ECSS/NASA)
Q_solar        = dados_p.Q_solar;
Q_albedo       = dados_p.Q_albedo;
Q_IR           = dados_p.Q_IR;

% Parâmetros da planta e radiador
R_mat          = dados_p.R_mat;
R_r_node       = dados_p.R_r_node;
C_dies         = dados_p.C_dies; % Rigorosamente 18.5 J/K
C_rad          = dados_p.C_rad;
V_min          = dados_p.V_min;
V_max          = dados_p.V_max;
f_min          = dados_p.f_min;
f_max          = dados_p.f_max;
beta_delay     = dados_p.beta_delay;
C_eff          = dados_p.C_eff;
P_leak_ref     = dados_p.P_leak_ref;
E_a            = dados_p.E_a;
T0_ref         = dados_p.T0_ref;
V0_ref         = dados_p.V0_ref;
sigma          = dados_p.sigma;
T_space        = 2.73;
epsilon        = 0.88;
A_rad          = 0.12;
k_B_eV         = 8.617333262e-5;

% Parâmetros de throughput dos modelos de IA embarcados (FPS nominal a 1.30 GHz)
% Nó 1: SAR-YOLOv8 (Detecção de Alvos) -> ~45 FPS nominais
% Nó 2: Cloud-Seg (Segmentação de Nuvens) -> ~60 FPS nominais
% Nó 3: Hyper-ViT (Classificação Hiperespectral) -> ~30 FPS nominais
% Nó 4: Housekeeping / Telemetria -> ~80 FPS nominais
FPS_nom_dies = [45.0, 60.0, 30.0, 80.0];

% Funções eletrotérmicas dos semicondutores
calc_V = @(f_val) max(V_min, min(V_max, V_min + beta_delay * (f_val - f_min)));
calc_Pdyn = @(alpha_val, f_val, V_val) (alpha_val * C_eff * (V_val^2) * (f_val * 1e9));
calc_Pleak = @(T_val, V_val) ...
    (P_leak_ref * ((V_val / V0_ref)^1.5) * ((max(200.0, min(500.0, T_val)) / T0_ref)^2) * ...
     exp((E_a / k_B_eV) * (1.0 / T0_ref - 1.0 / max(200.0, min(500.0, T_val)))));

% Condições iniciais
X0 = [310.0; 310.0; 310.0; 310.0; dados_p.T_rad_C(1) + 273.15];

%% 2. AUTO-TUNING SISTEMÁTICO DO SPACE-PID (CRITÉRIO ITAE)
fprintf('-> Executando Auto-Tuning do Space-PID (Otimização por Grade ITAE)...\n');

PID_params.T_target_C = 80.0;

Kp_candidates = [-0.025, -0.045, -0.070];
Ki_candidates = [-0.0005, -0.0020, -0.0050];
Kd_candidates = [-0.040, -0.080, -0.120];

best_itae = Inf;
best_Kp   = -0.045;
best_Ki   = -0.0020;
best_Kd   = -0.080;

Nt_tune = min(1200, Nt);
for kp_c = Kp_candidates
    for ki_c = Ki_candidates
        for kd_c = Kd_candidates
            X_t = zeros(Nt_tune, 5); X_t(1, :) = X0';
            itae_acc = 0.0;
            int_err  = zeros(1, N_nodes);
            p_err    = zeros(1, N_nodes);
            
            for kt = 1:(Nt_tune - 1)
                T_c = X_t(kt, 1:4) - 273.15;
                e_t = T_c - PID_params.T_target_C;
                int_err = max(-200, min(200, int_err + e_t * dt));
                d_err   = (e_t - p_err) / dt;
                p_err   = e_t;
                
                u_t = kp_c * e_t + ki_c * int_err + kd_c * d_err;
                f_t = min(f_max, max(f_min, f_nominal(kt, :) + u_t));
                
                viol_penal = sum(max(0, T_c - 85.0) .^ 2) * 50.0;
                itae_acc = itae_acc + (t(kt) * sum(abs(e_t)) + viol_penal) * dt;
                
                X_kt = X_t(kt, :)';
                [dX1_t, ~, ~, ~] = eval_planta(X_kt, kt, f_t, N_nodes, R_mat, R_r_node, ...
                    C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
                    alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
                X_t(kt+1, :) = (X_kt + dt * dX1_t)';
            end
            
            if itae_acc < best_itae
                best_itae = itae_acc;
                best_Kp   = kp_c;
                best_Ki   = ki_c;
                best_Kd   = kd_c;
            end
        end
    end
end

PID_params.Kp = best_Kp;
PID_params.Ki = best_Ki;
PID_params.Kd = best_Kd;
fprintf('   [Auto-Tuning Concluído]: Kp = %.4f | Ki = %.4f | Kd = %.4f\n', ...
        PID_params.Kp, PID_params.Ki, PID_params.Kd);

%% 3. ESTRUTURA PARA ARMAZENAMENTO DOS RESULTADOS
gov_names = {'Ondemand', 'Throttling', 'Space-PID', 'Power-Cap'};
N_gov = length(gov_names);

res = struct();
for g = 1:N_gov
    res.(gov_names{g}).T_dies_C    = zeros(Nt, N_nodes);
    res.(gov_names{g}).T_rad_C     = zeros(Nt, 1);
    res.(gov_names{g}).f_exec      = zeros(Nt, N_nodes);
    res.(gov_names{g}).P_tot_clust = zeros(Nt, 1);
    res.(gov_names{g}).P_leak_tot  = zeros(Nt, 1);
    res.(gov_names{g}).delta_T_esp = zeros(Nt, 1);
    res.(gov_names{g}).FPS_instant = zeros(Nt, N_nodes);
end

%% =========================================================================
%% 4. SIMULAÇÃO DO GOVERNADOR 1: LINUX ONDEMAND
%% =========================================================================
fprintf('\n[1/4] Simulando Governador: Linux Ondemand...\n');
X_ond = zeros(Nt, 5); X_ond(1, :) = X0';
f_ond = zeros(Nt, N_nodes);
P_ond = zeros(Nt, 1);
P_l_ond = zeros(Nt, 1);

for k = 1:(Nt - 1)
    f_k = f_min + alpha_workload(k, :) .* (f_max - f_min);
    f_ond(k, :) = f_k;
    
    X_k = X_ond(k, :)';
    [dX1, ~, P_lk1, P_tot1] = eval_planta(X_k, k, f_k, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    P_ond(k)   = sum(P_tot1);
    P_l_ond(k) = sum(P_lk1);
    
    X_pred = X_k + dt * dX1;
    f_next = f_min + alpha_workload(k+1, :) .* (f_max - f_min);
    [dX2, ~, ~, ~] = eval_planta(X_pred, k+1, f_next, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    
    X_ond(k+1, :) = (X_k + (dt / 2.0) * (dX1 + dX2))';
end
f_ond(Nt, :) = f_min + alpha_workload(Nt, :) .* (f_max - f_min);
[~, ~, P_lk_end, P_tot_end] = eval_planta(X_ond(Nt, :)', Nt, f_ond(Nt, :), N_nodes, R_mat, R_r_node, ...
    C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
    alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
P_ond(Nt)   = sum(P_tot_end);
P_l_ond(Nt) = sum(P_lk_end);

res.Ondemand.T_dies_C    = X_ond(:, 1:4) - 273.15;
res.Ondemand.T_rad_C     = X_ond(:, 5) - 273.15;
res.Ondemand.f_exec      = f_ond;
res.Ondemand.P_tot_clust = P_ond;
res.Ondemand.P_leak_tot  = P_l_ond;
res.Ondemand.delta_T_esp = max(res.Ondemand.T_dies_C, [], 2) - min(res.Ondemand.T_dies_C, [], 2);
res.Ondemand.FPS_instant = (f_ond ./ f_max) .* repmat(FPS_nom_dies, Nt, 1);

%% =========================================================================
%% 5. SIMULAÇÃO DO GOVERNADOR 2: THERMAL THROTTLING REATIVO (85°C / 75°C)
%% =========================================================================
fprintf('[2/4] Simulando Governador: Thermal Throttling Puro (Histerese)...\n');
X_thr = zeros(Nt, 5); X_thr(1, :) = X0';
f_thr = zeros(Nt, N_nodes);
P_thr = zeros(Nt, 1);
P_l_thr = zeros(Nt, 1);
throttled_state = false(1, N_nodes);

for k = 1:(Nt - 1)
    T_curr_C = X_thr(k, 1:4) - 273.15;
    
    for i = 1:N_nodes
        if T_curr_C(i) >= 85.0
            throttled_state(i) = true;
        elseif T_curr_C(i) <= 75.0
            throttled_state(i) = false;
        end
    end
    
    f_k = f_nominal(k, :);
    f_k(throttled_state) = f_min;
    f_thr(k, :) = f_k;
    
    X_k = X_thr(k, :)';
    [dX1, ~, P_lk1, P_tot1] = eval_planta(X_k, k, f_k, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    P_thr(k)   = sum(P_tot1);
    P_l_thr(k) = sum(P_lk1);
    
    X_pred = X_k + dt * dX1;
    T_pred_C = X_pred(1:4) - 273.15;
    throttled_pred = throttled_state;
    for i = 1:N_nodes
        if T_pred_C(i) >= 85.0
            throttled_pred(i) = true;
        elseif T_pred_C(i) <= 75.0
            throttled_pred(i) = false;
        end
    end
    f_pred = f_nominal(k+1, :);
    f_pred(throttled_pred) = f_min;
    
    [dX2, ~, ~, ~] = eval_planta(X_pred, k+1, f_pred, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    
    X_thr(k+1, :) = (X_k + (dt / 2.0) * (dX1 + dX2))';
end
f_thr(Nt, :) = f_thr(Nt - 1, :);
[~, ~, P_lk_end, P_tot_end] = eval_planta(X_thr(Nt, :)', Nt, f_thr(Nt, :), N_nodes, R_mat, R_r_node, ...
    C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
    alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
P_thr(Nt)   = sum(P_tot_end);
P_l_thr(Nt) = sum(P_lk_end);

res.Throttling.T_dies_C    = X_thr(:, 1:4) - 273.15;
res.Throttling.T_rad_C     = X_thr(:, 5) - 273.15;
res.Throttling.f_exec      = f_thr;
res.Throttling.P_tot_clust = P_thr;
res.Throttling.P_leak_tot  = P_l_thr;
res.Throttling.delta_T_esp = max(res.Throttling.T_dies_C, [], 2) - min(res.Throttling.T_dies_C, [], 2);
res.Throttling.FPS_instant = (f_thr ./ f_max) .* repmat(FPS_nom_dies, Nt, 1);

%% =========================================================================
%% 6. SIMULAÇÃO DO GOVERNADOR 3: SPACE-PID (COM AUTO-TUNING OTIMIZADO)
%% =========================================================================
fprintf('[3/4] Simulando Governador: Space-PID Descentralizado...\n');
X_pid = zeros(Nt, 5); X_pid(1, :) = X0';
f_pid = zeros(Nt, N_nodes);
P_pid = zeros(Nt, 1);
P_l_pid = zeros(Nt, 1);

integral_err = zeros(1, N_nodes);
prev_err     = zeros(1, N_nodes);

for k = 1:(Nt - 1)
    T_curr_C = X_pid(k, 1:4) - 273.15;
    err = T_curr_C - PID_params.T_target_C;
    
    integral_err = max(-300, min(300, integral_err + err * dt));
    deriv_err    = (err - prev_err) / dt;
    prev_err     = err;
    
    u_pid = PID_params.Kp * err + PID_params.Ki * integral_err + PID_params.Kd * deriv_err;
    f_k   = min(f_max, max(f_min, f_nominal(k, :) + u_pid));
    f_pid(k, :) = f_k;
    
    X_k = X_pid(k, :)';
    [dX1, ~, P_lk1, P_tot1] = eval_planta(X_k, k, f_k, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    P_pid(k)   = sum(P_tot1);
    P_l_pid(k) = sum(P_lk1);
    
    X_pred = X_k + dt * dX1;
    [dX2, ~, ~, ~] = eval_planta(X_pred, k+1, f_k, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    
    X_pid(k+1, :) = (X_k + (dt / 2.0) * (dX1 + dX2))';
end
f_pid(Nt, :) = f_pid(Nt - 1, :);
[~, ~, P_lk_end, P_tot_end] = eval_planta(X_pid(Nt, :)', Nt, f_pid(Nt, :), N_nodes, R_mat, R_r_node, ...
    C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
    alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
P_pid(Nt)   = sum(P_tot_end);
P_l_pid(Nt) = sum(P_lk_end);

res.Space_PID.T_dies_C    = X_pid(:, 1:4) - 273.15;
res.Space_PID.T_rad_C     = X_pid(:, 5) - 273.15;
res.Space_PID.f_exec      = f_pid;
res.Space_PID.P_tot_clust = P_pid;
res.Space_PID.P_leak_tot  = P_l_pid;
res.Space_PID.delta_T_esp = max(res.Space_PID.T_dies_C, [], 2) - min(res.Space_PID.T_dies_C, [], 2);
res.Space_PID.FPS_instant = (f_pid ./ f_max) .* repmat(FPS_nom_dies, Nt, 1);

%% =========================================================================
%% 7. SIMULAÇÃO DO GOVERNADOR 4: GLOBAL POWER CAP ESTÁTICO (P_cap = 65 W)
%% =========================================================================
fprintf('[4/4] Simulando Governador: Global Power Cap Estático (65 W)...\n');
X_cap = zeros(Nt, 5); X_cap(1, :) = X0';
f_cap = zeros(Nt, N_nodes);
P_cap = zeros(Nt, 1);
P_l_cap = zeros(Nt, 1);
P_budget_cluster = 65.0;

for k = 1:(Nt - 1)
    f_cand = f_nominal(k, :);
    V_cand = arrayfun(calc_V, f_cand);
    P_est  = zeros(1, N_nodes);
    for i = 1:N_nodes
        P_est(i) = calc_Pdyn(alpha_workload(k, i), f_cand(i), V_cand(i)) + ...
                   calc_Pleak(X_cap(k, i), V_cand(i));
    end
    P_soma = sum(P_est);
    
    if P_soma > P_budget_cluster
        gamma_scale = max(0.40, min(1.0, P_budget_cluster / P_soma));
        f_k = min(f_max, max(f_min, f_cand * sqrt(gamma_scale)));
    else
        f_k = f_cand;
    end
    f_cap(k, :) = f_k;
    
    X_k = X_cap(k, :)';
    [dX1, ~, P_lk1, P_tot1] = eval_planta(X_k, k, f_k, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    P_cap(k)   = sum(P_tot1);
    P_l_cap(k) = sum(P_lk1);
    
    X_pred = X_k + dt * dX1;
    [dX2, ~, ~, ~] = eval_planta(X_pred, k+1, f_k, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
    
    X_cap(k+1, :) = (X_k + (dt / 2.0) * (dX1 + dX2))';
end
f_cap(Nt, :) = f_cap(Nt - 1, :);
[~, ~, P_lk_end, P_tot_end] = eval_planta(X_cap(Nt, :)', Nt, f_cap(Nt, :), N_nodes, R_mat, R_r_node, ...
    C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
    alpha_workload, calc_V, calc_Pdyn, calc_Pleak);
P_cap(Nt)   = sum(P_tot_end);
P_l_cap(Nt) = sum(P_lk_end);

res.Power_Cap.T_dies_C    = X_cap(:, 1:4) - 273.15;
res.Power_Cap.T_rad_C     = X_cap(:, 5) - 273.15;
res.Power_Cap.f_exec      = f_cap;
res.Power_Cap.P_tot_clust = P_cap;
res.Power_Cap.P_leak_tot  = P_l_cap;
res.Power_Cap.delta_T_esp = max(res.Power_Cap.T_dies_C, [], 2) - min(res.Power_Cap.T_dies_C, [], 2);
res.Power_Cap.FPS_instant = (f_cap ./ f_max) .* repmat(FPS_nom_dies, Nt, 1);

%% =========================================================================
%% 8. CONSOLIDAÇÃO DE MÉTRICAS COMPLETAS (TABELA 1 DO ARTIGO 1)
%% =========================================================================
fprintf('\n====================================================================================================================\n');
fprintf(' RESUMO COMPARATIVO COMPLETO DOS GOVERNADORES BASELINES (TABELA 1 DO ARTIGO):\n');
fprintf('====================================================================================================================\n');
fprintf('%-11s | %-6s | %-6s | %-8s | %-7s | %-7s | %-9s | %-9s | %-9s | %-10s\n', ...
        'Governador', 'T_max', 'DeltaT', 'Viol(%)', 't_viol', 'FPS_med', 'EDP[kJ*s]', 'E_tot[kJ]', 'E_leak[kJ]', 'Freq_med');
fprintf('--------------------------------------------------------------------------------------------------------------------\n');

metricas = struct();
for g = 1:N_gov
    gn = gov_names{g};
    if strcmp(gn, 'Space-PID'), fld = 'Space_PID';
    elseif strcmp(gn, 'Power-Cap'), fld = 'Power_Cap';
    else, fld = gn; end
    
    T_mat = res.(fld).T_dies_C;
    viol_mask = any(T_mat > 85.0, 2);
    
    t_max_val   = max(T_mat(:));
    dt_max_val  = max(res.(fld).delta_T_esp);
    viol_pct    = (sum(viol_mask) / Nt) * 100.0;
    t_viol_s    = sum(viol_mask) * dt;
    f_med_val   = mean(res.(fld).f_exec(:));
    e_tot_kJ    = sum(res.(fld).P_tot_clust) * dt / 1000.0;
    e_leak_kJ   = sum(res.(fld).P_leak_tot) * dt / 1000.0;
    
    % Cálculo rigoroso de Throughput (FPS) e Energy-Delay Product (EDP)
    fps_cluster_inst = sum(res.(fld).FPS_instant, 2);
    fps_medio_tot    = mean(fps_cluster_inst);
    delay_medio_ms   = 1000.0 / max(1e-3, fps_medio_tot); % Delay médio de inferência por frame [ms]
    edp_kJ_s         = e_tot_kJ * (delay_medio_ms / 1000.0); % Energy-Delay Product [kJ * s]
    
    metricas.(fld).T_min       = min(T_mat(:));
    metricas.(fld).T_mean      = mean(T_mat(:));
    metricas.(fld).T_max       = t_max_val;
    metricas.(fld).viol_perc   = viol_pct;
    metricas.(fld).t_viol_seg  = t_viol_s;
    metricas.(fld).delta_T     = dt_max_val;
    metricas.(fld).f_media     = f_med_val;
    metricas.(fld).FPS_medio   = fps_medio_tot;
    metricas.(fld).Delay_medio = delay_medio_ms;
    metricas.(fld).EDP         = edp_kJ_s;
    metricas.(fld).energia_kJ  = e_tot_kJ;
    metricas.(fld).e_leak_kJ   = e_leak_kJ;
    
    fprintf('%-11s | %5.1f°C | %5.2fK | %6.1f%% | %6.0fs | %6.1f | %8.2f  | %8.1fkJ | %8.1fkJ | %7.2f GHz\n', ...
            gn, t_max_val, dt_max_val, viol_pct, t_viol_s, fps_medio_tot, ...
            edp_kJ_s, e_tot_kJ, e_leak_kJ, f_med_val);
end
fprintf('====================================================================================================================\n');

%% 9. EXPORTAÇÃO DOS DADOS DA ETAPA 3
mat_etapa3 = fullfile(script_dir, 'etapa3_dados_baselines.mat');
fprintf('-> Exportando dados para: %s\n', mat_etapa3);

save('-v7', mat_etapa3, 't', 'dt', 'Nt', 'res', 'metricas', 'gov_names', 'PID_params', 'FPS_nom_dies');
fprintf(' [OK] Arquivo "etapa3_dados_baselines.mat" salvo com sucesso!\n');

%% 10. VISUALIZAÇÃO GRÁFICA COMPARATIVA
try
    h_fig = figure('Name', 'Etapa 3: Comparativo dos 4 Governadores Baselines', ...
                   'Position', [100, 100, 1100, 850], 'Visible', 'off');

    subplot(3, 1, 1);
    plot(t/60, max(res.Ondemand.T_dies_C, [], 2),  'r-',  'LineWidth', 1.2); hold on;
    plot(t/60, max(res.Throttling.T_dies_C, [], 2),'m--', 'LineWidth', 1.2);
    plot(t/60, max(res.Space_PID.T_dies_C, [], 2), 'b-.', 'LineWidth', 1.3);
    plot(t/60, max(res.Power_Cap.T_dies_C, [], 2), 'g-',  'LineWidth', 1.2);
    yline(85, 'k--', 'LineWidth', 1.2);
    grid on;
    ylabel('T_{max} [°C]');
    legend('Linux Ondemand', 'Thermal Throttling', 'Space-PID', 'Global Power Cap', 'Location', 'SouthWest');
    title('Temperatura Máxima da Junção dos Processadores (T_{limite} = 85 °C)');

    subplot(3, 1, 2);
    plot(t/60, res.Ondemand.delta_T_esp,   'r-',  'LineWidth', 1.1); hold on;
    plot(t/60, res.Throttling.delta_T_esp, 'm--', 'LineWidth', 1.1);
    plot(t/60, res.Space_PID.delta_T_esp,  'b-.', 'LineWidth', 1.3);
    plot(t/60, res.Power_Cap.delta_T_esp,  'g-',  'LineWidth', 1.1);
    yline(15, 'k--', 'LineWidth', 1.2);
    grid on;
    ylabel('\DeltaT Espacial [°C]');
    legend('Ondemand', 'Throttling', 'Space-PID', 'Power Cap', 'Location', 'NorthEast');
    title('Gradiente Térmico Espacial Intra-Placa (\DeltaT \le 15 °C)');

    subplot(3, 1, 3);
    plot(t/60, sum(res.Ondemand.FPS_instant, 2),   'r-',  'LineWidth', 1.1); hold on;
    plot(t/60, sum(res.Throttling.FPS_instant, 2), 'm--', 'LineWidth', 1.1);
    plot(t/60, sum(res.Space_PID.FPS_instant, 2),  'b-.', 'LineWidth', 1.3);
    plot(t/60, sum(res.Power_Cap.FPS_instant, 2),  'g-',  'LineWidth', 1.1);
    grid on;
    xlabel('Tempo Orbital [min]');
    ylabel('Throughput Agregado [FPS]');
    legend('Ondemand', 'Throttling', 'Space-PID', 'Power Cap', 'Location', 'SouthWest');
    title('Throughput Efetivo de Inferência de IA do Cluster');

    fig_png = fullfile(script_dir, 'Diagnostico da etapa 3.png');
    saveas(h_fig, fig_png);
    fprintf(' [OK] Gráfico salvo em: %s\n', fig_png);
    close(h_fig);
catch
end

end % Fim da função principal

%% =========================================================================
% FUNÇÃO AUXILIAR: DINÂMICA DA PLANTA COM FLUXOS SEPARADOS
% =========================================================================
function [dX, P_dyn, P_leak, P_tot] = eval_planta(X, k_idx, f_vec, N_nodes, R_mat, R_r, ...
                                                   C_dies, C_rad, A_rad, epsilon, sigma, ...
                                                   T_space, Q_solar, Q_albedo, Q_IR, ...
                                                   alpha_w, calc_V, calc_Pdyn, calc_Pleak)
    dX     = zeros(5, 1);
    P_dyn  = zeros(N_nodes, 1);
    P_leak = zeros(N_nodes, 1);
    P_tot  = zeros(N_nodes, 1);
    
    T_nodes = X(1:4);
    T_rad   = X(5);
    Q_cond_para_rad = 0.0;
    
    for i = 1:N_nodes
        f_i = f_vec(i);
        V_i = calc_V(f_i);
        P_dyn(i)  = calc_Pdyn(alpha_w(k_idx, i), f_i, V_i);
        P_leak(i) = calc_Pleak(T_nodes(i), V_i);
        P_tot(i)  = P_dyn(i) + P_leak(i);
        
        Q_crosstalk = 0.0;
        for j = 1:N_nodes
            if i ~= j
                Q_crosstalk = Q_crosstalk + (T_nodes(j) - T_nodes(i)) / R_mat(i, j);
            end
        end
        
        Q_para_rad = (T_nodes(i) - T_rad) / R_r;
        Q_cond_para_rad = Q_cond_para_rad + Q_para_rad;
        
        dX(i) = (P_tot(i) + Q_crosstalk - Q_para_rad) / C_dies(i);
    end
    
    % Radiador com fluxos térmicos orbitais separados
    Q_rad_emitido  = epsilon * sigma * A_rad * (T_rad^4 - T_space^4);
    Q_rad_injetado = Q_solar(k_idx) + Q_albedo(k_idx) + Q_IR(k_idx);
    
    dX(5) = (Q_cond_para_rad + Q_rad_injetado - Q_rad_emitido) / C_rad;
end
