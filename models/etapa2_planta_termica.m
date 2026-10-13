% =========================================================================
% PROJETO DE MESTRADO: CONTROLE ESTATÍSTICO/PREDITIVO DE CLUSTERS DE IA
% ETAPA 2 (BLINDADA): Rede Térmica Compacta CTM (Espaço de Estados 5x5) e
%                     Modelo Eletrotérmico FinFET (NVIDIA Jetson Orin)
%
% Compatibilidade: GNU Octave >= 7.0 e MATLAB >= R2020b
% =========================================================================
function etapa2_planta_termica()

close all; clc;

fprintf('===================================================================\n');
fprintf(' ETAPA 2 (BLINDADA): Rede CTM (5x5) e NVIDIA Jetson Orin\n');
fprintf('===================================================================\n');

%% 1. CARREGAMENTO DOS DADOS DA ETAPA 1
script_dir = '.';
try
    [s_dir, ~, ~] = fileparts(mfilename('fullpath'));
    if ~isempty(s_dir)
        script_dir = s_dir;
    end
catch
    script_dir = '.';
end

mat_etapa1 = fullfile(script_dir, 'etapa1_dados_orbitais.mat');
if ~exist(mat_etapa1, 'file')
    if exist('etapa1_dados_orbitais.mat', 'file')
        mat_etapa1 = 'etapa1_dados_orbitais.mat';
    else
        error('Arquivo "%s" não encontrado. Execute a Etapa 1 primeiro.', mat_etapa1);
    end
end

fprintf('-> Carregando cargas orbitais da Etapa 1: %s\n', mat_etapa1);
dados_e1 = load(mat_etapa1);

t       = dados_e1.t;
dt      = dados_e1.dt;
Nt      = length(t);

% Extração rigorosa dos três fluxos orbitais separados (ECSS/NASA)
if isfield(dados_e1, 'Q_solar'),  Q_solar  = dados_e1.Q_solar;  else, Q_solar  = dados_e1.q_solar; end
if isfield(dados_e1, 'Q_albedo'), Q_albedo = dados_e1.Q_albedo; else, Q_albedo = dados_e1.q_albedo; end
if isfield(dados_e1, 'Q_IR'),     Q_IR     = dados_e1.Q_IR;     else, Q_IR     = dados_e1.q_IR; end

A_rad   = dados_e1.A_rad;
epsilon = dados_e1.epsilon;
C_rad   = dados_e1.C_rad;
T_space = dados_e1.T_space;
sigma   = dados_e1.sigma;
T_rad   = dados_e1.T_rad;

%% 2. PARÂMETROS DA REDE TÉRMICA COMPACTA (CTM 5x5)
N_nodes = 4;
R12 = 1.45; R34 = 1.45;
R13 = 1.52; R24 = 1.52;
R14 = 3.20; R23 = 3.20;
R_r_node = 2.10;

R_mat = [ Inf,  R12,  R13,  R14 ;
          R12,  Inf,  R23,  R24 ;
          R13,  R23,  Inf,  R34 ;
          R14,  R24,  R34,  Inf ];

% Capacitância térmica por nó [J/K] (Die Si 2.5g + IHS Cu 20g -> Tabela 5 do Artigo 1)
c_node = 18.5; 
C_dies = c_node * ones(N_nodes, 1);

%% 3. MODELO ELETROTÉRMICO DOS PROCESSADORES JETSON ORIN (FinFET)
V_min = 0.65; V_max = 1.05;
f_min = 0.30; f_max = 1.30;
beta_delay = (V_max - V_min) / (f_max - f_min);
C_eff = 24.5e-9;

T0_ref = 300.0; V0_ref = 0.85; P_leak_ref = 0.80;
k_B_eV = 8.617333262e-5; E_a = 0.20;

calc_V = @(f_val) max(V_min, min(V_max, V_min + beta_delay * (f_val - f_min)));
calc_Pdyn = @(alpha_val, f_val, V_val) (alpha_val * C_eff * (V_val^2) * (f_val * 1e9));
calc_Pleak = @(T_val, V_val) ...
    (P_leak_ref * ((V_val / V0_ref)^1.5) * ((max(200.0, min(500.0, T_val)) / T0_ref)^2) * ...
     exp((E_a / k_B_eV) * (1.0 / T0_ref - 1.0 / max(200.0, min(500.0, T_val)))));

%% 4. CENÁRIO DE CARGA DINÂMICA
alpha_workload = zeros(Nt, N_nodes);
f_nominal      = zeros(Nt, N_nodes);

for k = 1:Nt
    t_sec = t(k);
    alpha_workload(k, 1) = 0.85 + 0.10 * sin(2 * pi * t_sec / 1200); f_nominal(k, 1) = 1.15;
    alpha_workload(k, 2) = 0.70 + 0.20 * min(1.0, t_sec / (t(end) * 0.7)); f_nominal(k, 2) = 1.05;
    if mod(t_sec, 1800) > 900
        alpha_workload(k, 3) = 0.95; f_nominal(k, 3) = 1.30;
    else
        alpha_workload(k, 3) = 0.60; f_nominal(k, 3) = 0.80;
    end
    alpha_workload(k, 4) = 0.75; f_nominal(k, 4) = 1.00;
end

%% 5. SIMULAÇÃO TÉRMICA DINÂMICA (HEUN / RK2 COM FLUXOS SEPARADOS)
fprintf('-> Integrando sistema CTM 5x5 ao longo de %d passos (dt = %.2fs)...\n', Nt, dt);

X_states     = zeros(Nt, 5);
P_dyn_hist   = zeros(Nt, N_nodes);
P_leak_hist  = zeros(Nt, N_nodes);
P_total_hist = zeros(Nt, N_nodes);

X_states(1, 1:4) = 310.0;
X_states(1, 5)   = T_rad(1);

for k = 1:(Nt - 1)
    X_k = X_states(k, :)';
    [f_k, P_dyn_k, P_leak_k, P_tot_k] = calc_derivatives_blindada(X_k, k, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, f_nominal, calc_V, calc_Pdyn, calc_Pleak);
    X_pred = X_k + dt * f_k;
    
    [f_pred, ~, ~, ~] = calc_derivatives_blindada(X_pred, k + 1, N_nodes, R_mat, R_r_node, ...
        C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
        alpha_workload, f_nominal, calc_V, calc_Pdyn, calc_Pleak);
    
    X_next = X_k + (dt / 2.0) * (f_k + f_pred);
    X_states(k + 1, :) = X_next';
    P_dyn_hist(k, :)   = P_dyn_k';
    P_leak_hist(k, :)  = P_leak_k';
    P_total_hist(k, :) = P_tot_k';
end

% Passo final de potências
[~, P_dyn_hist(Nt, :), P_leak_hist(Nt, :), P_total_hist(Nt, :)] = ...
    calc_derivatives_blindada(X_states(Nt, :)', Nt, N_nodes, R_mat, R_r_node, ...
    C_dies, C_rad, A_rad, epsilon, sigma, T_space, Q_solar, Q_albedo, Q_IR, ...
    alpha_workload, f_nominal, calc_V, calc_Pdyn, calc_Pleak);

%% 6. MÉTRICAS ESTATÍSTICAS CONSOLIDADAS
T_celsius   = X_states - 273.15;
T_dies_C    = T_celsius(:, 1:4);
T_rad_C     = T_celsius(:, 5);
delta_T_esp = max(T_dies_C, [], 2) - min(T_dies_C, [], 2);

metricas_planta = struct();
metricas_planta.T_min       = min(T_dies_C(:));
metricas_planta.T_mean      = mean(T_dies_C(:));
metricas_planta.T_max       = max(T_dies_C(:));
metricas_planta.T_rad_min   = min(T_rad_C);
metricas_planta.T_rad_max   = max(T_rad_C);
metricas_planta.delta_T_max = max(delta_T_esp);
metricas_planta.P_med_tot   = mean(sum(P_total_hist, 2));
metricas_planta.P_pico_tot  = max(sum(P_total_hist, 2));
metricas_planta.P_leak_med  = mean(sum(P_leak_hist, 2));

fprintf('=== RESULTADOS CONSOLIDADOS DA PLANTA TÉRMICA ===\n');
for i = 1:N_nodes
    fprintf(' Nó %d: Min=%6.2f °C | Méd=%6.2f °C | Máx=%6.2f °C\n', ...
            i, min(T_dies_C(:, i)), mean(T_dies_C(:, i)), max(T_dies_C(:, i)));
end
fprintf(' Radiador: Min=%6.2f °C | Méd=%6.2f °C | Máx=%6.2f °C\n', ...
        min(T_rad_C), mean(T_rad_C), max(T_rad_C));
fprintf(' Gradiente Máximo Intra-Placa: %.2f °C\n', metricas_planta.delta_T_max);
fprintf(' Potência Média Cluster: %.2f W | Pico: %.2f W\n', ...
        metricas_planta.P_med_tot, metricas_planta.P_pico_tot);

%% 7. EXPORTAÇÃO BLINDADA
mat_etapa2 = fullfile(script_dir, 'etapa2_dados_planta.mat');
fprintf('-> Exportando dados da planta para: %s\n', mat_etapa2);

save('-v7', mat_etapa2, ...
     't', 'dt', 'Nt', 'N_nodes', 'X_states', 'T_dies_C', 'T_rad_C', 'delta_T_esp', ...
     'P_dyn_hist', 'P_leak_hist', 'P_total_hist', 'alpha_workload', 'f_nominal', ...
     'R_mat', 'R_r_node', 'C_dies', 'C_rad', ...
     'V_min', 'V_max', 'f_min', 'f_max', 'beta_delay', 'C_eff', 'P_leak_ref', 'E_a', ...
     'Q_solar', 'Q_albedo', 'Q_IR', ...
     'T0_ref', 'V0_ref', 'sigma', 'metricas_planta');

fprintf(' [OK] Arquivo "etapa2_dados_planta.mat" salvo com sucesso!\n');

%% 8. VISUALIZAÇÃO GRÁFICA
try
    h_fig = figure('Name', 'Etapa 2: Planta Térmica 5x5 e Dissipação dos Semicondutores', ...
                   'Position', [100, 100, 1000, 750], 'Visible', 'off');

    subplot(3, 1, 1);
    plot(t/60, T_dies_C(:, 1), 'r-', 'LineWidth', 1.2); hold on;
    plot(t/60, T_dies_C(:, 2), 'b-', 'LineWidth', 1.2);
    plot(t/60, T_dies_C(:, 3), 'm-', 'LineWidth', 1.2);
    plot(t/60, T_dies_C(:, 4), 'g-', 'LineWidth', 1.2);
    plot(t/60, T_rad_C,        'k--', 'LineWidth', 1.5);
    yline(85, 'r--', 'LineWidth', 1.2);
    grid on;
    ylabel('Temperatura [°C]');
    legend('Nó 1', 'Nó 2', 'Nó 3 (Rajadas)', 'Nó 4', 'Radiador', 'Location', 'SouthWest');
    title('Dinâmica Térmica da Planta CTM 5x5 em Malha Aberta');

    subplot(3, 1, 2);
    plot(t/60, delta_T_esp, 'k-', 'LineWidth', 1.2); hold on;
    yline(15, 'r--', 'LineWidth', 1.2);
    grid on;
    ylabel('\DeltaT Espacial [°C]');
    legend('\DeltaT_{max}', 'Location', 'NorthEast');
    title('Gradiente Térmico Espacial Intra-Placa (\DeltaT \le 15 °C)');

    subplot(3, 1, 3);
    plot(t/60, sum(P_total_hist, 2), 'k-', 'LineWidth', 1.3); hold on;
    plot(t/60, sum(P_leak_hist, 2),  'r:', 'LineWidth', 1.2);
    grid on;
    xlabel('Tempo Orbital [min]');
    ylabel('Potência [W]');
    legend('P_{total}', 'P_{leak}', 'Location', 'NorthEast');
    title('Dissipação de Potência Total e Fuga Estática');

    fig_png = fullfile(script_dir, 'Diagnóstico da etapa 2.png');
    saveas(h_fig, fig_png);
    fprintf(' [OK] Gráfico salvo em: %s\n', fig_png);
    close(h_fig);
catch
end

end % Fim da função principal etapa2_planta_termica

%% =========================================================================
% FUNÇÃO AUXILIAR: DERIVADAS DE ESTADO COM BALANÇO RADIATIVO SEPARADO
% =========================================================================
function [dX, P_dyn, P_leak, P_tot] = calc_derivatives_blindada(X, k_idx, N_nodes, R_mat, R_r, ...
                                                                C_dies, C_rad, A_rad, epsilon, sigma, ...
                                                                T_space, Q_solar, Q_albedo, Q_IR, ...
                                                                alpha_w, f_nom, calc_V, calc_Pdyn, calc_Pleak)
    dX     = zeros(5, 1);
    P_dyn  = zeros(N_nodes, 1);
    P_leak = zeros(N_nodes, 1);
    P_tot  = zeros(N_nodes, 1);
    
    T_nodes = X(1:4); 
    T_rad   = X(5);
    Q_cond_para_rad = 0.0;
    
    for i = 1:N_nodes
        f_i = f_nom(k_idx, i);
        V_i = calc_V(f_i);
        P_dyn(i)  = calc_Pdyn(alpha_w(k_idx, i), f_i, V_i);
        P_leak(i) = calc_Pleak(T_nodes(i), V_i);
        P_tot(i)  = P_dyn(i) + P_leak(i);
        
        % Crosstalk térmico condutivo entre chips vizinhos (Kirchhoff)
        Q_crosstalk = 0.0;
        for j = 1:N_nodes
            if i ~= j
                Q_crosstalk = Q_crosstalk + (T_nodes(j) - T_nodes(i)) / R_mat(i, j);
            end
        end
        
        % Condução individual para o radiador
        Q_para_rad = (T_nodes(i) - T_rad) / R_r;
        Q_cond_para_rad = Q_cond_para_rad + Q_para_rad;
        
        % dT_i / dt
        dX(i) = (P_tot(i) + Q_crosstalk - Q_para_rad) / C_dies(i);
    end
    
    % Balanço de energia do radiador com fluxos orbitais separados (ECSS/NASA)
    Q_rad_emitido  = epsilon * sigma * A_rad * (T_rad^4 - T_space^4);
    Q_rad_injetado = Q_solar(k_idx) + Q_albedo(k_idx) + Q_IR(k_idx);
    
    % dT_rad / dt
    dX(5) = (Q_cond_para_rad + Q_rad_injetado - Q_rad_emitido) / C_rad;
end
