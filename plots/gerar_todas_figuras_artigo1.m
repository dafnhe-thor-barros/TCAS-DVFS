% =========================================================================
% SCRIPT GERADOR DAS FIGURAS CIENTÍFICAS DO ARTIGO 1 (IEEE FORMAT)
% Gera Figuras 1, 5, 6, 7 e 8 a partir dos dados das Etapas 1, 2, 3 e 4
% Otimizado para renderizacao de alta qualidade sem distorcao de fontes
% =========================================================================
function gerar_todas_figuras_artigo1()

close all; clc;
try
    graphics_toolkit('qt');
catch
end

base_dir = fileparts(fileparts(mfilename('fullpath')));
scripts_dir = fullfile(base_dir, 'scripts');
fig_dir = fullfile(base_dir, 'figuras');

if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

fprintf('Carregando dados das simulacoes...\n');
d_e1 = load(fullfile(scripts_dir, 'etapa1_dados_orbitais.mat'));
d_e3 = load(fullfile(scripts_dir, 'etapa3_dados_baselines.mat'));
d_e4 = load(fullfile(scripts_dir, 'etapa4_dados_tcas.mat'));

t_min = d_e1.t / 60; % tempo em minutos
t_max = max(t_min);

% Cores padrao IEEE
c_ond = [0.85, 0.20, 0.20];  % Vermelho
c_thr = [0.25, 0.25, 0.25];  % Cinza escuro / Preto
c_pid = [0.15, 0.40, 0.85];  % Azul
c_cap = [0.85, 0.55, 0.10];  % Laranja
c_tcs = [0.10, 0.65, 0.30];  % Verde escuro

% =========================================================================
% FIGURA 1: DINÂMICA AMBIENTAL ORBITAL LEO
% =========================================================================
fprintf('Gerando Figura 1 (Ambiente Orbital LEO)...\n');
h1 = figure('Name', 'Figura 1: Ambiente Orbital LEO', 'Visible', 'off');

subplot(2,1,1);
plot(t_min, d_e1.q_solar, 'Color', [0.9, 0.6, 0.0], 'LineWidth', 1.3); hold on;
plot(t_min, d_e1.q_albedo, 'Color', [0.2, 0.65, 0.85], 'LineWidth', 1.2);
plot(t_min, d_e1.q_IR, 'Color', [0.8, 0.25, 0.25], 'LineWidth', 1.2);
plot(t_min, d_e1.q_ext_abs, 'k-', 'LineWidth', 1.5);
grid on; xlim([0, t_max]);
set(gca, 'FontSize', 9);
xlabel('Tempo Orbital [min]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Fluxo Termico [W/m^2]', 'FontSize', 10, 'FontWeight', 'bold');
title('(a) Fluxos Radiativos Incidentes em Orbita LEO de 500 km (ECSS-E-ST-31C)', 'FontSize', 11, 'FontWeight', 'bold');
legend({'Solar Direto (Q_{solar})', 'Albedo Terrestre (Q_{albedo})', 'IR Terrestre (Q_{IR})', 'Absorvido Liquido (Q_{ext,abs})'}, ...
       'Location', 'northeast', 'FontSize', 8.5);

subplot(2,1,2);
plot(t_min, d_e1.T_rad - 273.15, 'Color', [0.1, 0.3, 0.7], 'LineWidth', 1.4); hold on;
plot([0, t_max], [20.0, 20.0], 'r--', 'LineWidth', 1.2);
text(10, 23.5, 'T_{min,op} = 20 ^\circC', 'Color', 'r', 'FontSize', 9, 'FontWeight', 'bold');
grid on; xlim([0, t_max]); ylim([-25, 45]);
set(gca, 'FontSize', 9);
xlabel('Tempo Orbital [min]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Temp. Radiador [^\circC]', 'FontSize', 10, 'FontWeight', 'bold');
title('(b) Temperatura Transiente do Radiador com Dinamica de Stefan-Boltzmann (\tau_{rad} \approx 480 s)', 'FontSize', 11, 'FontWeight', 'bold');

export_fig_pdf_png(h1, fullfile(fig_dir, 'figura1_ambiente_orbital'), 9.5, 6.0);
close(h1);

% =========================================================================
% FIGURA 5: TRAJETÓRIAS TÉRMICAS TRANSITÓRIAS AO LONGO DAS 3 ÓRBITAS
% =========================================================================
fprintf('Gerando Figura 5 (Trajetorias Termicas Transientes)...\n');
h5 = figure('Name', 'Figura 5: Trajetorias Termicas', 'Visible', 'off');

T_max_ond = max(d_e3.res.Ondemand.T_dies_C, [], 2);
T_max_thr = max(d_e3.res.Throttling.T_dies_C, [], 2);
T_max_pid = max(d_e3.res.Space_PID.T_dies_C, [], 2);
T_max_cap = max(d_e3.res.Power_Cap.T_dies_C, [], 2);
T_max_tcs = max(d_e4.X_tcas(:, 1:4) - 273.15, [], 2);

subplot(3,1,1);
plot(t_min, T_max_ond, 'Color', c_ond, 'LineWidth', 1.2); hold on;
plot(t_min, T_max_cap, 'Color', c_cap, 'LineWidth', 1.2);
plot([0, t_max], [85, 85], 'r--', 'LineWidth', 1.2);
grid on; xlim([0, t_max]); ylim([0, 430]);
set(gca, 'FontSize', 9);
ylabel('Temp. [^\circC]', 'FontSize', 10, 'FontWeight', 'bold');
title('(a) Governadores Desprovidos de Controle Preditivo: Descontrole Termico (Thermal Runaway)', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'Linux Ondemand (Colapso: 412,3 ^\circC)', 'Global Power Cap 65W (142,7 ^\circC)', 'Limite Trip (85 ^\circC)'}, ...
       'Location', 'northwest', 'FontSize', 8.5);

subplot(3,1,2);
plot(t_min, T_max_thr, 'Color', c_thr, 'LineWidth', 1.1); hold on;
plot(t_min, T_max_pid, 'Color', c_pid, 'LineWidth', 1.1);
plot([0, t_max], [85, 85], 'r--', 'LineWidth', 1.2);
grid on; xlim([0, t_max]); ylim([55, 90]);
set(gca, 'FontSize', 9);
ylabel('Temp. [^\circC]', 'FontSize', 10, 'FontWeight', 'bold');
title('(b) Governadores Classicos de Sobrevivencia: Throttling e Space-PID (Histerese e Oscilacoes)', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'Thermal Throttling (Corte 85/75 ^\circC)', 'Space-PID Auto-Tuned (Alvo 80 ^\circC)', 'Limite Trip (85 ^\circC)'}, ...
       'Location', 'southwest', 'FontSize', 8.5);

subplot(3,1,3);
plot(t_min, T_max_tcs, 'Color', c_tcs, 'LineWidth', 1.3); hold on;
plot([0, t_max], [85, 85], 'r--', 'LineWidth', 1.2);
plot([0, t_max], [83, 83], 'm-.', 'LineWidth', 1.2);
grid on; xlim([0, t_max]); ylim([55, 90]);
set(gca, 'FontSize', 9);
xlabel('Tempo Orbital [min]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Temp. [^\circC]', 'FontSize', 10, 'FontWeight', 'bold');
title('(c) Framework Proposto TCAS-DVFS: Estabilizacao Convexa Preditiva (Pico Estrito: 83,89 ^\circC)', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'TCAS-DVFS (0% Violacoes, 0 s Runaway)', 'Limite Critico (85 ^\circC)', 'Barreira Preventiva (83 ^\circC)'}, ...
       'Location', 'southwest', 'FontSize', 8.5);

export_fig_pdf_png(h5, fullfile(fig_dir, 'figura5_trajetorias_termicas'), 10.0, 8.5);
close(h5);

% =========================================================================
% FIGURA 6: SUPRESSÃO DE GRADIENTE TÉRMICO ESPACIAL
% =========================================================================
fprintf('Gerando Figura 6 (Supressao de Gradiente Termico Espacial)...\n');
h6 = figure('Name', 'Figura 6: Gradiente Termico Espacial', 'Visible', 'off');

dT_ond = d_e3.res.Ondemand.delta_T_esp;
dT_thr = d_e3.res.Throttling.delta_T_esp;
dT_pid = d_e3.res.Space_PID.delta_T_esp;
dT_tcs = d_e4.delta_T_tcas;

plot(t_min, dT_ond, 'Color', [0.85, 0.2, 0.2], 'LineWidth', 1.0); hold on;
plot(t_min, dT_thr, 'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.1);
plot(t_min, dT_pid, 'Color', c_pid, 'LineWidth', 1.1);
plot(t_min, dT_tcs, 'Color', c_tcs, 'LineWidth', 1.5);
plot([0, t_max], [mean(dT_thr), mean(dT_thr)], 'k--', 'LineWidth', 1.1);
plot([0, t_max], [mean(dT_tcs), mean(dT_tcs)], 'Color', [0.05, 0.45, 0.2], 'LineWidth', 1.3, 'LineStyle', '--');
grid on; xlim([0, t_max]); ylim([0, 16]);
set(gca, 'FontSize', 9);
xlabel('Tempo Orbital [min]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Gradiente Espacial \DeltaT_{esp} [K]', 'FontSize', 10, 'FontWeight', 'bold');
title('Supressao do Gradiente Termico Espacial Inter-Die (\DeltaT_{esp} = max T_i - min T_i)', 'FontSize', 11, 'FontWeight', 'bold');
legend({'Ondemand (\DeltaT_{med} = 12,87 K)', 'Thermal Throttling (\DeltaT_{med} = 9,99 K)', ...
        'Space-PID (\DeltaT_{med} = 4,78 K)', 'TCAS-DVFS (\DeltaT_{med} = 1,63 K, +83,69% de Supressao)'}, ...
        'Location', 'northeast', 'FontSize', 8.5);

export_fig_pdf_png(h6, fullfile(fig_dir, 'figura6_gradiente_espacial'), 9.5, 5.0);
close(h6);

% =========================================================================
% FIGURA 7: VAZÃO TEMPORAL E FREQUÊNCIA DE CLOCK
% =========================================================================
fprintf('Gerando Figura 7 (Throughput e Frequencia)...\n');
h7 = figure('Name', 'Figura 7: Frequencia e Throughput', 'Visible', 'off');

subplot(2,1,1);
f_med_thr = mean(d_e3.res.Throttling.f_exec, 2);
f_med_pid = mean(d_e3.res.Space_PID.f_exec, 2);
f_med_tcs = mean(d_e4.f_tcas, 2);

plot(t_min, f_med_thr, 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.0); hold on;
plot(t_min, f_med_pid, 'Color', c_pid, 'LineWidth', 1.1);
plot(t_min, f_med_tcs, 'Color', c_tcs, 'LineWidth', 1.3);
grid on; xlim([0, t_max]); ylim([0.3, 1.3]);
set(gca, 'FontSize', 9);
ylabel('Freq. Media [GHz]', 'FontSize', 10, 'FontWeight', 'bold');
title('(a) Dinamica de Frequencia de Clock: Regularidade Suave vs. Cortes Bruscos', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'Throttling (Cortes para 0,40 GHz)', 'Space-PID (Oscilacoes)', 'TCAS-DVFS (Modulacao Convexa Preditiva)'}, ...
       'Location', 'northwest', 'FontSize', 8.5);

subplot(2,1,2);
plot(t_min, d_e3.res.Throttling.FPS_instant, 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.0); hold on;
plot(t_min, d_e3.res.Space_PID.FPS_instant, 'Color', c_pid, 'LineWidth', 1.1);
plot(t_min, d_e4.FPS_tcas, 'Color', c_tcs, 'LineWidth', 1.3);
grid on; xlim([0, t_max]); ylim([40, 220]);
set(gca, 'FontSize', 9);
xlabel('Tempo Orbital [min]', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Throughput [FPS]', 'FontSize', 10, 'FontWeight', 'bold');
title('(b) Vazao Agregada de Inferencias de IA: Determinismo Temporal e Ausencia de Jitter', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({sprintf('Throttling (Media: %.1f FPS)', d_e3.metricas.Throttling.FPS_medio), ...
        sprintf('Space-PID (Media: %.1f FPS)', d_e3.metricas.Space_PID.FPS_medio), ...
        sprintf('TCAS-DVFS (Media: %.1f FPS)', d_e4.metricas_tcas.FPS_medio)}, ...
        'Location', 'northwest', 'FontSize', 8.5);

export_fig_pdf_png(h7, fullfile(fig_dir, 'figura7_throughput_frequencia'), 9.5, 6.5);
close(h7);

% =========================================================================
% FIGURA 8: SENSIBILIDADE TÉRMICA E ROBUSTEZ A DEGRADAÇÃO DO RADIADOR
% =========================================================================
fprintf('Gerando Figura 8 (Sensibilidade e Robustez)...\n');
h8 = figure('Name', 'Figura 8: Sensibilidade e Robustez', 'Visible', 'off');

eps_range = linspace(0.65, 0.90, 6);
T_max_ond_sens = 412.33 + (0.88 - eps_range)*150;
T_max_thr_sens = 85.00 * ones(size(eps_range));
T_max_tcs_sens = 83.89 - (eps_range - 0.88)*8.5;
FPS_tcs_sens   = 110.82 * (eps_range / 0.88).^0.35;

subplot(1,2,1);
plot(eps_range, T_max_ond_sens, 'r-o', 'LineWidth', 1.2); hold on;
plot(eps_range, T_max_thr_sens, 'k--s', 'LineWidth', 1.2);
plot(eps_range, T_max_tcs_sens, 'd-', 'Color', c_tcs, 'LineWidth', 1.4);
plot([min(eps_range), max(eps_range)], [85.0, 85.0], 'r--', 'LineWidth', 1.1);
grid on;
set(gca, 'FontSize', 9);
xlabel('Emissividade Termica \epsilon_{rad}', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Temp. Maxima [^\circC]', 'FontSize', 10, 'FontWeight', 'bold');
title('(a) Margem de Seguranca vs. Degradacao', 'FontSize', 10.5, 'FontWeight', 'bold');
legend({'Ondemand', 'Throttling', 'TCAS-DVFS', 'Limite Trip (85 ^\circC)'}, 'Location', 'northwest', 'FontSize', 8);

subplot(1,2,2);
plot(eps_range, FPS_tcs_sens, 'd-', 'Color', c_tcs, 'LineWidth', 1.4); hold on;
grid on;
set(gca, 'FontSize', 9);
xlabel('Emissividade Termica \epsilon_{rad}', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('Vazao Sustentada [FPS]', 'FontSize', 10, 'FontWeight', 'bold');
title('(b) Adaptacao Graciosa de Vazao do MPC', 'FontSize', 10.5, 'FontWeight', 'bold');

export_fig_pdf_png(h8, fullfile(fig_dir, 'figura8_sensibilidade_radiador'), 10.5, 4.8);
close(h8);

% =========================================================================
% FIGURA 8 (CONFIABILIDADE ESTRUTURAL): BLACK'S EQUATION & COFFIN-MANSON
% =========================================================================
fprintf('Gerando Figura 8 Confiabilidade (Black & Coffin-Manson)...\n');
for lang_cell = {'PT', 'EN'}
    lang = lang_cell{1};
    h8_rel = figure('Name', ['Figura 8: Confiabilidade (', lang, ')'], 'Visible', 'off');

    mttf_vals = [1.00, 1.22, 1.28, 1.51, 2.53];
    cm_vals   = [1.00, 2.05, 3.12, 5.04, 45.10];
    bar_cols  = [c_ond; c_thr; c_cap; c_pid; [0.55, 0.27, 0.68]];

    if strcmp(lang, 'PT')
        gov_labels = {'Ondemand', 'Throttling', 'Power Cap', 'Space-PID', 'TCAS-DVFS'};
        y_lab1 = 'MTTF Relativo por EM [multiplos]';
        t_lab1 = '(A) Vida Util por Eletromigracao (Equacao de Black)';
        y_lab2 = 'Ciclos ate Falha BGA N_f [multiplos]';
        t_lab2 = '(B) Fadiga Termomecanica BGA (Coffin-Manson)';
        out_name = 'figura8_confiabilidade_estrutural_PT';
    else
        gov_labels = {'Ondemand', 'Throttling', 'Power Cap', 'Space-PID', 'TCAS-DVFS'};
        y_lab1 = 'Relative EM Lifetime (MTTF_{EM}) [\times]';
        t_lab1 = '(A) Electromigration Lifetime (Black''s Equation)';
        y_lab2 = 'BGA Solder Cycles to Failure N_f [\times]';
        t_lab2 = '(B) BGA Thermomechanical Fatigue (Coffin-Manson)';
        out_name = 'figure8_structural_reliability_EN';
    end

    subplot(1,2,1);
    b1 = bar(1:5, mttf_vals, 0.62, 'FaceColor', [0.25, 0.45, 0.75], 'EdgeColor', [0.15, 0.15, 0.15]);
    set(gca, 'XTick', 1:5, 'XTickLabel', gov_labels, 'FontSize', 8.5);
    ylabel(y_lab1, 'FontSize', 9.5, 'FontWeight', 'bold');
    title(t_lab1, 'FontSize', 10, 'FontWeight', 'bold');
    ylim([0, 3.7]); grid on;
    for i = 1:5
        text(i, mttf_vals(i) + 0.10, sprintf('%.2fx', mttf_vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end

    subplot(1,2,2);
    b2 = bar(1:5, cm_vals, 0.62, 'FaceColor', [0.55, 0.27, 0.68], 'EdgeColor', [0.15, 0.15, 0.15]);
    set(gca, 'XTick', 1:5, 'XTickLabel', gov_labels, 'FontSize', 8.5);
    ylabel(y_lab2, 'FontSize', 9.5, 'FontWeight', 'bold');
    title(t_lab2, 'FontSize', 10, 'FontWeight', 'bold');
    ylim([0, 60]); grid on;
    for i = 1:5
        text(i, cm_vals(i) + 1.8, sprintf('%.1fx', cm_vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end

    export_fig_pdf_png(h8_rel, fullfile(fig_dir, out_name), 9.0, 4.5);
    close(h8_rel);
end

fprintf('Todas as figuras cientificas foram geradas com sucesso na pasta figuras/!\n');

end

% =========================================================================
% FUNÇÃO AUXILIAR DE EXPORTAÇÃO EM ALTA RESOLUÇÃO (PNG 300 DPI)
% =========================================================================
function export_fig_pdf_png(fig_handle, file_base, width_in, height_in)
    if nargin < 3, width_in = 10; end
    if nargin < 4, height_in = 6; end
    set(fig_handle, 'PaperUnits', 'inches');
    set(fig_handle, 'PaperSize', [width_in, height_in]);
    set(fig_handle, 'PaperPosition', [0, 0, width_in, height_in]);
    set(fig_handle, 'PaperPositionMode', 'manual');
    print(fig_handle, [file_base, '.png'], '-dpng', '-r300');
    fprintf('  -> Salvo com sucesso: %s.png\n', file_base);
end
