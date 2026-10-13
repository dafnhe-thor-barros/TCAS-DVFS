% =========================================================================
% PROJETO DE MESTRADO: CONTROLE ESTATÍSTICO/PREDITIVO DE CLUSTERS DE IA
% ETAPA 1 (CONSOLIDADO): Modelagem do Ambiente Orbital LEO e Balanço Térmico
%
% Melhorias Implementadas:
% 1. Alinhamento angular do zênite solar ao meio-dia orbital (pico de ~16.4 W)
% 2. Albedo difuso coerente com a iluminação diurna (~2.1 W)
% 3. Carga infravermelha terrestre dinâmica variante com a latitude (9.2 a 10.6 W)
% 4. Passo temporal ajustado para dt = 0.5 s (equilíbrio ideal)
% 5. Salvamento automático no formato MATLAB -v7 no diretório do script
%
% Compatibilidade: GNU Octave >= 6.0 e MATLAB >= R2020b
% =========================================================================
clear all; close all; clc;

fprintf('===================================================================\n');
fprintf(' ETAPA 1 (CONSOLIDADO): Cargas Ambientais LEO 500 km (ECSS/NASA)\n');
fprintf('===================================================================\n');

%% 1. PARÂMETROS ORBITAIS
dt          = 0.5;               % Passo temporal [s]
mu_E        = 398600.4418;       % Constante gravitacional da Terra [km^3/s^2]
R_E         = 6371.0;            % Raio médio da Terra [km]
H_orb       = 500.0;             % Altitude orbital [km]
R_orb       = R_E + H_orb;       % Raio orbital [km]

% Velocidade e período orbital kepleriano
v_orb       = sqrt(mu_E / R_orb);
omega_orb   = v_orb / R_orb;
T_orb       = 2 * pi / omega_orb;              % ~5668.1 s (94.47 min)
n_orbitas   = 3;
t_final     = n_orbitas * T_orb;
t           = (0:dt:t_final)';
Nt          = length(t);

%% 2. PARÂMETROS AMBIENTAIS E PROPRIEDADES DA SUPERFÍCIE (Tabela 5 do Artigo)
S_solar     = 1361.0;            % Constante solar [W/m^2]
a_E         = 0.30;              % Albedo médio da Terra
sigma       = 5.670374e-8;       % Constante de Stefan-Boltzmann [W/m^2/K^4]
T_space     = 2.73;              % Fundo cósmico do espaço profundo [K]

% Radiador aeroespacial (Al 6061-T6 com acabamento OSR/Tinta Branca)
A_rad       = 0.12;              % Área efetiva do radiador [m^2]
alpha_s     = 0.12;              % Absortividade solar óptica (ondas curtas)
epsilon     = 0.88;              % Emissividade infravermelha (ondas longas)
m_rad       = 1.45;              % Massa da placa de alumínio [kg]
cp_rad      = 896.0;             % Calor específico Al 6061-T6 [J/(kg K)]
C_rad       = m_rad * cp_rad;    % Capacitância térmica concentrada (~1299.2 J/K)

% Fator de visão para a Terra (placa lateral inclinada a 60° do nadir)
F_view_earth = 0.42;

%% 3. GEOMETRIA SOLAR E DINÂMICA DE ECLIPSE ANALÍTICA
inc_orb     = deg2rad(97.4);     % Órbita Sol-Síncrona (SSO)
beta_angle  = deg2rad(15.0);     % Ângulo beta solar SSO (~15°)

% Fração de eclipse e passagem solar
sin_eclipse = R_E / R_orb;
theta_ecl   = asin(sin_eclipse);
frac_sombra = 2 * theta_ecl / (2 * pi); % ~37.8% (35.7 min em sombra)
frac_sol    = 1 - frac_sombra;          % ~62.2% (58.8 min em sol)
meio_dia    = frac_sol / 2.0;           % Ponto culminante do zênite solar

% Função de iluminação com transição suave sigmoidal (penumbra de ~10 s)
fase = mod(t, T_orb) / T_orb;
k_trans = 120;
illum = 1 ./ (1 + exp((fase - frac_sol) * k_trans));

% Ângulo solar zenital com o meio-dia da passagem:
% No nascer do sol (fase=0) -> ang = -pi/2 (cos = 0)
% No meio-dia solar (fase=meio_dia) -> ang = 0 (cos = 1)
% No pôr do sol (fase=frac_sol) -> ang = +pi/2 (cos = 0)
ang_solar_zenith = (pi / 2.0) * (fase - meio_dia) / meio_dia;

% Projeção na normal da placa (inclinada a 60° do nadir -> 30° da vertical):
cos_theta_sun = max(0.0, cos(ang_solar_zenith) * cos(deg2rad(30.0)) * cos(beta_angle));

%% 4. FLUXOS TÉRMICOS RADIATIVOS ABSORVIDOS [W]
% 1. Solar direto absorvido (pico realista de ~16.4 W)
q_solar   = illum .* (S_solar * alpha_s * A_rad .* cos_theta_sun);

% 2. Albedo planetário difuso (pico realista de ~2.1 W)
q_albedo  = illum .* (S_solar * a_E * alpha_s * A_rad * F_view_earth .* cos_theta_sun);

% 3. Infravermelho da Terra com modulação latitudinal (SSO passa por pólos e equador)
% Temperatura efetiva da Terra: 255 K no equador e 240 K nos pólos
T_earth_dyn = 255.0 - 15.0 * (sin(pi * (fase - meio_dia) / frac_sol).^2);
q_IR        = sigma * (T_earth_dyn.^4) * epsilon * A_rad * F_view_earth;

% Carga ambiental externa total absorvida
q_ext_abs = q_solar + q_albedo + q_IR;

% Dissipação térmica nominal dos 4 processadores (60 W ± 10 W de oscilação DVFS)
q_chips_in = 60.0 + 10.0 * sin(2 * pi * t / (600));

%% 5. INTEGRAÇÃO TÉRMICA TRANSITÓRIA DO RADIADOR (HEUN / RK2)
T_rad    = zeros(Nt, 1);
T_rad(1) = 270.0; % [K] (~ -3.15 °C)

for k = 1:(Nt - 1)
    T_curr = T_rad(k);

    % Passo 1: Preditor
    q_out_1 = epsilon * sigma * A_rad * (T_curr^4 - T_space^4);
    f1 = (q_chips_in(k) + q_ext_abs(k) - q_out_1) / C_rad;
    T_pred  = T_curr + dt * f1;

    % Passo 2: Corretor (reavalia emissão e fluxos no instante k+1)
    q_out_2 = epsilon * sigma * A_rad * (T_pred^4 - T_space^4);
    f2 = (q_chips_in(k+1) + q_ext_abs(k+1) - q_out_2) / C_rad;

    % Atualização de 2ª ordem
    T_rad(k+1) = T_curr + 0.5 * dt * (f1 + f2);
end

%% 6. SALVAMENTO AUTOMÁTICO DOS DADOS
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
save_path = fullfile(script_dir, 'etapa1_dados_orbitais.mat');

save('-v7', save_path, 't', 'dt', 'Nt', 'T_orb', 'q_solar', 'q_albedo', ...
     'q_IR', 'q_ext_abs', 'T_rad', 'q_chips_in', 'A_rad', 'epsilon', 'alpha_s', ...
     'C_rad', 'sigma', 'T_space', 'F_view_earth');
fprintf('-> Dados da Etapa 1 salvos com sucesso em: %s\n', save_path);

%% 7. RESULTADOS E GRÁFICOS
figure('Name', 'Etapa 1 Refinada: Cargas Ambientais e Térmica do Radiador');
subplot(2, 1, 1);
plot(t/60, q_ext_abs, 'r-', 'LineWidth', 1.5); hold on;
plot(t/60, q_solar, 'y--', 'LineWidth', 1.0);
plot(t/60, q_albedo, 'g--', 'LineWidth', 1.0);
plot(t/60, q_IR, 'm--', 'LineWidth', 1.0);
grid on;
xlabel('Tempo Orbital [min]');
ylabel('Fluxo Térmico [W]');
title('Cargas Radiativas Absorvidas no Radiador LEO (500 km, 3 Órbitas)');
legend('Total Absorvido', 'Solar Direto', 'Albedo Difuso', 'IR Terrestre Dinâmico', 'Location', 'NorthEast');

subplot(2, 1, 2);
plot(t/60, T_rad - 273.15, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Tempo Orbital [min]');
ylabel('Temperatura do Radiador [°C]');
title('Temperatura Dinâmica da Placa Radiadora T_{rad}(t)');

fprintf('===================================================================\n');
fprintf(' RESUMO DOS RESULTADOS DA ETAPA 1 REFINADA:\n');
fprintf('  - Pico de Fluxo Solar Absorvido:       %.2f W\n', max(q_solar));
fprintf('  - Pico de Albedo Absorvido:            %.2f W\n', max(q_albedo));
fprintf('  - Faixa da Carga IR Terrestre:         %.2f W a %.2f W\n', min(q_IR), max(q_IR));
fprintf('  - Pico de Carga Externa Absorvida:     %.2f W\n', max(q_ext_abs));
fprintf('  - Mínimo de Carga no Eclipse:          %.2f W\n', min(q_ext_abs));
fprintf('  - Temperatura Máxima do Radiador:      %.2f °C (%.2f K)\n', max(T_rad - 273.15), max(T_rad));
fprintf('  - Temperatura Mínima do Radiador:      %.2f °C (%.2f K)\n', min(T_rad - 273.15), min(T_rad));
fprintf('===================================================================\n');
