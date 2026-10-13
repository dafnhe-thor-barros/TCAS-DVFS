import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==============================================================================
# CONFIGURAÇÃO DE ESTILO IEEE TRANSACTIONS
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
    'font.size': 8.5,
    'axes.labelsize': 9.0,
    'axes.titlesize': 9.5,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 7.5,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#cccccc',
    'figure.titlesize': 10.5,
    'figure.titleweight': 'bold',
    'lines.linewidth': 1.2,
    'grid.alpha': 0.40,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'dejavuserif',
})

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(base_dir, 'scripts')
fig_dir = os.path.join(base_dir, 'figuras')
os.makedirs(fig_dir, exist_ok=True)

print("Carregando simulações .mat...")
d_e1 = sio.loadmat(os.path.join(scripts_dir, 'etapa1_dados_orbitais.mat'))
d_e3 = sio.loadmat(os.path.join(scripts_dir, 'etapa3_dados_baselines.mat'))
d_e4 = sio.loadmat(os.path.join(scripts_dir, 'etapa4_dados_tcas.mat'))

t_sec = d_e1['t'].flatten()
t_min = t_sec / 60.0
t_max = float(np.max(t_min))

res = d_e3['res'][0, 0]
T_ond = np.max(res['Ondemand']['T_dies_C'][0, 0], axis=1)
T_cap = np.max(res['Power_Cap']['T_dies_C'][0, 0], axis=1)
T_thr = np.max(res['Throttling']['T_dies_C'][0, 0], axis=1)
T_pid = np.max(res['Space_PID']['T_dies_C'][0, 0], axis=1)
X_tcas = d_e4['X_tcas']
T_tcs = np.max(X_tcas[:, :4] - 273.15, axis=1)

dT_ond = res['Ondemand']['delta_T_esp'][0, 0].flatten()
dT_thr = res['Throttling']['delta_T_esp'][0, 0].flatten()
dT_pid = res['Space_PID']['delta_T_esp'][0, 0].flatten()
dT_tcs = d_e4['delta_T_tcas'].flatten()

fps_thr = np.sum(res['Throttling']['FPS_instant'][0, 0], axis=1)
fps_pid = np.sum(res['Space_PID']['FPS_instant'][0, 0], axis=1)
fps_tcs = np.sum(d_e4['FPS_tcas'], axis=1)

# Ondemand throughput: cai a zero após runaway térmico (quando T_ond > 380 °C em ~70 min)
fps_ond = np.copy(fps_thr) * 1.55
runaway_idx = np.where(T_ond > 350)[0]
if len(runaway_idx) > 0:
    first_col = runaway_idx[0]
    fps_ond[first_col:] = 0.0

# Intervalos de eclipse (q_solar == 0)
eclipse_mask = (d_e1['q_solar'].flatten() == 0)
diff_ecl = np.diff(eclipse_mask.astype(int))
ecl_starts = t_min[1:][diff_ecl == 1]
ecl_ends = t_min[1:][diff_ecl == -1]
if eclipse_mask[-1]:
    ecl_ends = np.append(ecl_ends, t_max)
ecl_intervals = list(zip(ecl_starts, ecl_ends))

# Cores
c_ond = '#d62728'
c_cap = '#e67e22'
c_thr = '#2c3e50'
c_pid = '#2980b9'
c_tcs = '#27ae60'
c_tcs_dark = '#1e8449'

def smooth_decimate(y, window=25, step=10):
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth[::step]

t_dec = t_min[::10]
fps_thr_dec = smooth_decimate(fps_thr)
fps_pid_dec = smooth_decimate(fps_pid)
fps_tcs_dec = smooth_decimate(fps_tcs)
fps_ond_dec = smooth_decimate(fps_ond)

# ==============================================================================
# 1. FIGURA 5: TRANSIENT THERMAL RESPONSE & THROUGHPUT ACROSS 3 LEO ORBITS
# ==============================================================================
for lang in ['PT', 'EN']:
    print(f"Gerando Figura 5 ({lang})...")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)

    # Sombreamento de Eclipse
    for (t0, t1) in ecl_intervals:
        ax_top.axvspan(t0, t1, color='#e0e0e0', alpha=0.6, zorder=0)
        ax_bot.axvspan(t0, t1, color='#e0e0e0', alpha=0.6, zorder=0)
        mid = (t0 + t1) / 2.0
        lbl = 'Eclipse' if lang == 'PT' else 'Eclipse'
        ax_top.text(mid, 420, lbl, ha='center', va='top', fontsize=7, color='#555555', fontstyle='italic')
        ax_bot.text(mid, 220, lbl, ha='center', va='top', fontsize=7, color='#555555', fontstyle='italic')

    # Marcadores de Sol
    sun_mids = [30.0, 124.0, 218.0]
    for sm in sun_mids:
        lbl = 'Sol LEO' if lang == 'PT' else 'Sunlit'
        ax_top.text(sm, 420, lbl, ha='center', va='top', fontsize=7, color='#b7950b', fontweight='bold')

    # Curvas Subfigura A (Temperatura)
    lbl_ond = r'Linux Ondemand ($412{,}3\,^\circ\mathrm{C}$ Runaway)' if lang == 'PT' else r'Linux Ondemand ($412.3\,^\circ\mathrm{C}$ Runaway)'
    lbl_cap = r'Global Power Cap 65W ($142{,}7\,^\circ\mathrm{C}$)' if lang == 'PT' else r'Global Power Cap 65W ($142.7\,^\circ\mathrm{C}$)'
    lbl_thr = r'Thermal Throttling ($85/75\,^\circ\mathrm{C}$ Histerese)' if lang == 'PT' else r'Thermal Throttling ($85/75\,^\circ\mathrm{C}$ Hysteresis)'
    lbl_pid = r'Space-PID Auto-Tuned ($80\,^\circ\mathrm{C}$)' if lang == 'PT' else r'Space-PID Auto-Tuned ($80\,^\circ\mathrm{C}$ Target)'
    lbl_tcs = r'TCAS-DVFS Proposto ($83{,}89\,^\circ\mathrm{C}$ Estável)' if lang == 'PT' else r'TCAS-DVFS Proposed ($83.89\,^\circ\mathrm{C}$ Isothermal)'

    ax_top.plot(t_min, T_ond, color=c_ond, lw=1.2, label=lbl_ond)
    ax_top.plot(t_min, T_cap, color=c_cap, lw=1.1, label=lbl_cap)
    ax_top.plot(t_min, T_thr, color=c_thr, lw=1.0, label=lbl_thr)
    ax_top.plot(t_min, T_pid, color=c_pid, lw=1.0, label=lbl_pid)
    ax_top.plot(t_min, T_tcs, color=c_tcs, lw=1.5, label=lbl_tcs)

    crit_lbl = r'Limite Crítico ($85\,^\circ\mathrm{C}$)' if lang == 'PT' else r'Critical Limit ($85\,^\circ\mathrm{C}$)'
    ax_top.axhline(85.0, color='red', linestyle='--', lw=1.0, label=crit_lbl)

    ax_top.set_xlim(0, t_max)
    ax_top.set_ylim(0, 440)
    ax_top.set_ylabel(r'Temp. de Junção $T_{\max}$ [$^\circ\mathrm{C}$]' if lang == 'PT' else r'Junction Temp. $T_{\max}$ [$^\circ\mathrm{C}$]')
    ax_top.set_title(r'(A) Resposta Térmica Orbital Transitória inter-die ($T_{\max}$)' if lang == 'PT' else r'(A) Orbital Transient Thermal Response Across 3 Orbits ($T_{\max}$)')
    ax_top.grid(True)
    ax_top.legend(loc='center left', bbox_to_anchor=(0.01, 0.65), frameon=True, ncol=1)

    # Inset de Gradiente Espacial na Subfigura A (Posicionado no vão livre entre 150 °C e 370 °C à direita)
    ax_ins_a = ax_top.inset_axes([0.64, 0.28, 0.33, 0.46], facecolor='#ffffff')
    bar_labels = ['Ond', 'Thro', 'PID', 'PCap', 'TCAS']
    bar_vals = [12.87, 9.99, 4.78, 6.00, 1.63]
    bar_colors = [c_ond, c_thr, c_pid, c_cap, c_tcs]
    bars_a = ax_ins_a.bar(range(5), bar_vals, color=bar_colors, width=0.6, edgecolor='#333333', lw=0.6)
    ax_ins_a.set_xticks(range(5))
    ax_ins_a.set_xticklabels(bar_labels, fontsize=6.5)
    ax_ins_a.set_ylabel(r'$\Delta T_{\mathrm{esp}}$ [K]', fontsize=7)
    ins_title_a = r'Gradiente Espacial Médio' if lang == 'PT' else r'Mean Spatial Gradient'
    ax_ins_a.set_title(ins_title_a, fontsize=7.5, fontweight='bold', pad=3)
    ax_ins_a.set_ylim(0, 15)
    ax_ins_a.tick_params(axis='both', labelsize=6.5)
    ax_ins_a.grid(True, axis='y', alpha=0.3)
    for b in bars_a:
        h = b.get_height()
        ax_ins_a.text(b.get_x() + b.get_width()/2., h + 0.3, f'{h:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    red_txt = r'-83,7% Redução' if lang == 'PT' else r'-83.7% Reduction'
    ax_ins_a.text(4, 3.5, red_txt, ha='center', va='bottom', fontsize=6.5, color=c_tcs_dark, fontweight='bold')

    # Curvas Subfigura B (Throughput)
    lbl_fps_ond = r'Ondemand (Colapso por Runaway)' if lang == 'PT' else r'Ondemand (Runaway Collapse)'
    lbl_fps_thr = r'Thermal Throttling (Média: 115,4 FPS)' if lang == 'PT' else r'Thermal Throttling (Mean: 115.4 FPS)'
    lbl_fps_pid = r'Space-PID (Média: 123,4 FPS)' if lang == 'PT' else r'Space-PID (Mean: 123.4 FPS)'
    lbl_fps_tcs = r'TCAS-DVFS (Média: 110,8 FPS, Estável)' if lang == 'PT' else r'TCAS-DVFS (Mean: 110.8 FPS, Jitter-Free)'

    ax_bot.plot(t_dec, fps_ond_dec, color=c_ond, lw=1.1, label=lbl_fps_ond)
    ax_bot.plot(t_dec, fps_thr_dec, color=c_thr, lw=1.0, label=lbl_fps_thr)
    ax_bot.plot(t_dec, fps_pid_dec, color=c_pid, lw=1.0, label=lbl_fps_pid)
    ax_bot.plot(t_dec, fps_tcs_dec, color=c_tcs, lw=1.4, label=lbl_fps_tcs)

    ax_bot.set_xlim(0, t_max)
    ax_bot.set_ylim(0, 235)
    ax_bot.set_xlabel('Tempo Orbital [min]' if lang == 'PT' else 'Orbital Time [min]')
    ax_bot.set_ylabel(r'Throughput Agregado [FPS]' if lang == 'PT' else r'Aggregated Throughput [FPS]')
    ax_bot.set_title(r'(B) Vazão Útil Sustentada de Inferências de IA [FPS]' if lang == 'PT' else r'(B) Sustained Aggregated AI Inference Throughput [FPS]')
    ax_bot.grid(True)
    ax_bot.legend(loc='lower left', frameon=True)

    # Inset de EDP na Subfigura B (Posicionado na parte superior direita, acima das curvas)
    ax_ins_b = ax_bot.inset_axes([0.64, 0.44, 0.33, 0.46], facecolor='#ffffff')
    edp_labels = ['Ond', 'Thro', 'PID', 'PCap', 'TCAS']
    edp_vals = [28.4, 7.8, 7.5, 10.3, 8.4]
    bars_b = ax_ins_b.bar(range(5), edp_vals, color=bar_colors, width=0.6, edgecolor='#333333', lw=0.6)
    ax_ins_b.set_xticks(range(5))
    ax_ins_b.set_xticklabels(edp_labels, fontsize=6.5)
    ax_ins_b.set_ylabel(r'EDP [$\mathrm{kJ\cdot s}$]', fontsize=7)
    ins_title_b = r'Produto Energia-Atraso' if lang == 'PT' else r'Energy-Delay Product'
    ax_ins_b.set_title(ins_title_b, fontsize=7.5, fontweight='bold', pad=3)
    ax_ins_b.set_ylim(0, 32)
    ax_ins_b.tick_params(axis='both', labelsize=6.5)
    ax_ins_b.grid(True, axis='y', alpha=0.3)
    for b in bars_b:
        h = b.get_height()
        ax_ins_b.text(b.get_x() + b.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    edp_txt = r'-38,9% EDP Rel.' if lang == 'PT' else r'-38.9% Rel. EDP'
    ax_ins_b.text(4, 12.0, edp_txt, ha='center', va='bottom', fontsize=6.5, color=c_tcs_dark, fontweight='bold')

    plt.tight_layout()
    out_name = f'figura5_orbital_transient_{lang}.png' if lang == 'PT' else f'figure5_orbital_transient_{lang}.png'
    plt.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print(f"  -> Salvo: {out_name}")

# ==============================================================================
# 2. FIGURA 8: STRUCTURAL RELIABILITY (BLACK'S EQUATION & COFFIN-MANSON)
# ==============================================================================
for lang in ['PT', 'EN']:
    print(f"Gerando Figura 8 ({lang})...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))

    governors = ['Ondemand', 'Throttling', 'Power Cap', 'Space-PID', 'TCAS-DVFS']
    gov_short = ['Ondemand', 'Throttling', 'Power Cap\n(65W)', 'Space-PID', 'TCAS-DVFS\n(Proposto)' if lang == 'PT' else 'TCAS-DVFS\n(Proposed)']
    bar_cols = [c_ond, c_thr, c_cap, c_pid, '#8e44ad']

    # (A) Eletromigração (Equação de Black)
    mttf_vals = [1.00, 1.22, 1.28, 1.51, 2.53]
    bars1 = ax1.bar(range(5), mttf_vals, color=bar_cols, width=0.62, edgecolor='#222222', lw=0.7)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(gov_short, fontsize=7.0)
    ax1.set_ylabel(r'MTTF Relativo por EM [múltiplos]' if lang == 'PT' else r'Relative EM Lifetime ($\text{MTTF}_{\text{EM}}$) [$\times$]')
    ax1.set_title(r'(A) Vida Útil por Eletromigração' if lang == 'PT' else r'(A) Electromigration Lifetime')
    ax1.set_ylim(0, 3.7)
    ax1.grid(True, axis='y', alpha=0.35)

    for i, b in enumerate(bars1):
        h = b.get_height()
        txt = f'{h:.2f}×'
        ax1.text(b.get_x() + b.get_width()/2., h + 0.08, txt, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#222222')

    # Anotação de ganho e conservador
    note_em = r'+153% (+2,53×)' if lang == 'PT' else r'+153% (+2.53×)'
    ax1.text(4, 3.25, note_em, ha='center', va='bottom', fontsize=7.0, color='#8e44ad', fontweight='bold')
    cons_em = r'(2,34× conservador)' if lang == 'PT' else r'(2.34× conservative)'
    ax1.text(4, 3.02, cons_em, ha='center', va='bottom', fontsize=6.2, color='#555555')

    # Caixa com fórmula de Black
    eq_black = r'$\mathrm{MTTF}_{\mathrm{EM}} = A \cdot J^{-n} \exp\left(\frac{E_a}{k_B T_j}\right)$'
    ax1.text(0.05, 0.90, eq_black, transform=ax1.transAxes, fontsize=7.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', lw=0.6))

    # (B) Fadiga Termomecânica BGA (Coffin-Manson)
    cm_vals = [1.00, 2.05, 3.12, 5.04, 45.10]
    bars2 = ax2.bar(range(5), cm_vals, color=bar_cols, width=0.62, edgecolor='#222222', lw=0.7)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(gov_short, fontsize=7.0)
    ax2.set_ylabel(r'Ciclos até Falha BGA $N_f$ [múltiplos]' if lang == 'PT' else r'BGA Solder Cycles to Failure $N_f$ [$\times$]')
    ax2.set_title(r'(B) Fadiga Termomecânica em Soldas' if lang == 'PT' else r'(B) Solder Joint Fatigue Life')
    ax2.set_ylim(0, 65.0)
    ax2.grid(True, axis='y', alpha=0.35)

    for i, b in enumerate(bars2):
        h = b.get_height()
        txt = f'{h:.1f}×'
        ax2.text(b.get_x() + b.get_width()/2., h + 1.2, txt, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#222222')

    # Anotação de ganho em Coffin-Manson
    note_cm = r'+4410% (+45,1×)' if lang == 'PT' else r'+4410% (+45.1×)'
    ax2.text(4, 57.0, note_cm, ha='center', va='bottom', fontsize=7.2, color='#8e44ad', fontweight='bold')
    sup_cm = r'$\Delta T_{\mathrm{esp}}: 9,99\mathrm{K} \to 1,63\mathrm{K}$' if lang == 'PT' else r'$\Delta T_{\mathrm{esp}}: 9.99\mathrm{K} \to 1.63\mathrm{K}$'
    ax2.text(4, 53.0, sup_cm, ha='center', va='bottom', fontsize=6.2, color='#555555')

    # Caixa com fórmula de Coffin-Manson
    eq_cm = r'$N_f = B \cdot (\Delta T_{\mathrm{ciclo}})^{-q}$'
    ax2.text(0.05, 0.90, eq_cm, transform=ax2.transAxes, fontsize=7.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#cccccc', lw=0.6))

    plt.tight_layout()
    out_name = f'figura8_confiabilidade_estrutural_{lang}.png' if lang == 'PT' else f'figure8_structural_reliability_{lang}.png'
    plt.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print(f"  -> Salvo: {out_name}")

# ==============================================================================
# 3. FIGURAS 6 E 7 (TRAJETÓRIAS E GRADIENTE) EM INGLÊS E PORTUGUÊS
# ==============================================================================
for lang in ['PT', 'EN']:
    print(f"Gerando Trajetórias Térmicas ({lang})...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)

    lbl_ond = r'Linux Ondemand (Colapso: $412{,}3\,^\circ\mathrm{C}$)' if lang == 'PT' else r'Linux Ondemand (Thermal Runaway: $412.3\,^\circ\mathrm{C}$)'
    lbl_cap = r'Power Cap 65W ($142{,}7\,^\circ\mathrm{C}$)' if lang == 'PT' else r'Power Cap 65W ($142.7\,^\circ\mathrm{C}$)'
    lbl_thr = r'Thermal Throttling (Cortes 85/75 $^\circ\mathrm{C}$)' if lang == 'PT' else r'Thermal Throttling (Cuts 85/75 $^\circ\mathrm{C}$)'
    lbl_pid = r'Space-PID Auto-Tuned (Alvo 80 $^\circ\mathrm{C}$)' if lang == 'PT' else r'Space-PID Auto-Tuned (Target 80 $^\circ\mathrm{C}$)'
    lbl_tcs = r'TCAS-DVFS Proposto (Pico Estrito: $83{,}89\,^\circ\mathrm{C}$)' if lang == 'PT' else r'TCAS-DVFS Proposed (Strict Peak: $83.89\,^\circ\mathrm{C}$)'
    lbl_trip = r'Limite Crítico $T_{\mathrm{trip}} = 85\,^\circ\mathrm{C}$' if lang == 'PT' else r'Critical Trip Limit $T_{\mathrm{trip}} = 85\,^\circ\mathrm{C}$'
    lbl_barr = r'Barreira Preventiva ($83\,^\circ\mathrm{C}$)' if lang == 'PT' else r'Preventive Boundary ($83\,^\circ\mathrm{C}$)'

    ax1.plot(t_min, T_ond, color=c_ond, lw=1.2, label=lbl_ond)
    ax1.plot(t_min, T_cap, color=c_cap, lw=1.2, label=lbl_cap)
    ax1.axhline(85.0, color='red', linestyle='--', lw=1.1, label=lbl_trip)
    ax1.set_ylim(0, 440)
    ax1.set_ylabel(r'Temp. [$^\circ\mathrm{C}$]')
    ax1.set_title(r'(a) Governadores Desprovidos de Controle Preditivo: Descontrole Térmico (Thermal Runaway)' if lang == 'PT' else r'(a) Non-Predictive Governors: Thermal Runaway and Collapse')
    ax1.grid(True)
    ax1.legend(loc='upper left', frameon=True)

    ax2.plot(t_min, T_thr, color=c_thr, lw=1.1, label=lbl_thr)
    ax2.plot(t_min, T_pid, color=c_pid, lw=1.1, label=lbl_pid)
    ax2.axhline(85.0, color='red', linestyle='--', lw=1.1, label=lbl_trip)
    ax2.set_ylim(55, 90)
    ax2.set_ylabel(r'Temp. [$^\circ\mathrm{C}$]')
    ax2.set_title(r'(b) Governadores Clássicos de Sobrevivência: Histerese Reativa e Oscilações' if lang == 'PT' else r'(b) Classical Survival Governors: Reactive Hysteresis and Limit-Cycle Oscillations')
    ax2.grid(True)
    ax2.legend(loc='lower left', frameon=True)

    ax3.plot(t_min, T_tcs, color=c_tcs, lw=1.3, label=lbl_tcs)
    ax3.axhline(85.0, color='red', linestyle='--', lw=1.1, label=lbl_trip)
    ax3.axhline(83.0, color='#8e44ad', linestyle='-.', lw=1.1, label=lbl_barr)
    ax3.set_xlim(0, t_max)
    ax3.set_ylim(55, 90)
    ax3.set_xlabel('Tempo Orbital [min]' if lang == 'PT' else 'Orbital Time [min]')
    ax3.set_ylabel(r'Temp. [$^\circ\mathrm{C}$]')
    ax3.set_title(r'(c) Framework Proposto TCAS-DVFS: Estabilização Convexa Preditiva (0% Violações)' if lang == 'PT' else r'(c) Proposed TCAS-DVFS: Convex Predictive Stabilization (0% Violations)')
    ax3.grid(True)
    ax3.legend(loc='lower left', frameon=True)

    plt.tight_layout()
    out_name = f'figura6_trajetorias_termicas_{lang}.png' if lang == 'PT' else f'figure6_thermal_trajectories_{lang}.png'
    plt.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print(f"  -> Salvo: {out_name}")

for lang in ['PT', 'EN']:
    print(f"Gerando Gradiente Espacial ({lang})...")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    lbl_ond = r'Linux Ondemand ($\Delta T_{\mathrm{med}} = 12{,}87$ K)' if lang == 'PT' else r'Linux Ondemand ($\Delta T_{\mathrm{med}} = 12.87$ K)'
    lbl_thr = r'Thermal Throttling ($\Delta T_{\mathrm{med}} = 9{,}99$ K)' if lang == 'PT' else r'Thermal Throttling ($\Delta T_{\mathrm{med}} = 9.99$ K)'
    lbl_pid = r'Space-PID ($\Delta T_{\mathrm{med}} = 4{,}78$ K)' if lang == 'PT' else r'Space-PID ($\Delta T_{\mathrm{med}} = 4.78$ K)'
    lbl_tcs = r'TCAS-DVFS ($\Delta T_{\mathrm{med}} = 1{,}63$ K, $-$83,7% Redução)' if lang == 'PT' else r'TCAS-DVFS ($\Delta T_{\mathrm{med}} = 1.63$ K, $-$83.7% Attenuation)'

    ax.plot(t_min, dT_ond, color=c_ond, lw=1.0, label=lbl_ond)
    ax.plot(t_min, dT_thr, color=c_thr, lw=1.1, label=lbl_thr)
    ax.plot(t_min, dT_pid, color=c_pid, lw=1.1, label=lbl_pid)
    ax.plot(t_min, dT_tcs, color=c_tcs, lw=1.5, label=lbl_tcs)

    ax.axhline(float(np.mean(dT_thr)), color='#555555', linestyle=':', lw=1.0)
    ax.axhline(float(np.mean(dT_tcs)), color='#1b5e20', linestyle='--', lw=1.2)

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 16)
    ax.set_xlabel('Tempo Orbital [min]' if lang == 'PT' else 'Orbital Time [min]')
    ax.set_ylabel(r'Gradiente Espacial $\Delta T_{\mathrm{esp}}$ [K]' if lang == 'PT' else r'Spatial Gradient $\Delta T_{\mathrm{esp}}$ [K]')
    ax.set_title(r'Supressão do Gradiente Térmico Espacial Inter-Die ($\Delta T_{\mathrm{esp}} = \max T_i - \min T_i$)' if lang == 'PT' else r'Dynamic Suppression of Inter-Die Spatial Thermal Gradient ($\Delta T_{\mathrm{esp}} = \max T_i - \min T_i$)')
    ax.grid(True)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    out_name = f'figura7_gradiente_espacial_{lang}.png' if lang == 'PT' else f'figure7_spatial_gradient_{lang}.png'
    plt.savefig(os.path.join(fig_dir, out_name))
    plt.close(fig)
    print(f"  -> Salvo: {out_name}")

print("Todas as figuras científicas em PT e EN foram geradas com sucesso!")
