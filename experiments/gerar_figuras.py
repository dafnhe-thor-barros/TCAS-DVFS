# -*- coding: utf-8 -*-
"""
gerar_figuras.py - Gera todas as figuras do artigo em ingles (padrao Elsevier).
Salva em results/figuras/ e sincroniza em figuras/.
"""

import sys, os
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import time
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import tcas_core as C
import tcas_dvfs as D

REF = "Throttling 85/75"
PROP = "TCAS-DVFS"

def fig_arquitetura(caminho):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)

    caixas = [
        (0.1, 2.8, "5-Node Thermal Plant\n(4 silicon dies + radiator)"),
        (3.5, 2.8, "Thermal Sensors\n(Gaussian noise $\\sigma_s = 0.25$ K)"),
        (6.9, 2.8, "EKF (Joseph Form)\nState estimate $\\hat{T}_i(k)$"),
        (6.9, 0.5, "ADMM-MPC ($\\Delta t = 2$ s)\nConstraint $\\hat{T}_i \\leq 83^\\circ$C"),
        (3.5, 0.5, "SDP Task Scheduler\n($\\Delta t_{sched} = 120$ s)"),
        (0.1, 0.5, "Safety Backstop\n($\\hat{T}_i \\geq 84.5^\\circ$C)")
    ]

    for x, y, txt in caixas:
        ax.add_patch(FancyBboxPatch((x, y), 3.0, 1.2, boxstyle="round,pad=0.05", fc="#eef3fa", ec="k"))
        ax.text(x + 1.5, y + 0.6, txt, ha="center", va="center", fontsize=7.5)

    seta = lambda a, b: ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.1))
    seta((3.15, 3.4), (3.5, 3.4))
    seta((6.55, 3.4), (6.9, 3.4))
    seta((8.4, 2.8), (8.4, 1.7))
    seta((6.9, 1.1), (6.55, 1.1))
    seta((3.5, 1.1), (3.15, 1.1))
    seta((1.6, 1.7), (1.6, 2.8))

    ax.text(1.75, 2.25, "Frequency $\\mathbf{f}(k)$ (GHz)", fontsize=7.5)
    ax.text(0.1, 4.2, "Exogenous inputs: workload trace $\\boldsymbol{\\alpha}(t)$ and orbital flux $Q_{ext}(t)$", fontsize=7.5)
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Gerada: {caminho}")


def gerar_todas_figuras(pasta_saida, pasta_artigo_figuras, n_orb=3, semente=0):
    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(pasta_artigo_figuras, exist_ok=True)

    print("Iniciando simulacao dos governadores para geracao dos graficos em ingles...", flush=True)
    t0 = time.time()
    sc = C.cenario(n_orb=n_orb)
    ganhos = C.autotune_pid(sc)
    
    gov = {
        "Ondemand": lambda: C.gov_ondemand(),
        REF: lambda: C.gov_throttling(85.0, 75.0),
        "Space-PID": lambda: C.gov_pid(*ganhos),
        "Power Cap 65 W": lambda: C.gov_power_cap(65.0),
        PROP: lambda: D.gov_tcas_dvfs()
    }

    runs = {}
    for nome, mk in gov.items():
        t_sub = time.time()
        runs[nome] = C.simular(sc, mk(), semente)
        print(f"  - {nome} simulado em {time.time() - t_sub:.1f} s", flush=True)

    t = sc["t"][:-1] / 60.0
    mostrar = ["Ondemand", REF, "Space-PID", "Power Cap 65 W", PROP]
    suav = lambda v: np.convolve(v, np.ones(240) / 240, mode="same")

    # 1. Transiente
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for n in mostrar:
        r = runs[n]
        ax[0].plot(t, r["T"].max(1), lw=1.0, label=n)
        ax[1].plot(t, suav(r["fps"]), lw=1.0, label=n)
    ax[0].axhline(85, color="k", ls="--", lw=0.8)
    ax[0].text(t[-1] * 0.70, 87.5, r"Critical Limit ($85^\circ$C)", fontsize=7.5, color="k")
    ax[0].set_ylim(20, 140)
    ax[0].set_ylabel(r"$T_{\max}$ ($^\circ$C) [truncated at 140$^\circ$C]")
    ax[1].set_ylabel("Aggregate Throughput (FPS, 120-s MA)")
    ax[1].set_xlabel("Time (min)")
    ax[0].legend(fontsize=7, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    p_trans = os.path.join(pasta_saida, "fig_transiente.png")
    fig.savefig(p_trans, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Gerada: {p_trans}")

    # 2. Trajetorias
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for a, n in zip(ax, [REF, "Space-PID", PROP]):
        a.plot(t, runs[n]["T"], lw=0.7)
        a.axhline(85, color="k", ls="--", lw=0.8)
        a.set_ylim(60, 90)
        a.set_ylabel(f"{n}\n" + r"Junction ($^\circ$C)")
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("Time (min)")
    p_traj = os.path.join(pasta_saida, "fig_trajetorias.png")
    fig.savefig(p_traj, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Gerada: {p_traj}")

    # 3. Gradiente espacial
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for n in [REF, "Space-PID", PROP]:
        T = runs[n]["T"]
        ax.plot(t, suav(T.max(1) - T.min(1)), lw=1.0, label=n)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"Spatial Gradient $\Delta T$ (K)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p_grad = os.path.join(pasta_saida, "fig_gradiente.png")
    fig.savefig(p_grad, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Gerada: {p_grad}")

    # 4. Confiabilidade / Rainflow
    s0 = min(int(2 * 3600 / sc["dt"]), len(t) // 2)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for n in [REF, PROP]:
        a = np.array([x for i in range(4) for x, w in C.rainflow(runs[n]["T"][s0:, i], 0.25)])
        if len(a):
            ax.hist(a, bins=np.logspace(-0.7, 1.3, 30), histtype="step", lw=1.3, label=n)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Rainflow Thermal Cycle Range $\Delta T$ (K)")
    ax.set_ylabel("Cycle Count (Log Scale)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p_conf = os.path.join(pasta_saida, "fig_confiabilidade.png")
    fig.savefig(p_conf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Gerada: {p_conf}")

    # 5. Arquitetura
    p_arq = os.path.join(pasta_saida, "fig_arquitetura.png")
    fig_arquitetura(p_arq)

    # Sincronizar todas as figuras geradas para a pasta figuras do artigo
    for f in ["fig_transiente.png", "fig_trajetorias.png", "fig_gradiente.png", "fig_confiabilidade.png", "fig_arquitetura.png"]:
        origem = os.path.join(pasta_saida, f)
        destino = os.path.join(pasta_artigo_figuras, f)
        shutil.copy2(origem, destino)
        print(f"  [SYNC] Sincronizado para: {destino}")

    print(f"\nTodas as figuras em ingles foram geradas com sucesso em {time.time() - t0:.1f} s!")


if __name__ == "__main__":
    pasta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pasta_saida = os.path.join(pasta_raiz, "results", "figuras")
    pasta_artigo = os.path.join(pasta_raiz, "figuras")
    gerar_todas_figuras(pasta_saida, pasta_artigo)
