import sys, os
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
# -*- coding: utf-8 -*-
"""
run_experimentos.py - Tabela I, ablacao, confiabilidade, multi-semente e figuras.
Uso:  python run_experimentos.py --sementes 10 --saida saida
Opcoes: --orbitas 3 --orbitas-ablacao 1 --semente0 0 --sem-figuras
"""
import argparse, csv, json, os, time
import numpy as np
import tcas_core as C
import tcas_dvfs as D

REF = "Throttling 85/75"; PROP = "TCAS-DVFS"


def governadores(g):
    return {"Ondemand": lambda: C.gov_ondemand(),
            "Ondemand+trip 95/85": lambda: C.gov_ondemand(95.0, 85.0),
            "Throttling 85/75": lambda: C.gov_throttling(85.0, 75.0),
            "Throttling 83/73": lambda: C.gov_throttling(83.0, 73.0),
            "Space-PID": lambda: C.gov_pid(*g),
            "Power Cap 65 W": lambda: C.gov_power_cap(65.0),
            "TCA-DVFS (simples)": lambda: C.gov_tca(),
            "TCAS-DVFS": lambda: D.gov_tcas_dvfs()}


def csv_dict(caminho, linhas):
    chaves = []
    for l in linhas:
        for k in l:
            if k not in chaves:
                chaves.append(k)
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=chaves, delimiter=";"); w.writeheader(); w.writerows(linhas)


def linha(nome, m):
    return {"estrategia": nome, **{k: v for k, v in m.items() if k != "info"}, **m["info"]}


def fig_arquitetura(caminho):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4)
    caixas = [(0.1, 2.8, "Planta CTM 5x5\n(4 dies + radiador)"), (3.5, 2.8, "Sensores\n(ruido 0,25 K)"),
              (6.9, 2.8, "EKF (Joseph)\nestado T-hat"), (6.9, 0.5, "MPC (a cada 2 s)\nrestricao T <= 83 C"),
              (3.5, 0.5, "Escalonador SDP\n(a cada 120 s)"), (0.1, 0.5, "Barreira de retaguarda\n(T-hat >= 84,5 C)")]
    for x, y, txt in caixas:
        ax.add_patch(FancyBboxPatch((x, y), 3.0, 1.2, boxstyle="round,pad=0.05", fc="#eef3fa", ec="k"))
        ax.text(x + 1.5, y + 0.6, txt, ha="center", va="center", fontsize=7.5)
    seta = lambda a, b: ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.1))
    seta((3.15, 3.4), (3.5, 3.4)); seta((6.55, 3.4), (6.9, 3.4)); seta((8.4, 2.8), (8.4, 1.7))
    seta((6.9, 1.1), (6.55, 1.1)); seta((3.5, 1.1), (3.15, 1.1)); seta((1.6, 1.7), (1.6, 2.8))
    ax.text(1.75, 2.25, "f (GHz)", fontsize=7.5)
    ax.text(0.1, 4.2, "Entradas conhecidas: carga alpha(t) e fluxo ambiental Q_ext(t)", fontsize=7.5)
    fig.savefig(caminho, dpi=200, bbox_inches="tight"); plt.close(fig)


def figuras(runs, sc, pasta):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = sc["t"][:-1] / 60.0
    mostrar = ["Ondemand", REF, "Space-PID", "Power Cap 65 W", PROP]
    suav = lambda v: np.convolve(v, np.ones(240) / 240, mode="same")

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    for n in mostrar:
        r = runs[n]
        ax[0].plot(t, r["T"].max(1), lw=1.0, label=n)
        ax[1].plot(t, suav(r["fps"]), lw=1.0, label=n)
    ax[0].axhline(85, color="k", ls="--", lw=0.8); ax[0].set_ylim(20, 140)
    ax[0].set_ylabel("T_max (C) [eixo cortado em 140 C]"); ax[1].set_ylabel("FPS agregado (media 120 s)")
    ax[1].set_xlabel("Tempo (min)"); ax[0].legend(fontsize=7, ncol=2); [a.grid(alpha=0.3) for a in ax]
    fig.savefig(os.path.join(pasta, "fig_transiente.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for a, n in zip(ax, [REF, "Space-PID", PROP]):
        a.plot(t, runs[n]["T"], lw=0.7); a.axhline(85, color="k", ls="--", lw=0.8)
        a.set_ylim(60, 90); a.set_ylabel(n + " (C)"); a.grid(alpha=0.3)
    ax[-1].set_xlabel("Tempo (min)")
    fig.savefig(os.path.join(pasta, "fig_trajetorias.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for n in [REF, "Space-PID", PROP]:
        T = runs[n]["T"]; ax.plot(t, suav(T.max(1) - T.min(1)), lw=1.0, label=n)
    ax.set_xlabel("Tempo (min)"); ax.set_ylabel("Delta T espacial (K)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(pasta, "fig_gradiente.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    s0 = min(int(2 * 3600 / sc["dt"]), len(t) // 2); fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for n in [REF, PROP]:
        a = np.array([x for i in range(4) for x, w in C.rainflow(runs[n]["T"][s0:, i], 0.25)])
        if len(a):
            ax.hist(a, bins=np.logspace(-0.7, 1.3, 30), histtype="step", lw=1.3, label=n)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Amplitude do ciclo (K)")
    ax.set_ylabel("Numero de ciclos"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(pasta, "fig_confiabilidade.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    fig_arquitetura(os.path.join(pasta, "fig_arquitetura.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=10)
    ap.add_argument("--semente0", type=int, default=0)
    ap.add_argument("--orbitas", type=int, default=3)
    ap.add_argument("--orbitas-ablacao", type=int, default=1)
    ap.add_argument("--saida", default="saida")
    ap.add_argument("--sem-figuras", action="store_true")
    a = ap.parse_args()
    pasta_fig = os.path.join(a.saida, "figuras"); os.makedirs(pasta_fig, exist_ok=True)
    t0 = time.time(); sc = C.cenario(n_orb=a.orbitas); ganhos = C.autotune_pid(sc)
    print("PID (ITAE): Kp=%.3f Ki=%.4f Kd=%.3f | nt=%d" % (*ganhos, sc["nt"]), flush=True)
    gov = governadores(ganhos); res = {"pid": ganhos, "nt": sc["nt"], "t_orb": sc["t_orb"], "orbitas": a.orbitas}

    # ---- Tabela I: uma semente, todos os governadores
    runs = {}; linhas = []; res["tabela1"] = {}
    for nome, mk in gov.items():
        runs[nome] = C.simular(sc, mk(), a.semente0)
        m = C.metricas(runs[nome], sc); res["tabela1"][nome] = m; linhas.append(linha(nome, m))
        print("%-20s Tmax=%7.2f dTpico=%5.2f dTmed=%5.2f tviol=%8.1f FPS=%6.2f EDP=%6.3f E=%7.1f (%.0f s)" %
              (nome, m["Tmax"], m["dT_pico"], m["dT_med"], m["t_viol_s"], m["FPS"], m["EDP"], m["E_kJ"], time.time() - t0), flush=True)
    csv_dict(os.path.join(a.saida, "tabela1.csv"), linhas)

    # ---- Ablacao (menos orbitas; mesma semente)
    sca = C.cenario(n_orb=a.orbitas_ablacao); res["ablacao"] = {}; linhas = []
    for nome, mk in {"completo (Hp=30)": lambda: D.gov_tcas_dvfs(),
                     "sem SDP": lambda: D.gov_tcas_dvfs(usa_sched=False),
                     "sem MPC": lambda: D.gov_tcas_dvfs(usa_mpc=False),
                     "Hp = 10": lambda: D.gov_tcas_dvfs(hp=10),
                     "Hp = 60": lambda: D.gov_tcas_dvfs(hp=60),
                     "TCA simples": lambda: C.gov_tca()}.items():
        m = C.metricas(C.simular(sca, mk(), a.semente0), sca); res["ablacao"][nome] = m; linhas.append(linha(nome, m))
        print("ablacao %-18s Tmax=%6.2f dTpico=%5.2f FPS=%6.2f EDP=%6.3f" % (nome, m["Tmax"], m["dT_pico"], m["FPS"], m["EDP"]), flush=True)
    csv_dict(os.path.join(a.saida, "ablacao.csv"), linhas)

    # ---- Confiabilidade indicativa (historicos da Tabela I)
    skip = 2.0 if a.orbitas >= 3 else 0.3
    res["conf_vs_thr85"] = C.confiabilidade(runs[REF], runs[PROP], sc, skip_h=skip)
    res["conf_vs_thr83"] = C.confiabilidade(runs["Throttling 83/73"], runs[PROP], sc, skip_h=skip)

    # ---- Multi-semente (tres estrategias principais)
    chaves = ["Tmax", "t_viol_s", "dT_pico", "dT_med", "FPS", "EDP", "E_kJ", "Tmed"]
    seeds = list(range(a.semente0, a.semente0 + a.sementes)); acum = {n: [] for n in (REF, "Space-PID", PROP)}
    for sd in seeds:
        for n in acum:
            acum[n].append(C.metricas(C.simular(sc, gov[n](), sd), sc, arred=False))
        print("semente %d concluida (%.0f s)" % (sd, time.time() - t0), flush=True)
    res["sementes"] = seeds; res["multi"] = {}; linhas = []
    for n, lst in acum.items():
        M = np.array([[m[k] for k in chaves] for m in lst])
        res["multi"][n] = {k: [float(M[:, i].mean()), float(M[:, i].std(ddof=1)) if len(M) > 1 else 0.0] for i, k in enumerate(chaves)}
        linhas.append({"estrategia": n, **{k + "_media": res["multi"][n][k][0] for k in chaves},
                       **{k + "_dp": res["multi"][n][k][1] for k in chaves}})
    csv_dict(os.path.join(a.saida, "tabela_sementes.csv"), linhas)

    with open(os.path.join(a.saida, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if not a.sem_figuras:
        try:
            figuras(runs, sc, pasta_fig)
        except ImportError:
            print("matplotlib ausente: figuras nao geradas.")
    print("Concluido em %.0f s. Saidas em %s/" % (time.time() - t0, a.saida))


if __name__ == "__main__":
    main()
