import sys, os
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
# -*- coding: utf-8 -*-
"""
run_incerteza.py - Incerteza parametrica (+-amp em R_ij, R_r, c_die, C_rad) por hipercubo latino.
A PLANTA e perturbada; EKF, MPC e governadores mantem parametros NOMINAIS (erro de modelo).
Uso:  python run_incerteza.py --amostras 30 --orbitas 3 --amp 0.20 --semente 7 --saida saida_inc --workers 4
"""
import argparse, csv, json, os, time
from multiprocessing import Pool, cpu_count
import numpy as np
import tcas_core as C
import tcas_dvfs as D

FAT = ["Rij", "Rr", "c_die", "C_rad"]
MET = ["Tmax", "t_viol_s", "dT_med", "dT_pico", "FPS", "EDP"]
Z975 = 1.959964


def lhs(n, d, rng, amp):
    u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    return 1.0 + amp * (2.0 * u - 1.0)


def t_975(df):                                   # quantil t(0,975; df), expansao de Cornish-Fisher
    z = Z975
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def ic(x, rng, B=4000):
    x = np.asarray(x, float); n = len(x); m = float(np.mean(x)); s = float(np.std(x, ddof=1))
    h = t_975(n - 1) * s / np.sqrt(n)
    bs = np.sort([rng.choice(x, n, replace=True).mean() for _ in range(B)])
    return {"media": m, "dp": s, "t_lo": m - h, "t_hi": m + h,
            "boot_lo": float(bs[int(0.025 * B)]), "boot_hi": float(bs[int(0.975 * B) - 1])}


def rank(v):
    """Atribui postos médios (average ranks) tratando empates canonicamente."""
    v = np.asarray(v)
    order = np.argsort(v)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    vals, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            mask = (inv == i)
            r[mask] = r[mask].mean()
    return r


def spearman(a, b):
    """Coeficiente de correlação de postos de Spearman com tratamento exato de empates."""
    try:
        from scipy.stats import spearmanr
        res = spearmanr(a, b)
        val = res.correlation if hasattr(res, 'correlation') else res.statistic
        return float(val) if np.isfinite(val) else 0.0
    except ImportError:
        ra, rb = rank(a), rank(b)
        std_a, std_b = np.std(ra), np.std(rb)
        if std_a < 1e-12 or std_b < 1e-12:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])


def clopper_pearson_sup(k, n, conf=0.95):
    """
    Limite superior unilateral (one-sided, 1 - alfa) do IC exato de Clopper-Pearson.
    Para k=0, retorna p = 1 - (1 - conf)**(1/n).
    Para k=n, retorna 1.0.
    """
    if k >= n:
        return 1.0
    if k == 0:
        return float(1.0 - (1.0 - conf) ** (1.0 / n))
    from math import comb
    alfa = 1.0 - conf
    def cdf(p):
        return sum(comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) > alfa:
            lo = mid
        else:
            hi = mid
    return float(hi)


def linha(r, sc):
    m = C.metricas(r, sc, arred=False); return [m[k] for k in MET]


def simular_amostra(args):
    i, f_vals, n_orb = args
    sc = C.cenario(n_orb=n_orb)
    par = C.parametros(*f_vals)
    nomes = [
        ("Throttling 85/75", lambda: C.gov_throttling(85.0, 75.0)),
        ("Throttling 83/73", lambda: C.gov_throttling(83.0, 73.0)),
        ("TCAS-DVFS", lambda: D.gov_tcas_dvfs())
    ]
    res_amostra = {}
    linhas_amostra = []
    info_tcas = {}
    for n, mk in nomes:
        sim_out = C.simular(sc, mk(), seed=i, par=par)
        m = C.metricas(sim_out, sc, arred=False)
        v = [m[k] for k in MET]
        res_amostra[n] = v
        extra_cols = {}
        if n == "TCAS-DVFS":
            inf = m.get("info", {})
            info_tcas = {
                "n_mpc": int(inf.get("n_mpc", 0)),
                "n_converged_qp": int(inf.get("n_converged_qp", 0)),
                "n_max_iter_qp": int(inf.get("n_max_iter_qp", 0)),
                "n_fallback_mpc": int(inf.get("n_fallback_mpc", 0)),
                "Tmax_fallback_C": float(inf.get("Tmax_fallback_C", 0.0)),
                "n_barreira": int(inf.get("n_barreira", 0)),
            }
            extra_cols = info_tcas
        linhas_amostra.append({
            "amostra": i,
            "estrategia": n,
            **{FAT[j]: f_vals[j] for j in range(4)},
            **dict(zip(MET, v)),
            **extra_cols,
        })
    return i, res_amostra, linhas_amostra, info_tcas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostras", type=int, default=30); ap.add_argument("--orbitas", type=int, default=3)
    ap.add_argument("--amp", type=float, default=0.20); ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--saida", default="saida_inc")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.saida, exist_ok=True); t0 = time.time()
    sc = C.cenario(n_orb=a.orbitas); rng = np.random.default_rng(a.semente); Fm = lhs(a.amostras, 4, rng, a.amp)

    n_workers = a.workers if a.workers > 0 else min(cpu_count(), 8)
    tarefas = [(i, Fm[i], a.orbitas) for i in range(a.amostras)]

    nomes_lista = ["Throttling 85/75", "Throttling 83/73", "TCAS-DVFS"]
    R = {n: [None] * a.amostras for n in nomes_lista}
    infos_tcas = [None] * a.amostras
    linhas = []

    print(f"Iniciando execucao de incerteza ({a.amostras} amostras, {a.orbitas} orbitas, {n_workers} workers)...", flush=True)

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for idx, (i, res_amostra, linhas_amostra, info_tcas) in enumerate(pool.imap_unordered(simular_amostra, tarefas)):
                for n in nomes_lista:
                    R[n][i] = res_amostra[n]
                infos_tcas[i] = info_tcas
                linhas.extend(linhas_amostra)
                print("amostra %d concluida [%d/%d] (%.0f s)" % (i + 1, idx + 1, a.amostras, time.time() - t0), flush=True)
    else:
        for idx, t_args in enumerate(tarefas):
            i, res_amostra, linhas_amostra, info_tcas = simular_amostra(t_args)
            for n in nomes_lista:
                R[n][i] = res_amostra[n]
            infos_tcas[i] = info_tcas
            linhas.extend(linhas_amostra)
            print("amostra %d/%d (%.0f s)" % (i + 1, a.amostras, time.time() - t0), flush=True)

    # Ordenar linhas por amostra para consistencia
    linhas.sort(key=lambda x: (x["amostra"], x["estrategia"]))
    R = {n: np.array(v) for n, v in R.items()}

    chaves_csv = []
    for l in linhas:
        for k in l:
            if k not in chaves_csv:
                chaves_csv.append(k)
    with open(os.path.join(a.saida, "amostras.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=chaves_csv, delimiter=";"); w.writeheader(); w.writerows(linhas)

    rb = np.random.default_rng(123)
    res = {
        "n": a.amostras,
        "amp": a.amp,
        "orbitas": a.orbitas,
        "escopo_e_metodo_lhs": (
            "Latin Hypercube Sampling (LHS) estratificada sobre os 4 parametros passivos da rede termica "
            "(Rij, Rr, c_die, C_rad). Como LHS nao constitui amostragem Bernoulli IID, os resultados sao "
            "reportados como analise de sensibilidade descritiva sem inferencia binomial de Clopper-Pearson."
        ),
        "individual": {},
        "pareado_pct": {},
        "pareado_abs": {},
    }
    for n, M in R.items():
        res["individual"][n] = {MET[j]: ic(M[:, j], rb) for j in range(6)}
        k = int((M[:, 1] > 0).sum())
        res["individual"][n]["amostras_com_violacao"] = k
        res["individual"][n]["fracao_amostral_violacao"] = float(k / a.amostras)
    tot_mpc = sum(x["n_mpc"] for x in infos_tcas if x)
    tot_conv = sum(x["n_converged_qp"] for x in infos_tcas if x)
    tot_fb = sum(x["n_fallback_mpc"] for x in infos_tcas if x)
    tmax_fb = max((x["Tmax_fallback_C"] for x in infos_tcas if x), default=0.0)
    res["mpc_diagnostico_lhs"] = {
        "total_mpc_calls": int(tot_mpc),
        "total_converged_qp": int(tot_conv),
        "total_fallback_mpc": int(tot_fb),
        "convergence_rate_pct": float(100.0 * tot_conv / max(1, tot_mpc)),
        "fallback_rate_pct": float(100.0 * tot_fb / max(1, tot_mpc)),
        "Tmax_during_fallback_C": float(tmax_fb),
        "fallback_per_sample_mean": float(np.mean([x["n_fallback_mpc"] for x in infos_tcas if x])),
        "fallback_per_sample_max": int(max((x["n_fallback_mpc"] for x in infos_tcas if x), default=0)),
    }
    T = R["TCAS-DVFS"]
    for ref in ("Throttling 85/75", "Throttling 83/73"):
        B = R[ref]
        res["pareado_pct"][ref] = {MET[j]: ic((T[:, j] - B[:, j]) / B[:, j] * 100.0, rb) for j in (2, 3, 4, 5)}
        res["pareado_abs"][ref] = {"Tmax_K": ic(T[:, 0] - B[:, 0], rb)}
    res["spearman"] = {m: {FAT[i]: spearman(Fm[:, i], T[:, MET.index(m)]) for i in range(4)} for m in ("Tmax", "dT_med", "FPS")}
    with open(os.path.join(a.saida, "incerteza.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("Concluido em %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
