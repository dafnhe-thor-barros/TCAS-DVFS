# -*- coding: utf-8 -*-
"""
validate_mpc_admm.py - Validação numérica forense do controlador LTV-MPC / ADMM (Fase 2: A06, A07, A08).

Executa:
1. A06: Comparação determinística entre o solver ADMM (OSQP-like) de `fabrica_mpc` e um solver QP
   de referência independente (`scipy.optimize.minimize` SLSQP com gradiente e Jacobiano analíticos
   e tolerância ftol=1e-12), avaliando:
   - Frequências ótimas f*
   - Variáveis auxiliares z*
   - Variáveis de folga (slacks) s*
   - Valor da função objetivo J(x*)
   - Violação máxima das restrições lineares da caixa l <= A x <= u
   - Resíduos primal (r_pri) e dual (r_dua)
   - Comparação Warm Start vs Cold Start em sequência temporal de múltiplos passos
   - Casos: (1) Factível nominal, (2) Restrições ativas no teto térmico, (3) Quase-inviável com ativação
     de soft-constraints (s > 0), (4) Orçamento fisicamente inviável (P_budget < P_min_cores).
2. A07: Análise de condicionamento numérico:
   - Número de condição kappa_2(Pm), kappa_2(A_raw), kappa_2(A_norm) e kappa_2(K_ADMM)
   - Sensibilidade paramétrica a rho_a, rho (gradiente espacial) e lambda (penalidade de slack).
3. A08: Validação do preditor térmico linearizado de 30 passos (60 s) contra a integração não linear
   exata de Heun da planta CTM de 5 estados, quantificando o erro por die (i=1..4) e por passo do
   horizonte (m=1..30) em múltiplos regimes operacionais.
"""
import csv
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tcas_core as C
import tcas_dvfs as D


def construir_qp_mpc(xh, fp, al, fnom, slope, pbud, hp=30, hc=8, dtc=2.0, t_cap=83.0,
                     w=1.0, Rw=0.2, rho=0.02, lam=200.0, par=None):
    """
    Reconstrói exatamente as matrizes do subproblema QP do LTV-MPC para auditoria independente.
    Minimiza (1/2) x^T Pm x + q^T x sujeito a lo <= A x <= up.
    """
    par = C.NOM if par is None else par
    N = 4
    nf = hc * N
    n_slacks = hp * N
    n_vars = nf + n_slacks
    D_ = np.eye(nf) - np.eye(nf, k=-N)
    ar = np.arange(hp)
    Jt = np.maximum(0, ar[:, None] - ar[None, :]).astype(float)
    c = float(par["C"])
    RR = float(par["RR"])
    Lp = np.asarray(par["LAP"], dtype=float)

    def ptot(T, f, al_):
        pd, pl = C.potencias(T, f, al_)
        return pd + pl

    T0_ = xh[:4]
    Trad = xh[4]
    P0 = ptot(T0_, fp, al)
    dd = 1e-3
    a = (ptot(T0_, fp + dd, al) - ptot(T0_, fp - dd, al)) / (2 * dd)
    b = ptot(T0_ + 0.5, fp, al) - ptot(T0_ - 0.5, fp, al)
    lm, V = np.linalg.eigh((-Lp - np.eye(N) / RR + np.diag(b)) / c)
    g = (np.exp(lm * dtc) - 1.0) / lm / c
    ms = np.arange(1, hp + 1)
    Apow = np.einsum("ik,mk,jk->mij", V, np.exp(np.outer(ms, lm) * dtc), V)
    W = np.einsum("ik,sk,jk->sij", V, g[None, :] * np.exp(np.outer(ar, lm) * dtc), V)
    Cs = np.cumsum(W, axis=0)
    dv = P0 - a * fp - b * T0_ + Trad / RR
    free = (
        np.einsum("mij,j->mi", Apow, T0_)
        + np.einsum("mij,j->mi", Cs, dv)
        + (slope * dtc / RR) * (Jt @ W.sum(2))
    )
    G0 = np.zeros((hp, N, hc, N))
    for j in range(min(hc - 1, hp)):
        G0[j:, :, j, :] = W[:hp - j]
    if hp >= hc:
        G0[hc - 1 :, :, hc - 1, :] = Cs[: hp - hc + 1]
    Gam = G0.reshape(hp * N, nf) * np.tile(a, hc)[None, :]
    Gh = Gam.reshape(hp, N, nf)
    Gh = (Gh - Gh.mean(1, keepdims=True)).reshape(hp * N, nf)
    fc = (free - free.mean(1, keepdims=True)).reshape(-1)

    Pm = np.zeros((n_vars, n_vars))
    Pm[:nf, :nf] = 2 * w * np.eye(nf) + 2 * Rw * (D_.T @ D_) + 2 * rho * (Gh.T @ Gh)
    Pm[nf:, nf:] = 2 * lam * np.eye(n_slacks)

    e0 = np.zeros(nf)
    e0[:N] = fp
    q = np.zeros(n_vars)
    q[:nf] = -2 * w * fnom.reshape(-1) - 2 * Rw * (D_.T @ e0) + 2 * rho * (Gh.T @ fc)

    Tc = t_cap + 273.15
    fl = free.reshape(-1)
    A1 = np.hstack([Gam, -np.eye(n_slacks)])
    A2 = np.hstack([np.eye(nf), np.zeros((nf, n_slacks))])
    A3 = np.hstack([np.zeros((n_slacks, nf)), np.eye(n_slacks)])
    A4 = np.zeros((hc, n_vars))
    for j in range(hc):
        A4[j, j * N : (j + 1) * N] = a

    A_raw = np.vstack([A1, A2, A3, A4])
    lo_raw = np.concatenate([np.full(n_slacks, -1e9), np.full(nf, C.F_MIN), np.zeros(n_slacks), np.full(hc, -1e9)])
    up_raw = np.concatenate([Tc - fl, np.full(nf, C.F_MAX), np.full(n_slacks, 1e3), np.full(hc, pbud - (P0 - a * fp).sum())])

    rn = np.abs(A_raw).max(1)
    rn[rn < 1e-9] = 1.0
    A = A_raw / rn[:, None]
    lo = lo_raw / rn
    up = up_raw / rn

    return {
        "Pm": Pm,
        "q": q,
        "A_raw": A_raw,
        "lo_raw": lo_raw,
        "up_raw": up_raw,
        "A": A,
        "lo": lo,
        "up": up,
        "rn": rn,
        "Gam": Gam,
        "free": free,
        "fl": fl,
        "Tc": Tc,
        "a": a,
        "b": b,
        "P0": P0,
        "nf": nf,
        "n_slacks": n_slacks,
        "n_vars": n_vars,
    }


def resolver_qp_referencia_slsqp(qp, fnom):
    """
    Resolve o subproblema QP (1/2) x^T Pm x + q^T x com precisão de referência
    usando SLSQP com gradiente e Jacobiano analíticos.
    Restrições:
      - Box bounds: f in [F_MIN, F_MAX], s in [0, 1000]
      - Desigualdades lineares:
        (1) Gam @ f - s <= Tc - fl
        (2) A4_f @ f <= up_raw_A4
    """
    Pm = qp["Pm"]
    q = qp["q"]
    nf = qp["nf"]
    n_slacks = qp["n_slacks"]
    n_vars = qp["n_vars"]

    def obj(x):
        return float(0.5 * (x @ (Pm @ x)) + q @ x)

    def grad(x):
        return Pm @ x + q

    # Restrições de desigualdade c(x) >= 0 para SLSQP: up_raw - A_ineq @ x >= 0
    # Linhas de A1 (0..120) e A4 (272..280)
    A_ineq = np.vstack([qp["A_raw"][:n_slacks], qp["A_raw"][n_slacks + nf + n_slacks :]])
    u_ineq = np.concatenate([qp["up_raw"][:n_slacks], qp["up_raw"][n_slacks + nf + n_slacks :]])

    cons = {
        "type": "ineq",
        "fun": lambda x: u_ineq - A_ineq @ x,
        "jac": lambda x: -A_ineq,
    }
    bounds = [(C.F_MIN, C.F_MAX)] * nf + [(0.0, 1e3)] * n_slacks
    x0 = np.concatenate([np.clip(fnom.reshape(-1), C.F_MIN, C.F_MAX), np.zeros(n_slacks)])
    # Se houver violação térmica inicial em x0, inicializa slacks compatíveis
    s0 = np.maximum(0.0, qp["fl"] + qp["Gam"] @ x0[:nf] - qp["Tc"])
    x0[nf:] = s0

    res = minimize(obj, x0, method="SLSQP", jac=grad, bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
    x_ref = res.x
    z_ref = np.clip(qp["A"] @ x_ref, qp["lo"], qp["up"])
    viol_box = float(np.maximum(0.0, np.maximum(qp["A"] @ x_ref - qp["up"], qp["lo"] - qp["A"] @ x_ref)).max())
    return {
        "x": x_ref,
        "f": x_ref[:nf],
        "s": x_ref[nf:],
        "z": z_ref,
        "J": obj(x_ref),
        "viol_box": viol_box,
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
    }


def executar_admm_detalhado(qp, fnom, iters=60, rho_a=1.0, sg=1e-6, al_=1.6, x_init=None, y_init=None):
    """
    Executa o loop ADMM exatamente como implementado em fabrica_mpc, retornando
    todas as variáveis internas (x, f, s, z, y, J, r_pri, r_dua) para auditoria.
    """
    Pm = qp["Pm"]
    q = qp["q"]
    A = qp["A"]
    lo = qp["lo"]
    up = qp["up"]
    nf = qp["nf"]
    n_slacks = qp["n_slacks"]
    n_vars = qp["n_vars"]

    if x_init is None:
        x = np.concatenate([fnom.reshape(-1), np.zeros(n_slacks)])
        y = np.zeros(A.shape[0])
    else:
        x = x_init.copy()
        y = y_init.copy()

    x[:nf] = np.clip(x[:nf], C.F_MIN, C.F_MAX)
    x[nf:] = np.maximum(0.0, x[nf:])
    z = np.clip(A @ x, lo, up)

    Kmat = Pm + sg * np.eye(n_vars) + rho_a * (A.T @ A)
    Kinv = np.linalg.inv(Kmat)

    r_pri = 0.0
    r_dua = 0.0
    eps_pri = 0.0
    eps_dua = 0.0
    it = 0
    for it in range(iters):
        z_old = z.copy()
        xt = Kinv @ (sg * x - q + A.T @ (rho_a * z - y))
        zt = A @ xt
        xr = al_ * xt + (1 - al_) * x
        zr = al_ * zt + (1 - al_) * z
        zn = np.clip(zr + y / rho_a, lo, up)
        y = y + rho_a * (zr - zn)
        x = xr
        z = zn
        r_pri = float(np.abs(A @ x - z).max())
        r_dua = float(np.abs(rho_a * (A.T @ (z - z_old))).max())

        eps_pri = 2e-3 + 1e-3 * max(float(np.abs(A @ x).max()), float(np.abs(z).max()))
        scale_dua = max(float(np.abs(Pm @ x).max()), float(np.abs(A.T @ y).max()), float(np.abs(q).max()))
        eps_dua = 5e-3 + 1e-3 * scale_dua

        if it >= 15 and it % 5 == 4 and r_pri <= eps_pri and r_dua <= eps_dua:
            break

    f_proj = np.clip(x[:nf], C.F_MIN, C.F_MAX)
    s_proj = np.maximum(0.0, x[nf:])
    x_proj = np.concatenate([f_proj, s_proj])
    J_raw = float(0.5 * (x @ (Pm @ x)) + q @ x)
    J_proj = float(0.5 * (x_proj @ (Pm @ x_proj)) + q @ x_proj)
    viol_box = float(np.maximum(0.0, np.maximum(A @ x - up, lo - A @ x)).max())
    viol_termica = float(max(0.0, (qp["fl"] + qp["Gam"] @ f_proj - qp["Tc"]).max()))
    converged = bool(r_pri <= eps_pri and r_dua <= eps_dua)

    return {
        "x": x,
        "f": f_proj,
        "s": s_proj,
        "z": z,
        "y": y,
        "J_raw": J_raw,
        "J_proj": J_proj,
        "viol_box": viol_box,
        "viol_termica_K": viol_termica,
        "r_pri": r_pri,
        "r_dua": r_dua,
        "eps_pri": eps_pri,
        "eps_dua": eps_dua,
        "it": it + 1,
        "converged": converged,
        "cond_Pm": float(np.linalg.cond(Pm)),
        "cond_A_raw": float(np.linalg.cond(qp["A_raw"])),
        "cond_A_norm": float(np.linalg.cond(A)),
        "cond_Kmat": float(np.linalg.cond(Kmat)),
    }


def main():
    print("=" * 76)
    print("  AUDITORIA NUMÉRICA DO MPC / ADMM (FASE 2: A06, A07, A08)")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 1. A06: Comparação contra solver de referência em 4 casos determinísticos
    # ------------------------------------------------------------------
    fnom = np.tile(np.array([1.25, 1.15, 1.05, 0.95]), (8, 1))
    al = np.array([0.90, 0.85, 0.80, 0.75])
    fp = np.array([1.15, 1.10, 1.00, 0.90])

    casos = [
        {
            "id": "C1_factivel_nominal",
            "descricao": "Operação nominal fria (T0 ~ 55 C), restrições térmicas inativas (s* = 0)",
            "xh": np.array([328.15, 327.65, 326.95, 327.15, 312.15]),
            "slope": 0.005,
            "pbud": 115.0,
        },
        {
            "id": "C2_restricoes_ativas",
            "descricao": "Operação aquecida (T0 ~ 79.5 C) e orçamento restrito (P_bud = 75 W), restrições ativas",
            "xh": np.array([352.65, 351.95, 351.15, 350.65, 336.15]),
            "slope": 0.015,
            "pbud": 75.0,
        },
        {
            "id": "C3_quase_inviavel_soft_constraint",
            "descricao": "Sobrecarga térmica inicial acima do teto (T0 = 83.4 C > T_cap = 83.0 C), exigindo slacks s* > 0",
            "xh": np.array([356.55, 356.25, 355.95, 355.65, 348.15]),
            "slope": 0.030,
            "pbud": 68.0,
        },
        {
            "id": "C4_inviavel_fisico_orcamento",
            "descricao": "Orçamento fisicamente abaixo do piso P_min_cores (P_bud = 12 W < P_min ~ 19.3 W)",
            "xh": np.array([354.15, 353.65, 353.15, 352.65, 346.15]),
            "slope": 0.025,
            "pbud": 12.0,
        },
    ]

    relatorio_casos = {}
    for c in casos:
        cid = c["id"]
        xh = c["xh"]
        pbud = c["pbud"]
        slope = c["slope"]

        pd_min, pl_min = C.potencias(xh[:4], np.full(4, C.F_MIN), al)
        p_min_cores = float((pd_min + pl_min).sum())

        qp = construir_qp_mpc(xh, fp, al, fnom, slope, pbud)
        ref = resolver_qp_referencia_slsqp(qp, fnom)
        admm60 = executar_admm_detalhado(qp, fnom, iters=60)
        admm300 = executar_admm_detalhado(qp, fnom, iters=300)

        # Verifica correspondência exata com fabrica_mpc de tcas_dvfs.py
        mpc_fn = D.fabrica_mpc(hp=30, hc=8, dtc=2.0, t_cap=83.0, iters=60)
        F_mod, dg_mod = mpc_fn(xh, fp, al, fnom, slope, pbud)
        diff_mod = float(np.max(np.abs(F_mod.reshape(-1) - admm60["f"])))

        err_f_60 = float(np.max(np.abs(admm60["f"] - ref["f"])))
        err_f_l2_60 = float(np.linalg.norm(admm60["f"] - ref["f"]))
        err_s_60 = float(np.max(np.abs(admm60["s"] - ref["s"])))
        err_z_60 = float(np.max(np.abs(admm60["z"] - ref["z"])))
        err_J_rel_60 = float(abs(admm60["J_proj"] - ref["J"]) / max(abs(ref["J"]), 1e-9))

        err_f_300 = float(np.max(np.abs(admm300["f"] - ref["f"])))
        err_s_300 = float(np.max(np.abs(admm300["s"] - ref["s"])))
        err_J_rel_300 = float(abs(admm300["J_proj"] - ref["J"]) / max(abs(ref["J"]), 1e-9))

        relatorio_casos[cid] = {
            "descricao": c["descricao"],
            "pbud_W": pbud,
            "P_min_cores_W": p_min_cores,
            "fisicamente_viavel_no_orcamento": bool(pbud >= p_min_cores),
            "bloqueado_por_contingencia_em_gov_tcas_dvfs": bool(pbud < p_min_cores),
            "consistencia_com_fabrica_mpc_linf": diff_mod,
            "slsqp_ref": {
                "success": ref["success"],
                "nit": ref["nit"],
                "J": ref["J"],
                "f_min": float(ref["f"].min()),
                "f_max": float(ref["f"].max()),
                "s_max": float(ref["s"].max()),
                "viol_box": ref["viol_box"],
            },
            "admm_60_iters": {
                "it": admm60["it"],
                "converged": admm60["converged"],
                "status_modulo": dg_mod["status"],
                "J_proj": admm60["J_proj"],
                "f_min": float(admm60["f"].min()),
                "f_max": float(admm60["f"].max()),
                "s_max": float(admm60["s"].max()),
                "r_pri": admm60["r_pri"],
                "r_dua": admm60["r_dua"],
                "viol_box": admm60["viol_box"],
                "viol_termica_K": admm60["viol_termica_K"],
                "err_f_linf_vs_ref": err_f_60,
                "err_f_l2_vs_ref": err_f_l2_60,
                "err_s_linf_vs_ref": err_s_60,
                "err_z_linf_vs_ref": err_z_60,
                "err_J_rel_vs_ref": err_J_rel_60,
            },
            "admm_300_iters": {
                "it": admm300["it"],
                "converged": admm300["converged"],
                "r_pri": admm300["r_pri"],
                "r_dua": admm300["r_dua"],
                "err_f_linf_vs_ref": err_f_300,
                "err_s_linf_vs_ref": err_s_300,
                "err_J_rel_vs_ref": err_J_rel_300,
            },
        }
        print(f"  [{cid}] conv60={admm60['converged']} (it={admm60['it']}) | "
              f"||df||_inf(60)={err_f_60:.4e} GHz | ||df||_inf(300)={err_f_300:.4e} GHz | "
              f"||ds||_inf(60)={err_s_60:.4e} K | dJ_rel(60)={err_J_rel_60:.4e}")

    # ------------------------------------------------------------------
    # 2. Warm Start vs Cold Start ao longo de 15 passos consecutivos
    # ------------------------------------------------------------------
    mpc_warm = D.fabrica_mpc(hp=30, hc=8, dtc=2.0, t_cap=83.0, iters=60)
    xh_seq = np.array([349.15, 348.65, 348.15, 347.65, 332.15])
    fp_seq = fp.copy()
    warm_iters = []
    cold_iters = []
    warm_cold_f_diff = []
    for step in range(15):
        F_w, dg_w = mpc_warm(xh_seq, fp_seq, al, fnom, 0.012, 85.0)
        mpc_cold = D.fabrica_mpc(hp=30, hc=8, dtc=2.0, t_cap=83.0, iters=60)
        F_c, dg_c = mpc_cold(xh_seq, fp_seq, al, fnom, 0.012, 85.0)
        warm_iters.append(dg_w["it"])
        cold_iters.append(dg_c["it"])
        warm_cold_f_diff.append(float(np.max(np.abs(F_w - F_c))))
        # Avança planta 2 s (4 passos de 0.5 s)
        for _ in range(4):
            d1, _, _ = C.deriv(xh_seq, F_w[0], al, 22.0)
            d2, _, _ = C.deriv(xh_seq + 0.5 * d1, F_w[0], al, 22.0)
            xh_seq = xh_seq + 0.25 * (d1 + d2)
        fp_seq = F_w[0].copy()

    warm_vs_cold = {
        "n_steps": 15,
        "mean_iters_warm": float(np.mean(warm_iters)),
        "mean_iters_cold": float(np.mean(cold_iters)),
        "max_f_diff_warm_vs_cold_GHz": float(np.max(warm_cold_f_diff)),
        "warm_iters_series": warm_iters,
        "cold_iters_series": cold_iters,
    }
    print(f"  [Warm vs Cold] it_warm_medio={warm_vs_cold['mean_iters_warm']:.1f} vs "
          f"it_cold_medio={warm_vs_cold['mean_iters_cold']:.1f} | max_df={warm_vs_cold['max_f_diff_warm_vs_cold_GHz']:.4e} GHz")

    # ------------------------------------------------------------------
    # 3. A07: Condicionamento numérico e sensibilidade a rho_a, rho, lam
    # ------------------------------------------------------------------
    qp_nom = construir_qp_mpc(casos[1]["xh"], fp, al, fnom, casos[1]["slope"], casos[1]["pbud"])
    ref_nom = resolver_qp_referencia_slsqp(qp_nom, fnom)

    sens_rho_a = []
    for ra in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        res_ra = executar_admm_detalhado(qp_nom, fnom, iters=60, rho_a=ra)
        sens_rho_a.append({
            "rho_a": ra,
            "it": res_ra["it"],
            "converged": res_ra["converged"],
            "r_pri": res_ra["r_pri"],
            "r_dua": res_ra["r_dua"],
            "cond_Kmat": res_ra["cond_Kmat"],
            "err_f_linf_vs_ref": float(np.max(np.abs(res_ra["f"] - ref_nom["f"]))),
        })

    sens_rho_spatial = []
    for r_sp in [0.005, 0.02, 0.05, 0.10]:
        qp_sp = construir_qp_mpc(casos[1]["xh"], fp, al, fnom, casos[1]["slope"], casos[1]["pbud"], rho=r_sp)
        ref_sp = resolver_qp_referencia_slsqp(qp_sp, fnom)
        res_sp = executar_admm_detalhado(qp_sp, fnom, iters=60)
        sens_rho_spatial.append({
            "rho_spatial": r_sp,
            "cond_Pm": res_sp["cond_Pm"],
            "it": res_sp["it"],
            "converged": res_sp["converged"],
            "err_f_linf_vs_ref": float(np.max(np.abs(res_sp["f"] - ref_sp["f"]))),
        })

    sens_lam = []
    for lv in [10.0, 50.0, 200.0, 500.0]:
        qp_l = construir_qp_mpc(casos[1]["xh"], fp, al, fnom, casos[1]["slope"], casos[1]["pbud"], lam=lv)
        ref_l = resolver_qp_referencia_slsqp(qp_l, fnom)
        res_l = executar_admm_detalhado(qp_l, fnom, iters=60)
        sens_lam.append({
            "lam": lv,
            "cond_Pm": res_l["cond_Pm"],
            "cond_Kmat": res_l["cond_Kmat"],
            "it": res_l["it"],
            "converged": res_l["converged"],
            "err_f_linf_vs_ref": float(np.max(np.abs(res_l["f"] - ref_l["f"]))),
        })

    condicionamento = {
        "cond_Pm_nominal": float(np.linalg.cond(qp_nom["Pm"])),
        "cond_A_unnormalized": float(np.linalg.cond(qp_nom["A_raw"])),
        "cond_A_row_equilibrated": float(np.linalg.cond(qp_nom["A"])),
        "fator_melhoria_equilibracao_A": float(np.linalg.cond(qp_nom["A_raw"]) / np.linalg.cond(qp_nom["A"])),
        "sensibilidade_rho_a": sens_rho_a,
        "sensibilidade_rho_espacial": sens_rho_spatial,
        "sensibilidade_lambda_slack": sens_lam,
    }
    print(f"  [Condicionamento] kappa(Pm)={condicionamento['cond_Pm_nominal']:.2f} | "
          f"kappa(A_raw)={condicionamento['cond_A_unnormalized']:.2f} -> kappa(A_norm)={condicionamento['cond_A_row_equilibrated']:.2f} "
          f"(melhoria de {condicionamento['fator_melhoria_equilibracao_A']:.1f}x)")

    # ------------------------------------------------------------------
    # 4. A08: Erro de predição linearizada de 30 passos vs Planta Não Linear (Heun)
    # ------------------------------------------------------------------
    regimes_pred = [
        {
            "regime": "R1_nominal_estacionario",
            "xh": np.array([335.15, 334.65, 333.95, 334.15, 318.15]),
            "qext": 18.0,
            "pbud": 95.0,
        },
        {
            "regime": "R2_aquecimento_ativo_dvfs",
            "xh": np.array([351.65, 350.85, 350.15, 349.65, 334.15]),
            "qext": 26.0,
            "pbud": 78.0,
        },
        {
            "regime": "R3_transicao_alta_carga",
            "xh": np.array([354.65, 353.95, 353.15, 352.65, 340.15]),
            "qext": 32.0,
            "pbud": 68.0,
        },
    ]

    csv_rows = []
    resumo_predicao = {}
    dt_plant = 0.5
    dtc = 2.0
    n_sub = int(round(dtc / dt_plant))
    hp = 30
    hc = 8

    for reg in regimes_pred:
        rname = reg["regime"]
        xh0 = reg["xh"].copy()
        qext0 = reg["qext"]
        slope0 = float(C.deriv(xh0, fp, al, qext0)[0][4])
        qp_r = construir_qp_mpc(xh0, fp, al, fnom, slope0, reg["pbud"], hp=hp, hc=hc, dtc=dtc)
        admm_r = executar_admm_detalhado(qp_r, fnom, iters=60)
        F_seq = admm_r["f"].reshape(hc, 4)

        # Predição linearizada nos 30 passos: T_lin(m, i)
        T_lin = (qp_r["fl"] + qp_r["Gam"] @ admm_r["f"]).reshape(hp, 4)

        # Integração não linear de Heun da planta CTM 5x5 nos mesmos 30 passos (60 s)
        x_nl = xh0.copy()
        T_nl = np.zeros((hp, 4))
        for m in range(hp):
            f_m = F_seq[min(m, hc - 1)]
            for _ in range(n_sub):
                d1, _, _ = C.deriv(x_nl, f_m, al, qext0)
                d2, _, _ = C.deriv(x_nl + dt_plant * d1, f_m, al, qext0)
                x_nl = x_nl + 0.5 * dt_plant * (d1 + d2)
            T_nl[m] = x_nl[:4]

        err_mat = T_lin - T_nl  # [hp, 4] em K
        abs_err = np.abs(err_mat)

        for m in range(hp):
            csv_rows.append({
                "regime": rname,
                "step_m": m + 1,
                "horizon_time_s": (m + 1) * dtc,
                "err_die1_K": float(err_mat[m, 0]),
                "err_die2_K": float(err_mat[m, 1]),
                "err_die3_K": float(err_mat[m, 2]),
                "err_die4_K": float(err_mat[m, 3]),
                "abs_err_max_K": float(abs_err[m].max()),
                "abs_err_mean_K": float(abs_err[m].mean()),
            })

        resumo_predicao[rname] = {
            "erro_passo_1_dtc_2s_max_K": float(abs_err[0].max()),
            "erro_passo_5_10s_max_K": float(abs_err[4].max()),
            "erro_passo_15_30s_max_K": float(abs_err[14].max()),
            "erro_passo_30_60s_max_K": float(abs_err[29].max()),
            "mae_por_die_K": [float(abs_err[:, i].mean()) for i in range(4)],
            "max_abs_err_por_die_K": [float(abs_err[:, i].max()) for i in range(4)],
            "rmse_global_horizonte_K": float(np.sqrt(np.mean(err_mat ** 2))),
        }
        print(f"  [Predição {rname}] erro_2s={abs_err[0].max():.4f} K | erro_10s={abs_err[4].max():.4f} K | "
              f"erro_30s={abs_err[14].max():.4f} K | erro_60s={abs_err[29].max():.4f} K | RMSE={resumo_predicao[rname]['rmse_global_horizonte_K']:.4f} K")

    out_dir = os.path.join(ROOT, "audit", "mpc")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "mpc_nonlinear_prediction_error.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    report = {
        "fase": "Fase 2 - Validacao Numerica do MPC/ADMM (A06, A07, A08)",
        "A06_validacao_referencia_qp": relatorio_casos,
        "A06_warm_vs_cold_start": warm_vs_cold,
        "A07_condicionamento_e_sensibilidade": condicionamento,
        "A08_erro_predicao_linear_vs_planta_nao_linear": resumo_predicao,
    }
    json_path = os.path.join(out_dir, "mpc_validation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Relatórios salvos em: {json_path} e {csv_path}")


if __name__ == "__main__":
    main()
