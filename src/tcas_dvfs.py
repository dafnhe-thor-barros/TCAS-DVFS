# -*- coding: utf-8 -*-
"""
tcas_dvfs.py - TCAS-DVFS: Escalonador topologico SDP + LTV-MPC + EKF Joseph + Trava deterministica.

Auditoria Forense e Principios Fisicos (PINN-Oriented):
- C01: dt da simulacao parametrizado dinamicamente em ksub e ks (invariancia temporal de MPC e SDP).
- C02: EKF propagado com Jacobiano analitico exato dP_leak/dT e parametros fisicos par.
- D01: Orcamento radiativo pb derivado do balanco termico real do radiador com Trad = xh[4]
       (Stefan-Boltzmann Q_rad = eps*sigma*A*(Trad^4 - Tspace^4) - qm + Crad*(Trad_max - Trad)/tau),
       eliminando o piso artificial de 15 W e registrando deficit fisico explicitamente.
- D02: Solver ADMM do MPC estilo OSQP com residuo dual exato: r_dua = ||rho_a * A.T @ (z - z_old)||_inf,
       com classificacao formal de convergencia (convergiu, max_iter, subotimo, inviavel_termico).
- D03: Restricoes termicas com slacks individuais por die s in R^N (N=4), desacoplando termicamente
       os nucleos no preditor e eliminando o relaxamento mutuo indevido.
- D04: Documentacao do MPC como linearizacao sucessiva local por passo de controle (frozen trajectory).
- D05/D06: SDP ADMM com residuo de consenso primal/dual e medicao de excesso heuristico vs enumeracao exata.
- D07: Trava de seguranca de retaguarda formalizada como Deterministic Safety Override heuristico.
"""
import itertools
import numpy as np
from tcas_core import (LAP, C_DIE, R_R, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX, T_LIM, DT, NOM,
                       deriv, jacobiano, potencias)


# ---------------------------------------------------------------- SDP (ADMM)
def fabrica_sdp(N=4, iters=120, rho=1.0, n_round=24, seed=0):
    """
    Escalonador por relaxacao semidefinida (SDP) para mapeamento de N tarefas a N dies.
    Minimiza a diafonia termica inter-dies: min x'[(p p') (x) Psi]x - mu*x_prev'x.
    Para N=4, a enumeracao exata das 24 permutacoes serve como ground-truth de validacao offline.
    """
    n = N * N; d = n + 1; E = []; e = []

    def fila(ent, b):
        rw = np.zeros(d * d)
        for (i, j, v) in ent:
            rw[i * d + j] += v
        E.append(rw); e.append(b)

    fila([(0, 0, 1.0)], 1.0)                                       # Z00 = 1
    for i in range(1, d):
        fila([(i, i, 1.0), (0, i, -1.0)], 0.0)                     # Zii = Z0i (binario)
    for i in range(d):
        for j in range(i + 1, d):
            fila([(i, j, 1.0), (j, i, -1.0)], 0.0)                 # Simetria
    Acon = []
    for p in range(N):
        a = np.zeros(n); a[p * N:(p + 1) * N] = 1.0; Acon.append(a)   # Cada tarefa em exatamente um die
    for i in range(N):
        a = np.zeros(n); a[i::N] = 1.0; Acon.append(a)                # Cada die com exatamente uma tarefa
    for a in Acon:
        for l in range(d):
            fila([(1 + j, l, a[j]) for j in range(n) if a[j] != 0] + [(0, l, -1.0)], 0.0)
    E = np.array(E); e = np.array(e); Ep = np.linalg.pinv(E)
    perms = np.array(list(itertools.permutations(range(N)))); Xv = np.zeros((len(perms), n))
    for kq, pm in enumerate(perms):
        for i in range(N):
            Xv[kq, pm[i] * N + i] = 1.0                            # perm[i] = tarefa alocada no die i
    st = {"Z1": None, "Z2": None, "U1": None, "U2": None}; rng = np.random.default_rng(seed)

    def psd(M):
        w_, V_ = np.linalg.eigh(0.5 * (M + M.T)); return (V_ * np.maximum(w_, 0.0)) @ V_.T

    def escolher(Q, modo="sdp", warm=True):
        Q = 0.5 * (Q + Q.T); Jall = np.einsum('ki,ij,kj->k', Xv, Q, Xv); kex = int(np.argmin(Jall))
        dg = {"Jex": float(Jall[kex]), "Jmed": float(Jall.mean()), "it": 0}
        if modo == "exato":                                        # Enumeracao exata para benchmarking
            dg.update({"J": dg["Jex"], "gap_faixa": 0.0, "excess_exato": 0.0, "hit": 1, "converged": True, "r_pri": 0.0, "r_dua": 0.0})
            return perms[kex], dg
        sc_ = max(1e-9, float(np.abs(Q).max())); Cm = np.zeros((d, d)); Cm[1:, 1:] = Q / sc_
        if (not warm) or st["Z1"] is None:
            v0 = np.concatenate([[1.0], np.full(n, 1.0 / N)])
            Z1 = np.outer(v0, v0); Z2 = Z1.copy(); U1 = np.zeros((d, d)); U2 = np.zeros((d, d))
        else:
            Z1, Z2, U1, U2 = st["Z1"], st["Z2"], st["U1"], st["U2"]
        it = 0
        r_pri = 0.0
        r_dua = 0.0
        for it in range(iters):                                    # ADMM: Z1 in PSD, Z2 >= 0, restricoes afins
            Z1_old = Z1.copy(); Z2_old = Z2.copy()
            tg = 0.5 * ((Z1 - U1) + (Z2 - U2)) - Cm / (2.0 * rho); v = tg.reshape(-1)
            Y = (v - Ep @ (E @ v - e)).reshape(d, d)
            Z1 = psd(Y + U1); Z2 = np.maximum(Y + U2, 0.0); U1 = U1 + Y - Z1; U2 = U2 + Y - Z2
            r_pri = float(np.linalg.norm(Y - Z1) + np.linalg.norm(Y - Z2))
            r_dua = float(rho * (np.linalg.norm(Z1 - Z1_old) + np.linalg.norm(Z2 - Z2_old)))
            if it % 20 == 19 and r_pri < 1e-3 and r_dua < 1e-3:
                break
        st["Z1"], st["Z2"], st["U1"], st["U2"] = Z1, Z2, U1, U2
        w_, V_ = np.linalg.eigh(Z1); Vm = V_ * np.sqrt(np.maximum(w_, 0.0))
        cands = [int(np.argmax(Xv @ Z1[1:, 0]))]                   # Projecao da 1a coluna
        for _ in range(n_round):                                   # Arredondamento por hiperplanos aleatorios
            pj = Vm @ rng.standard_normal(d); sg = 1.0 if pj[0] >= 0 else -1.0
            cands.append(int(np.argmax(Xv @ (pj[1:] * sg))))
        kb = min(set(cands), key=lambda kk: Jall[kk])
        # Excesso heuristico normalizado em relacao ao otimo exato
        exc = float((Jall[kb] - Jall[kex]) / max(dg["Jmed"] - dg["Jex"], 1e-9))
        convergiu = bool(r_pri < 1e-3 and r_dua < 1e-3)
        dg.update({"J": float(Jall[kb]), "it": it + 1, "hit": int(kb == kex),
                   "gap_faixa": exc, "excess_exato": exc, "converged": convergiu,
                   "r_pri": r_pri, "r_dua": r_dua})
        return perms[kb], dg
    return escolher


# ---------------------------------------------------------------- EKF (Filtro de Kalman Estendido)
def fabrica_ekf(sigma_s=0.25, par=NOM):
    """
    Filtro de Kalman Estendido na forma de Joseph.
    Propagacao de covariancia com discretizacao de Taylor de 2a ordem e Jacobiano exato dP_leak/dT.
    """
    st = {"x": None, "P": np.diag([1.0, 1.0, 1.0, 1.0, 2.0])}
    Q = np.diag([0.02, 0.02, 0.02, 0.02, 0.05]); R = sigma_s ** 2 * np.eye(5); I5 = np.eye(5)

    def passo(y, prev):
        if st["x"] is None or prev is None:
            st["x"] = y.copy(); return st["x"]
        dt, alp, qp, fp = prev; x = st["x"]
        d = deriv(x, fp, alp, qp, par)[0]
        dF = dt * jacobiano(x, fp, alp, par); Phi = I5 + dF + 0.5 * dF @ dF
        Pp = Phi @ st["P"] @ Phi.T + Q * dt; xp = x + dt * d
        K = Pp @ np.linalg.inv(Pp + R); IK = I5 - K
        st["x"] = xp + K @ (y - xp)
        st["P"] = IK @ Pp @ IK.T + K @ R @ K.T                     # Forma de Joseph (simetrica e definida positiva)
        return st["x"]
    return passo


# ---------------------------------------------------------------- MPC (Controle Preditivo)
def fabrica_mpc(hp=30, hc=8, dtc=2.0, t_cap=83.0, w=1.0, Rw=0.2, rho=0.02, lam=200.0, iters=60, par=None):
    """
    Controlador Preditivo Linearizado a cada passo de controle (Successive Local Linearization) via ADMM (OSQP-like).
    Slacks termicos individuais por die s in R^N (N=4) para desacoplamento espacial das restricoes termicas.
    Residuo dual rigoroso do ADMM com projecao via A.T: r_dua = ||rho_a * A.T @ (z - z_old)||_inf.
    """
    par = NOM if par is None else par
    N = 4; nf = hc * N; D_ = np.eye(nf) - np.eye(nf, k=-N)
    ar = np.arange(hp); Jt = np.maximum(0, ar[:, None] - ar[None, :]).astype(float)
    st = {"x": None, "y": None}; c = float(par["C"]); RR = float(par["RR"]); Lp = np.asarray(par["LAP"], dtype=float)

    def ptot(T, f, al):
        pd, pl = potencias(T, f, al); return pd + pl

    def resolver(xh, fp, al, fnom, slope, pbud):
        T0_ = xh[:4]; Trad = xh[4]; P0 = ptot(T0_, fp, al); dd = 1e-3
        a = (ptot(T0_, fp + dd, al) - ptot(T0_, fp - dd, al)) / (2 * dd)        # dP/df
        b = ptot(T0_ + 0.5, fp, al) - ptot(T0_ - 0.5, fp, al)                   # dP/dT
        lm, V = np.linalg.eigh((-Lp - np.eye(N) / RR + np.diag(b)) / c)
        g = (np.exp(lm * dtc) - 1.0) / lm / c; ms = np.arange(1, hp + 1)
        Apow = np.einsum('ik,mk,jk->mij', V, np.exp(np.outer(ms, lm) * dtc), V)
        W = np.einsum('ik,sk,jk->sij', V, g[None, :] * np.exp(np.outer(ar, lm) * dtc), V)
        Cs = np.cumsum(W, axis=0)
        dv = P0 - a * fp - b * T0_ + Trad / RR
        free = (np.einsum('mij,j->mi', Apow, T0_) + np.einsum('mij,j->mi', Cs, dv)
                + (slope * dtc / RR) * (Jt @ W.sum(2)))                         # Resposta livre do radiador
        G0 = np.zeros((hp, N, hc, N))
        for j in range(min(hc - 1, hp)):
            G0[j:, :, j, :] = W[:hp - j]
        if hp >= hc:
            G0[hc - 1:, :, hc - 1, :] = Cs[:hp - hc + 1]                        # Ultimo controle mantido
        Gam = G0.reshape(hp * N, nf) * np.tile(a, hc)[None, :]
        Gh = Gam.reshape(hp, N, nf); Gh = (Gh - Gh.mean(1, keepdims=True)).reshape(hp * N, nf)   # Penalidade de gradiente espacial
        fc = (free - free.mean(1, keepdims=True)).reshape(-1)

        # Decisao: x = [f_1..f_nf, s_1..s_N]^T in R^(nf + N)
        # Slacks termicos individuais desacoplados por die
        n_vars = nf + N
        Pm = np.zeros((n_vars, n_vars))
        Pm[:nf, :nf] = 2 * w * np.eye(nf) + 2 * Rw * (D_.T @ D_) + 2 * rho * (Gh.T @ Gh)
        Pm[nf:, nf:] = 2 * lam * np.eye(N)                                     # Penalidade individual dos slacks
        
        e0 = np.zeros(nf); e0[:N] = fp
        q = np.zeros(n_vars)
        q[:nf] = -2 * w * fnom.reshape(-1) - 2 * Rw * (D_.T @ e0) + 2 * rho * (Gh.T @ fc)

        Tc = t_cap + 273.15; fl = free.reshape(-1)
        # Matriz de slacks: cada passo de predicao associa o die i ao seu respectivo slack s_i
        S_block = np.tile(np.eye(N), (hp, 1))
        A1 = np.hstack([Gam, -S_block])                                         # Gam * f - s_i <= Tc - fl
        A2 = np.hstack([np.eye(nf), np.zeros((nf, N))])                        # Limites de frequencia
        A3 = np.hstack([np.zeros((N, nf)), np.eye(N)])                         # Limites dos slacks (s_i >= 0)
        A4 = np.zeros((hc, n_vars))
        for j in range(hc):
            A4[j, j * N:(j + 1) * N] = a                                       # Orcamento de potencia por passo de controle
        
        A = np.vstack([A1, A2, A3, A4])
        lo = np.concatenate([np.full(hp * N, -1e9), np.full(nf, F_MIN), np.zeros(N), np.full(hc, -1e9)])
        up = np.concatenate([Tc - fl, np.full(nf, F_MAX), np.full(N, 1e3), np.full(hc, pbud - (P0 - a * fp).sum())])
        
        rn = np.abs(A).max(1); rn[rn < 1e-9] = 1.0
        A = A / rn[:, None]; lo = lo / rn; up = up / rn

        if st["x"] is None:
            x = np.concatenate([fnom.reshape(-1), np.zeros(N)])
            y = np.zeros(A.shape[0])
        else:
            x = np.zeros(n_vars)
            x[:nf - N] = st["x"][N:nf]
            x[nf - N:nf] = st["x"][nf - N:nf]
            x[nf:] = 0.0                                                        # Reset dos slacks
            y = st["y"]

        x[:nf] = np.clip(x[:nf], F_MIN, F_MAX)
        x[nf:] = np.maximum(0.0, x[nf:])
        z = np.clip(A @ x, lo, up)

        rho_a = 1.0; sg = 1e-6; al_ = 1.6
        Kinv = np.linalg.inv(Pm + sg * np.eye(n_vars) + rho_a * (A.T @ A))
        it = 0; r_pri = 0.0; r_dua = 0.0
        for it in range(iters):                                                  # ADMM estilo OSQP
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
            # Residuo dual exato projetado no espaco primal via A.T (Boyd et al., Stellato et al.)
            r_dua = float(np.abs(rho_a * (A.T @ (z - z_old))).max())
            
            # Criterio de convergencia primal e dual simultaneos
            if it >= 15 and it % 5 == 4 and r_pri < 2e-3 and r_dua < 5e-3:
                break

        st["x"] = x; st["y"] = y
        Ff = np.clip(x[:nf], F_MIN, F_MAX)
        viol_termica = float(max(0.0, (fl + Gam @ Ff - Tc).max()))
        slack_max = float(np.max(x[nf:]))
        convergiu = bool(r_pri < 2e-3 and r_dua < 5e-3)
        
        status = "convergiu" if convergiu else ("max_iter" if it >= iters - 1 else "subotimo")
        if slack_max > 5.0:
            status = "inviavel_termico"

        return Ff.reshape(hc, N), {
            "it": it + 1,
            "viol": viol_termica,
            "slack_max": slack_max,
            "converged": convergiu,
            "status": status,
            "r_pri": r_pri,
            "r_dua": r_dua
        }
    return resolver


# ---------------------------------------------------------------- Governador TCAS-DVFS
def gov_tcas_dvfs(usa_sched=True, usa_mpc=True, hp=30, hc=8, ksub=None, dt=DT, sched_s=120.0,
                  mpc_period_s=2.0, t_cap=83.0, t_bar=84.5, mu_mig=1.0, sigma_s=0.25, modo="sdp", par=NOM):
    """
    Governador preditivo hierarquico TCAS-DVFS.
    Parametrizacao invariante de tempo:
      dt = passo da planta fisica [s]
      sched_s = periodo do escalonador SDP [s] (nominal: 120 s)
      mpc_period_s = periodo de atuacao do MPC [s] (nominal: 2 s)
    """
    N = 4
    if ksub is None:
        ksub = max(1, int(round(mpc_period_s / dt)))
    ks = max(1, int(round(sched_s / dt)))
    dt_mpc = ksub * dt

    sdp = fabrica_sdp(N)
    mpc = fabrica_mpc(hp, hc, dt_mpc, t_cap, par=par)
    ekf = fabrica_ekf(sigma_s, par=par)

    Phi = np.linalg.inv(par["LAP"] + np.eye(N) / par["RR"])
    Hc_ = np.eye(N) - np.ones((N, N)) / N
    Psi = Phi @ Hc_ @ Phi

    s = {"perm": np.arange(N), "f": None, "fcmd": None}
    info = {"perm": np.arange(N), "n_mpc": 0, "qp_it": 0, "viol_max": 0.0, "viol_sum": 0.0,
            "n_barreira": 0, "n_sched": 0, "n_inviavel_qp": 0, "n_deficit_orcamento": 0,
            "deficit_max_w": 0.0, "gap": [], "hit": []}

    def acao(k, y, sc):
        cur_dt = sc["dt"]
        prev = None if s["f"] is None else (cur_dt, sc["al"][k - 1][s["perm"]], sc["qext"][k - 1], s["f"])
        xh = ekf(y, prev)

        if usa_sched and k % ks == 0:                              # Escalonador topologico SDP
            ke = min(sc["nt"] - 1, k + ks); phat = np.zeros(N)
            for kk in (k, (k + ke) // 2, ke):
                pd, pl = potencias(np.full(N, 353.15), sc["fn"][kk], sc["al"][kk])
                phat += (pd + pl) / 3.0
            Xprev = np.zeros((N, N)); Xprev[s["perm"], np.arange(N)] = 1.0
            Q = np.kron(np.outer(phat, phat), Psi) + np.diag(-mu_mig * Xprev.reshape(-1))
            pm, dg = sdp(Q, modo)
            s["perm"] = pm; info["perm"] = pm; info["n_sched"] += 1
            info["gap"].append(dg["excess_exato"]); info["hit"].append(dg["hit"])

        perm = s["perm"]; fnk = sc["fn"][k][perm]

        if usa_mpc and k % ksub == 0:                              # Atuacao preditiva MPC
            al = sc["al"][k][perm]; fp = sc["fn"][0][perm] if s["f"] is None else s["f"]
            idx = np.minimum(sc["nt"] - 1, k + np.arange(hc) * ksub); fnom = sc["fn"][idx][:, perm]
            slope = deriv(xh, fp, al, sc["qext"][k], par)[0][4]
            np_h = max(1, int(round(240.0 / cur_dt)))
            ke = min(sc["nt"], k + np_h)
            # sc["cs"] e a integral cumulativa prefix-sum de qext [W]
            qm = (sc["cs"][ke] - sc["cs"][k]) / max(1, ke - k)

            # Balanco termico fisico do radiador baseado no estado estimado Trad = xh[4]
            Trad_est = xh[4]                                      # Temperatura do radiador em Kelvin [K]
            Tcap_K = t_cap + 273.15                               # Limite de temperatura dos dies em Kelvin [K]
            
            # 1. Capacidade de emissao radiativa do radiador a Trad_est (Stefan-Boltzmann)
            Q_rad_emit = EPS * SIGMA * A_RAD * (Trad_est ** 4 - T_SPACE ** 4)
            # Rejeicao radiativa liquida (descontando o fluxo ambiental absorvido qm)
            Q_rad_liq = Q_rad_emit - qm
            
            # 2. Capacidade transiente de absorcao do radiador ate limite seguro Trad_max
            Trad_max = Tcap_K - 5.0                               # Margem condutiva minima die->radiador [K]
            tau_rad = float(par["CRAD"]) * float(par["RR"]) / 4.0 # Constante de tempo termica caracteristica
            P_buffer = float(par["CRAD"]) * max(0.0, Trad_max - Trad_est) / max(60.0, tau_rad)
            
            # 3. Limite de conducao termica admissivel dies -> radiador sem violar Tcap
            P_cond_max = 4.0 * max(0.0, Tcap_K - Trad_est) / float(par["RR"])
            
            # Orcamento fisico admissivel: limitado pela conducao e pela rejeicao liquida com buffer
            P_budget_fisico = min(P_cond_max, Q_rad_liq + P_buffer)
            
            # Potencia minima dos dies na frequencia minima F_MIN
            pd_min, pl_min = potencias(xh[:4], np.full(N, F_MIN), al)
            P_min_cores = float((pd_min + pl_min).sum())
            
            if P_budget_fisico < P_min_cores:
                # Deficit fisico de capacidade de refrigeracao registrado sem mascaramento artificial
                info["n_deficit_orcamento"] += 1
                info["deficit_max_w"] = max(info.get("deficit_max_w", 0.0), P_min_cores - P_budget_fisico)
                pb = P_min_cores                                  # Passa piso numerico ao QP para evitar inviabilidade trivial
            else:
                pb = P_budget_fisico

            F, dg = mpc(xh, fp, al, fnom, slope, pb); s["fcmd"] = F[0]
            info["n_mpc"] += 1; info["qp_it"] += dg["it"]
            info["viol_max"] = max(info["viol_max"], dg["viol"]); info["viol_sum"] += dg["viol"]
            if not dg.get("converged", True):
                info["n_inviavel_qp"] += 1

        if not usa_mpc:
            s["fcmd"] = fnk
        f = s["fcmd"] if s["fcmd"] is not None else fnk

        # Trava deterministica de seguranca de retaguarda (Deterministic Safety Override)
        T = xh[:4] - 273.15; m = T >= t_bar
        if m.any():
            cap = fnk * np.maximum(0.15, (T_LIM - T) / (T_LIM - 75.0))
            f = np.where(m, np.maximum(F_MIN, np.minimum(f, cap)), f); info["n_barreira"] += 1

        s["f"] = f
        return f
    return acao, info
