# -*- coding: utf-8 -*-
"""
tcas_dvfs.py - TCAS-DVFS: escalonador por relaxacao SDP (ADMM, numpy) + MPC + EKF + barreira de retaguarda.

  SDP : atribuicao de N tarefas a N dies,  min x'[(p p') (x) Psi]x - mu*x_prev'x ,
        Psi = Phi H Phi, Phi = (L + I/R_r)^-1, H = I - 11'/N. Relaxacao de Shor sobre [1 x']'[1 x']
        e arredondamento por hiperplanos (24 sorteios + projecao da 1a coluna). SEM limite inferior certificado.
  MPC : CTM linearizada em torno do estado atual (autovalores exatos, passo dtc = ksub*DT), Hp passos,
        Hc de controle, restricao suave T <= t_cap, orcamento de potencia; QP por ADMM (estilo OSQP).
"""
import itertools
import numpy as np
from tcas_core import (LAP, C_DIE, R_R, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX, T_LIM, DT, NOM,
                       deriv, jacobiano, potencias)


# ---------------------------------------------------------------- SDP (ADMM)
def fabrica_sdp(N=4, iters=120, rho=1.0, n_round=24, seed=0):
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
            fila([(i, j, 1.0), (j, i, -1.0)], 0.0)                 # simetria
    Acon = []
    for p in range(N):
        a = np.zeros(n); a[p * N:(p + 1) * N] = 1.0; Acon.append(a)   # cada tarefa em um die
    for i in range(N):
        a = np.zeros(n); a[i::N] = 1.0; Acon.append(a)                # cada die com uma tarefa
    for a in Acon:
        for l in range(d):
            fila([(1 + j, l, a[j]) for j in range(n) if a[j] != 0] + [(0, l, -1.0)], 0.0)
    E = np.array(E); e = np.array(e); Ep = np.linalg.pinv(E)
    perms = np.array(list(itertools.permutations(range(N)))); Xv = np.zeros((len(perms), n))
    for kq, pm in enumerate(perms):
        for i in range(N):
            Xv[kq, pm[i] * N + i] = 1.0                            # perm[i] = tarefa no die i
    st = {"Z1": None, "Z2": None, "U1": None, "U2": None}; rng = np.random.default_rng(seed)

    def psd(M):
        w_, V_ = np.linalg.eigh(0.5 * (M + M.T)); return (V_ * np.maximum(w_, 0.0)) @ V_.T

    def escolher(Q, modo="sdp", warm=True):
        Q = 0.5 * (Q + Q.T); Jall = np.einsum('ki,ij,kj->k', Xv, Q, Xv); kex = int(np.argmin(Jall))
        dg = {"Jex": float(Jall[kex]), "Jmed": float(Jall.mean()), "it": 0}
        if modo == "exato":                                        # enumeracao exata (so para N pequeno)
            dg.update({"J": dg["Jex"], "gap_faixa": 0.0, "hit": 1}); return perms[kex], dg
        sc_ = max(1e-9, float(np.abs(Q).max())); Cm = np.zeros((d, d)); Cm[1:, 1:] = Q / sc_
        if (not warm) or st["Z1"] is None:
            v0 = np.concatenate([[1.0], np.full(n, 1.0 / N)])
            Z1 = np.outer(v0, v0); Z2 = Z1.copy(); U1 = np.zeros((d, d)); U2 = np.zeros((d, d))
        else:
            Z1, Z2, U1, U2 = st["Z1"], st["Z2"], st["U1"], st["U2"]
        it = 0
        for it in range(iters):                                    # ADMM: Z1 PSD, Z2 >= 0, restricoes afins
            tg = 0.5 * ((Z1 - U1) + (Z2 - U2)) - Cm / (2.0 * rho); v = tg.reshape(-1)
            Y = (v - Ep @ (E @ v - e)).reshape(d, d)
            Z1 = psd(Y + U1); Z2 = np.maximum(Y + U2, 0.0); U1 = U1 + Y - Z1; U2 = U2 + Y - Z2
            if it % 20 == 19 and np.linalg.norm(Y - Z1) + np.linalg.norm(Y - Z2) < 1e-3:
                break
        st["Z1"], st["Z2"], st["U1"], st["U2"] = Z1, Z2, U1, U2
        w_, V_ = np.linalg.eigh(Z1); Vm = V_ * np.sqrt(np.maximum(w_, 0.0))
        cands = [int(np.argmax(Xv @ Z1[1:, 0]))]                   # projecao da 1a coluna
        for _ in range(n_round):                                   # arredondamento por hiperplano
            pj = Vm @ rng.standard_normal(d); sg = 1.0 if pj[0] >= 0 else -1.0
            cands.append(int(np.argmax(Xv @ (pj[1:] * sg))))
        kb = min(set(cands), key=lambda kk: Jall[kk])
        dg.update({"J": float(Jall[kb]), "it": it + 1, "hit": int(kb == kex),
                   "gap_faixa": float((Jall[kb] - Jall[kex]) / max(dg["Jmed"] - dg["Jex"], 1e-9))})
        return perms[kb], dg                                       # NAO ha limite inferior certificado
    return escolher


# ---------------------------------------------------------------- EKF
def fabrica_ekf(sigma_s=0.25):
    st = {"x": None, "P": np.diag([1.0, 1.0, 1.0, 1.0, 2.0])}
    Q = np.diag([0.02, 0.02, 0.02, 0.02, 0.05]); R = sigma_s ** 2 * np.eye(5); I5 = np.eye(5)

    def passo(y, prev):
        if st["x"] is None or prev is None:
            st["x"] = y.copy(); return st["x"]
        dt, alp, qp, fp = prev; x = st["x"]
        d = deriv(x, fp, alp, qp)[0]                               # propaga pelo MODELO nominal
        dF = dt * jacobiano(x); Phi = I5 + dF + 0.5 * dF @ dF
        Pp = Phi @ st["P"] @ Phi.T + Q * dt; xp = x + dt * d
        K = Pp @ np.linalg.inv(Pp + R); IK = I5 - K
        st["x"] = xp + K @ (y - xp)
        st["P"] = IK @ Pp @ IK.T + K @ R @ K.T                     # forma de Joseph
        return st["x"]
    return passo


# ---------------------------------------------------------------- MPC
def fabrica_mpc(hp=30, hc=8, dtc=2.0, t_cap=83.0, w=1.0, Rw=0.2, rho=0.02, lam=200.0, iters=60, par=None):
    par = NOM if par is None else par
    N = 4; nf = hc * N; D_ = np.eye(nf) - np.eye(nf, k=-N)
    ar = np.arange(hp); Jt = np.maximum(0, ar[:, None] - ar[None, :]).astype(float)
    st = {"x": None, "y": None}; c = par["C"]; RR = par["RR"]; Lp = par["LAP"]

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
                + (slope * dtc / RR) * (Jt @ W.sum(2)))                         # resposta livre (radiador em rampa)
        G0 = np.zeros((hp, N, hc, N))
        for j in range(min(hc - 1, hp)):
            G0[j:, :, j, :] = W[:hp - j]
        if hp >= hc:
            G0[hc - 1:, :, hc - 1, :] = Cs[:hp - hc + 1]                        # ultimo controle mantido
        Gam = G0.reshape(hp * N, nf) * np.tile(a, hc)[None, :]
        Gh = Gam.reshape(hp, N, nf); Gh = (Gh - Gh.mean(1, keepdims=True)).reshape(hp * N, nf)   # gradiente (H T)
        fc = (free - free.mean(1, keepdims=True)).reshape(-1)
        Pm = np.zeros((nf + 1, nf + 1))
        Pm[:nf, :nf] = 2 * w * np.eye(nf) + 2 * Rw * (D_.T @ D_) + 2 * rho * (Gh.T @ Gh); Pm[nf, nf] = 2 * lam
        e0 = np.zeros(nf); e0[:N] = fp
        q = np.zeros(nf + 1); q[:nf] = -2 * w * fnom.reshape(-1) - 2 * Rw * (D_.T @ e0) + 2 * rho * (Gh.T @ fc)
        Tc = t_cap + 273.15; fl = free.reshape(-1)
        A1 = np.hstack([Gam, -np.ones((hp * N, 1))]); A2 = np.hstack([np.eye(nf), np.zeros((nf, 1))])
        A3 = np.zeros((1, nf + 1)); A3[0, nf] = 1.0; A4 = np.zeros((hc, nf + 1))
        for j in range(hc):
            A4[j, j * N:(j + 1) * N] = a
        A = np.vstack([A1, A2, A3, A4])
        lo = np.concatenate([np.full(hp * N, -1e9), np.full(nf, F_MIN), [0.0], np.full(hc, -1e9)])
        up = np.concatenate([Tc - fl, np.full(nf, F_MAX), [1e3], np.full(hc, pbud - (P0 - a * fp).sum())])
        rn = np.abs(A).max(1); rn[rn < 1e-9] = 1.0; A = A / rn[:, None]; lo = lo / rn; up = up / rn
        if st["x"] is None:
            x = np.concatenate([fnom.reshape(-1), [0.0]]); y = np.zeros(A.shape[0])
        else:
            x = st["x"].copy(); x[:nf - N] = st["x"][N:nf]; y = st["y"]         # warm start deslocado
        x[:nf] = np.clip(x[:nf], F_MIN, F_MAX); z = np.clip(A @ x, lo, up)
        rho_a = 1.0; sg = 1e-6; al_ = 1.6
        Kinv = np.linalg.inv(Pm + sg * np.eye(nf + 1) + rho_a * (A.T @ A)); it = 0
        for it in range(iters):                                                  # ADMM estilo OSQP
            xt = Kinv @ (sg * x - q + A.T @ (rho_a * z - y)); zt = A @ xt
            xr = al_ * xt + (1 - al_) * x; zr = al_ * zt + (1 - al_) * z
            zn = np.clip(zr + y / rho_a, lo, up); y = y + rho_a * (zr - zn); x = xr; z = zn
            if it >= 19 and it % 10 == 9 and np.abs(A @ x - z).max() < 2e-3:
                break
        st["x"] = x; st["y"] = y; Ff = np.clip(x[:nf], F_MIN, F_MAX)
        return Ff.reshape(hc, N), {"it": it + 1, "viol": float(max(0.0, (fl + Gam @ Ff - Tc).max()))}
    return resolver


# ---------------------------------------------------------------- governador
def gov_tcas_dvfs(usa_sched=True, usa_mpc=True, hp=30, hc=8, ksub=4, sched_s=120.0, t_cap=83.0, t_bar=84.5,
                  mu_mig=1.0, sigma_s=0.25, modo="sdp"):
    N = 4; sdp = fabrica_sdp(N); mpc = fabrica_mpc(hp, hc, ksub * DT, t_cap); ekf = fabrica_ekf(sigma_s)
    Phi = np.linalg.inv(LAP + np.eye(N) / R_R); Hc_ = np.eye(N) - np.ones((N, N)) / N; Psi = Phi @ Hc_ @ Phi
    ks = int(round(sched_s / DT)); s = {"perm": np.arange(N), "f": None, "fcmd": None}
    info = {"perm": np.arange(N), "n_mpc": 0, "qp_it": 0, "viol_max": 0.0, "viol_sum": 0.0,
            "n_barreira": 0, "n_sched": 0, "gap": [], "hit": []}

    def acao(k, y, sc):
        dt = sc["dt"]
        prev = None if s["f"] is None else (dt, sc["al"][k - 1][s["perm"]], sc["qext"][k - 1], s["f"])
        xh = ekf(y, prev)
        if usa_sched and k % ks == 0:                              # escalonador SDP
            ke = min(sc["nt"] - 1, k + ks); phat = np.zeros(N)
            for kk in (k, (k + ke) // 2, ke):
                pd, pl = potencias(np.full(N, 353.15), sc["fn"][kk], sc["al"][kk]); phat += (pd + pl) / 3.0
            Xprev = np.zeros((N, N)); Xprev[s["perm"], np.arange(N)] = 1.0
            Q = np.kron(np.outer(phat, phat), Psi) + np.diag(-mu_mig * Xprev.reshape(-1))
            pm, dg = sdp(Q, modo); s["perm"] = pm; info["perm"] = pm; info["n_sched"] += 1
            info["gap"].append(dg["gap_faixa"]); info["hit"].append(dg["hit"])
        perm = s["perm"]; fnk = sc["fn"][k][perm]
        if usa_mpc and k % ksub == 0:                              # MPC a cada ksub*DT
            al = sc["al"][k][perm]; fp = sc["fn"][0][perm] if s["f"] is None else s["f"]
            idx = np.minimum(sc["nt"] - 1, k + np.arange(hc) * ksub); fnom = sc["fn"][idx][:, perm]
            slope = deriv(xh, fp, al, sc["qext"][k])[0][4]
            ke = min(sc["nt"], k + 480); qm = (sc["cs"][ke] - sc["cs"][k]) / max(1, ke - k)
            pb = max(15.0, EPS * SIGMA * A_RAD * ((83.0 + 273.15) ** 4 - T_SPACE ** 4) - qm)
            F, dg = mpc(xh, fp, al, fnom, slope, pb); s["fcmd"] = F[0]
            info["n_mpc"] += 1; info["qp_it"] += dg["it"]
            info["viol_max"] = max(info["viol_max"], dg["viol"]); info["viol_sum"] += dg["viol"]
        if not usa_mpc:
            s["fcmd"] = fnk
        f = s["fcmd"] if s["fcmd"] is not None else fnk
        T = xh[:4] - 273.15; m = T >= t_bar                        # barreira de retaguarda
        if m.any():
            cap = fnk * np.maximum(0.15, (T_LIM - T) / (T_LIM - 75.0))
            f = np.where(m, np.maximum(F_MIN, np.minimum(f, cap)), f); info["n_barreira"] += 1
        s["f"] = f
        return f
    return acao, info
