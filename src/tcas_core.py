# -*- coding: utf-8 -*-
"""
tcas_core.py - Modelo, governadores baseline, metricas e confiabilidade.
Python >= 3.9, numpy. Estados em [K]; metricas em [degC]; f em [GHz]; potencia em [W]; energia em [kJ].
"""
import math
import numpy as np

# ------------------------------------------------------------------ constantes
DT = 0.5
MU_E, R_E, H_ORB = 398600.4418, 6371.0, 500.0
S_SOLAR, A_E, SIGMA, T_SPACE = 1361.0, 0.30, 5.670374e-8, 2.73
A_RAD, ALPHA_S, EPS, C_RAD, F_VIEW = 0.12, 0.12, 0.88, 1.45 * 896.0, 0.42
C_DIE, R_R = 18.5, 2.10
V_MIN, V_MAX, F_MIN, F_MAX = 0.65, 1.05, 0.30, 1.30
BETA = (V_MAX - V_MIN) / (F_MAX - F_MIN)
C_EFF, P_LEAK_REF, V0, T0, EA, KB = 24.5e-9, 0.80, 0.85, 300.0, 0.20, 8.617333262e-5
FPS_NOM = np.array([45.0, 60.0, 30.0, 80.0])      # SAR-YOLO, Cloud-Seg, Hyper-ViT, telemetria
T_LIM, NP_H = 85.0, 480
RES = {(0, 1): 1.45, (2, 3): 1.45, (0, 2): 1.52, (1, 3): 1.52, (0, 3): 3.20, (1, 2): 3.20}
GM = np.zeros((4, 4))
for _par, _r in RES.items():
    GM[_par[0], _par[1]] = 1.0 / _r
    GM[_par[1], _par[0]] = 1.0 / _r
LAP = np.diag(GM.sum(1)) - GM                       # Laplaciana condutiva G_crosstalk
S_GRID = np.linspace(1.0, 0.2, 33)
NOM = {"LAP": LAP, "C": C_DIE, "RR": R_R, "CRAD": C_RAD}


def parametros(sRij=1.0, sRr=1.0, sC=1.0, sCr=1.0):
    """Fatores multiplicativos sobre R_ij, R_r, c_die e C_rad da PLANTA."""
    return {"LAP": LAP / sRij, "C": C_DIE * sC, "RR": R_R * sRr, "CRAD": C_RAD * sCr}


# ------------------------------------------------------------------ cenario
def cenario(n_orb=3, dt=DT):
    r_orb = R_E + H_ORB
    t_orb = 2 * math.pi * math.sqrt(r_orb ** 3 / MU_E)
    nt = int(math.floor(n_orb * t_orb / dt + 1e-9)) + 1
    t = np.arange(nt) * dt
    fsol = 1 - math.asin(R_E / r_orb) / math.pi     # fracao iluminada (beta = 0)
    md = fsol / 2
    fase = np.mod(t, t_orb) / t_orb
    ill = 1 / (1 + np.exp(np.clip((fase - fsol) * 120, -700, 700)))
    cs = np.maximum(0.0, np.cos((math.pi / 2) * (fase - md) / md) * math.cos(math.radians(30)) * math.cos(math.radians(15)))
    te = 255.0 - 15.0 * np.sin(math.pi * (fase - md) / fsol) ** 2
    qext = ill * S_SOLAR * ALPHA_S * A_RAD * cs * (1 + A_E * F_VIEW) + SIGMA * te ** 4 * EPS * A_RAD * F_VIEW
    al = np.zeros((nt, 4)); fn = np.zeros((nt, 4))
    al[:, 0] = 0.85 + 0.10 * np.sin(2 * np.pi * t / 1200); fn[:, 0] = 1.15
    al[:, 1] = 0.70 + 0.20 * np.minimum(1.0, t / (t[-1] * 0.7)); fn[:, 1] = 1.05
    alto = np.mod(t, 1800) > 900
    al[:, 2] = np.where(alto, 0.95, 0.60); fn[:, 2] = np.where(alto, 1.30, 0.80)
    al[:, 3] = 0.75; fn[:, 3] = 1.00
    return {"dt": dt, "nt": nt, "t": t, "t_orb": t_orb, "qext": qext, "al": al, "fn": fn,
            "x0": np.array([310.0, 310.0, 310.0, 310.0, 270.0]),
            "cs": np.concatenate([[0.0], np.cumsum(qext)])}


# ------------------------------------------------------------------ planta (CTM 5x5)
def tensao(f):
    return np.minimum(V_MAX, np.maximum(V_MIN, V_MIN + BETA * (f - F_MIN)))


def potencias(T, f, al):
    V = tensao(f)
    pd = al * C_EFF * V * V * f * 1e9
    Tc = np.minimum(500.0, np.maximum(200.0, T))
    pl = P_LEAK_REF * (V / V0) ** 1.5 * (Tc / T0) ** 2 * np.exp((EA / KB) * (1.0 / T0 - 1.0 / Tc))
    return pd, pl


def deriv(X, f, al, qext, par=NOM):
    T = X[:4]; tr = X[4]
    pd, pl = potencias(T, f, al)
    p = pd + pl
    qr = (T - tr) / par["RR"]
    d = np.empty(5)
    d[:4] = (p - par["LAP"] @ T - qr) / par["C"]
    d[4] = (qr.sum() + qext - EPS * SIGMA * A_RAD * (tr ** 4 - T_SPACE ** 4)) / par["CRAD"]
    return d, p, pl


def escala_orcamento(fc, T, al, p_max):
    """Maior fator comum s em S_GRID tal que P_total(T, s*fc) <= p_max."""
    F = np.minimum(F_MAX, np.maximum(F_MIN, fc[None, :] * S_GRID[:, None]))
    pd, pl = potencias(T, F, al)
    ok = np.nonzero((pd + pl).sum(1) <= p_max)[0]
    if len(ok) == 0:
        return F[-1], 1
    return F[ok[0]], int(ok[0] > 0)


def jacobiano(x):
    F = np.zeros((5, 5))
    F[:4, :4] = -LAP / C_DIE - np.eye(4) / (R_R * C_DIE)
    F[:4, 4] = 1.0 / (R_R * C_DIE)
    F[4, :4] = 1.0 / (C_RAD * R_R)
    F[4, 4] = -4.0 / (R_R * C_RAD) - 4.0 * EPS * SIGMA * A_RAD * x[4] ** 3 / C_RAD
    return F


# ------------------------------------------------------------------ governadores baseline
# Cada governador devolve (acao, info); acao(k, y_medido, sc) -> vetor f [GHz].
def gov_ondemand(trip=None, rel=None):
    s = {"est": np.zeros(4, bool)}
    def acao(k, y, sc):
        f = F_MIN + sc["al"][k] * (F_MAX - F_MIN)
        if trip is not None:
            T = y[:4] - 273.15
            s["est"] = np.where(T >= trip, True, np.where(T <= rel, False, s["est"]))
            f = np.where(s["est"], F_MIN, f)
        return f
    return acao, {}


def gov_throttling(trip=85.0, rel=75.0):
    s = {"est": np.zeros(4, bool)}
    def acao(k, y, sc):
        T = y[:4] - 273.15
        s["est"] = np.where(T >= trip, True, np.where(T <= rel, False, s["est"]))
        return np.where(s["est"], F_MIN, sc["fn"][k])
    return acao, {}


def gov_pid(kp, ki, kd, alvo=80.0):
    s = {"ie": np.zeros(4), "pe": np.zeros(4)}
    def acao(k, y, sc):
        e = y[:4] - 273.15 - alvo
        s["ie"] = np.clip(s["ie"] + e * sc["dt"], -300, 300)
        de = (e - s["pe"]) / sc["dt"]; s["pe"] = e
        return np.clip(sc["fn"][k] + kp * e + ki * s["ie"] + kd * de, F_MIN, F_MAX)
    return acao, {}


def gov_power_cap(p_cap=65.0):
    info = {"n_atuou": 0}
    def acao(k, y, sc):
        f, a = escala_orcamento(sc["fn"][k], y[:4], sc["al"][k], p_cap)
        info["n_atuou"] += a
        return f
    return acao, info


def gov_tca(lam=0.35, t_seg=83.0, t_ref=75.0, np_h=NP_H, usa_lap=True, usa_barreira=True, sigma_s=0.25):
    """TCA-DVFS simples (versao anterior): EKF + lei laplaciana + orcamento + barreira. Sem SDP nem MPC."""
    s = {"xh": None, "P": np.diag([1.0, 1.0, 1.0, 1.0, 2.0]), "f": None}
    info = {"n": 0, "n_barreira": 0, "n_orcamento": 0}
    Q = np.diag([0.02, 0.02, 0.02, 0.02, 0.05]); R = sigma_s ** 2 * np.eye(5); I5 = np.eye(5)

    def acao(k, y, sc):
        dt = sc["dt"]
        if s["xh"] is None:
            s["xh"] = y.copy()
        else:
            xh = s["xh"]
            d = deriv(xh, s["f"], sc["al"][k - 1], sc["qext"][k - 1])[0]
            dF = dt * jacobiano(xh); Phi = I5 + dF + 0.5 * dF @ dF
            Pp = Phi @ s["P"] @ Phi.T + Q * dt
            xp = xh + dt * d
            K = Pp @ np.linalg.inv(Pp + R); IK = I5 - K
            s["xh"] = xp + K @ (y - xp)
            s["P"] = IK @ Pp @ IK.T + K @ R @ K.T
        xh = s["xh"]; T = xh[:4] - 273.15; fn = sc["fn"][k]
        f = fn - 0.5 * lam * (LAP @ T) if usa_lap else fn.copy()
        f = np.minimum(F_MAX, np.maximum(F_MIN, f))
        k_end = min(sc["nt"], k + np_h)
        q_med = (sc["cs"][k_end] - sc["cs"][k]) / max(1, k_end - k)
        p_bud = max(15.0, EPS * SIGMA * A_RAD * ((t_seg + 273.15) ** 4 - T_SPACE ** 4) - q_med)
        f, atuou = escala_orcamento(f, xh[:4], sc["al"][k], p_bud)
        info["n_orcamento"] += atuou
        if usa_barreira:
            m = T >= t_seg
            if m.any():
                cap = fn * np.maximum(0.15, (T_LIM - T) / (T_LIM - t_ref))
                f = np.where(m, np.maximum(F_MIN, np.minimum(f, cap)), f)
                info["n_barreira"] += 1
        info["n"] += 1
        s["f"] = f
        return f
    return acao, info


def autotune_pid(sc, alvo=80.0, n_tune=1200):
    """Grade 3x3x3, criterio ITAE + penalidade de violacao, nos primeiros n_tune passos."""
    dt = sc["dt"]; melhor = (np.inf, None)
    for kp in (-0.025, -0.045, -0.070):
        for ki in (-0.0005, -0.0020, -0.0050):
            for kd in (-0.040, -0.080, -0.120):
                X = sc["x0"].copy(); ie = np.zeros(4); pe = np.zeros(4); acc = 0.0
                for k in range(n_tune - 1):
                    Tc = X[:4] - 273.15; e = Tc - alvo
                    ie = np.clip(ie + e * dt, -200, 200); de = (e - pe) / dt; pe = e
                    f = np.clip(sc["fn"][k] + kp * e + ki * ie + kd * de, F_MIN, F_MAX)
                    acc += (sc["t"][k] * np.abs(e).sum() + (np.maximum(0.0, Tc - 85.0) ** 2).sum() * 50.0) * dt
                    X = X + dt * deriv(X, f, sc["al"][k], sc["qext"][k])[0]
                if acc < melhor[0]:
                    melhor = (acc, (kp, ki, kd))
    return melhor[1]


# ------------------------------------------------------------------ simulacao e metricas
def simular(sc, gov, seed=0, sigma_s=0.25, par=NOM, max_steps=None, t_migr=2.0, fator_migr=0.1):
    """Heun (RK2). Ruido gaussiano branco de sensor, igual para todos os governadores na mesma semente.
    Se info['perm'] mudar, o die movido fica t_migr s com atividade fator_migr e vazao nula (premissa)."""
    rng = np.random.default_rng(seed); nt = sc["nt"]; dt = sc["dt"]
    ns = nt - 1 if max_steps is None else min(nt - 1, max_steps)
    ruido = sigma_s * rng.standard_normal((nt - 1, 5)); acao, info = gov
    X = sc["x0"].copy(); ident = np.arange(4); perm_ant = ident.copy()
    stall = np.zeros(4, int); nst = int(round(t_migr / dt)); nmov = 0
    Th = np.zeros((ns, 5)); Fh = np.zeros((ns, 4)); Pn = np.zeros((ns, 4)); Pl = np.zeros(ns); fps = np.zeros(ns)
    for k in range(ns):
        f = acao(k, X + ruido[k], sc); perm = np.asarray(info.get("perm", ident))
        mv = perm != perm_ant
        if mv.any():
            stall[mv] = nst; nmov += int(mv.sum()); perm_ant = perm.copy()
        act = stall > 0; stall = np.maximum(stall - 1, 0); fac = np.where(act, fator_migr, 1.0)
        al = sc["al"][k][perm] * fac; al2 = sc["al"][k + 1][perm] * fac
        Th[k] = X; Fh[k] = f
        d1, p, pl = deriv(X, f, al, sc["qext"][k], par)
        d2 = deriv(X + dt * d1, f, al2, sc["qext"][k + 1], par)[0]
        Pn[k] = p; Pl[k] = pl.sum(); fps[k] = float((f / F_MAX * FPS_NOM[perm] * (~act)).sum())
        X = X + 0.5 * dt * (d1 + d2)
    info["n_migr"] = nmov
    return {"T": Th[:, :4] - 273.15, "Trad": Th[:, 4] - 273.15, "f": Fh, "Pn": Pn, "Pl": Pl, "fps": fps, "info": info}


def metricas(r, sc, arred=True):
    T = r["T"]; mx = T.max(1); dT = mx - T.min(1); dt = sc["dt"]
    fps = float(r["fps"].mean()); E = float(r["Pn"].sum() * dt / 1000)
    m = {"Tmin": float(T.min()), "Tmed": float(T.mean()), "Tmax": float(mx.max()),
         "dT_pico": float(dT.max()), "dT_med": float(dT.mean()),
         "t_viol_s": float((mx > T_LIM).sum() * dt), "t_acima83_s": float((mx > 83.0).sum() * dt),
         "t_acima125_s": float((mx > 125.0).sum() * dt), "FPS": fps, "E_kJ": E,
         "E_leak_kJ": float(r["Pl"].sum() * dt / 1000), "EDP": E / fps,
         "P_med_W": float(r["Pn"].sum(1).mean()), "f_med": float(r["f"].mean()),
         "Trad_max": float(r["Trad"].max())}
    m["info"] = {kk: float(vv) for kk, vv in r["info"].items() if np.isscalar(vv)}
    if "hit" in r["info"]:
        m["info"]["sdp_hit"] = float(sum(r["info"]["hit"]))
        m["info"]["sdp_chamadas"] = float(len(r["info"]["hit"]))
    if not arred:
        return m
    return {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in m.items()}


# ------------------------------------------------------------------ confiabilidade (indicativa)
def zigzag(x, th):
    pts = [x[0]]; direcao = 0; ult = x[0]
    for v in x[1:]:
        if direcao >= 0:
            if v > ult:
                ult = v; direcao = 1
            elif ult - v >= th:
                pts.append(ult); direcao = -1; ult = v
        if direcao < 0:
            if v < ult:
                ult = v
            elif v - ult >= th:
                pts.append(ult); direcao = 1; ult = v
    pts.append(ult)
    return pts


def rainflow(x, th=0.25):
    """Contagem de ciclos (ASTM E1049, 3 pontos). Retorna [(amplitude [K], peso)]."""
    rv = zigzag(list(x), th); pilha = []; ciclos = []
    for p in rv:
        pilha.append(p)
        while len(pilha) >= 3:
            xr = abs(pilha[-1] - pilha[-2]); yr = abs(pilha[-2] - pilha[-3])
            if xr < yr:
                break
            if len(pilha) == 3:
                ciclos.append((yr, 0.5)); pilha.pop(0)
            else:
                ciclos.append((yr, 1.0)); ult = pilha.pop(); pilha.pop(); pilha.pop(); pilha.append(ult)
    for i in range(len(pilha) - 1):
        ciclos.append((abs(pilha[i + 1] - pilha[i]), 0.5))
    return ciclos


def dano_black(r, jref, dt, n=1.8, ea=0.7):
    J = r["Pn"] / tensao(r["f"]) / jref                       # corrente ~ P/V, normalizada
    return (J ** n * np.exp(-ea / (KB * (r["T"] + 273.15)))).sum(0) * dt


def confiabilidade(rb, rp, sc, skip_h=2.0, limiares=(0.25, 1.0, 2.0)):
    """Razoes base/prop (>1 favorece o proposto). Black usa o historico completo;
    rainflow descarta as primeiras skip_h horas (aquecimento). Use skip_h=2 com 3 orbitas."""
    dt = sc["dt"]; s0 = int(skip_h * 3600 / dt)
    jref = float((rb["Pn"] / tensao(rb["f"])).mean())
    bb = dano_black(rb, jref, dt); bp = dano_black(rp, jref, dt)
    out = {"Black_razao_soma": float(bb.sum() / bp.sum()), "Black_razao_pior_no": float(bb.max() / bp.max()),
           "J_rel_prop": float((rp["Pn"] / tensao(rp["f"])).mean() / jref), "CM": {}}
    for th in limiares:
        d = {}
        for nome, r in (("b", rb), ("p", rp)):
            T = r["T"][s0:]
            cic = [rainflow(T[:, i], th) for i in range(4)]
            d[nome] = {"q21": [sum(n * a ** 2.1 for a, n in c) for c in cic],
                       "q19": [sum(n * a ** 1.9 for a, n in c) for c in cic],
                       "amp": max(a for c in cic for a, n in c), "n": sum(len(c) for c in cic)}
        out["CM"]["%.2f" % th] = {
            "q21_pior_no": round(max(d["b"]["q21"]) / max(max(d["p"]["q21"]), 1e-12), 2),
            "q21_soma": round(sum(d["b"]["q21"]) / max(sum(d["p"]["q21"]), 1e-12), 2),
            "q19_soma": round(sum(d["b"]["q19"]) / max(sum(d["p"]["q19"]), 1e-12), 2),
            "amp_base": round(d["b"]["amp"], 2), "amp_prop": round(d["p"]["amp"], 2),
            "n_base": d["b"]["n"], "n_prop": d["p"]["n"]}
    return out


if __name__ == "__main__":                 # verificacoes estruturais rapidas
    sc = cenario()
    assert sc["nt"] == 34009, sc["nt"]
    assert np.allclose(LAP.sum(1), 0.0)
    print("T_orb = %.1f s | nt = %d | Qext [%.1f, %.1f] W, media %.1f W" %
          (sc["t_orb"], sc["nt"], sc["qext"].min(), sc["qext"].max(), sc["qext"].mean()))
    print("tau dos dies [s]:", np.round(1.0 / np.linalg.eigvalsh(LAP / C_DIE + np.eye(4) / (R_R * C_DIE)), 1))
