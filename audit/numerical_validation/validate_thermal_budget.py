# -*- coding: utf-8 -*-
"""
validate_thermal_budget.py - Derivação física formal e validação numérica do orçamento térmico (Fase 3: A09, A10).

Executa:
1. A09: Verificação dimensional, consistência de sinais e documentação formal da derivação:
   - Limite superior de temperatura do radiador:
       T_rad,max = T_cap - (P_min,cores * R_r) / 4   [K]
   - Constante de tempo RC condutiva radiador-dies:
       tau_rad = (C_rad * R_r) / 4                   [s]
   - Buffer térmico transiente analítico no horizonte tau_horiz = 240 s:
       P_buffer = (C_rad * (T_rad,max - T_rad) / tau_horiz) * (1 - exp(-tau_horiz / tau_rad))  [W]
   - Limite instantâneo de condução de Fourier:
       P_cond,max = 4 * max(0, T_cap - T_rad) / R_r  [W]
   - Fechamento do orçamento admissível:
       P_budget = min(P_cond,max, Q_rad,emit - q_m + P_buffer)  [W]

2. A10 (proxy numérico): Comparação exploratória entre o orçamento analítico e uma integração
   Heun de 5 estados com injeção térmica constante distribuída igualmente entre os dies.
   Esta integração NÃO recalcula potência dinâmica e fuga em função de T e f e, portanto,
   NÃO representa a planta eletrotérmica completa nem valida o controlador TCAS-DVFS. Regimes:
   (1) Regime estacionário
   (2) Aquecimento transitório
   (3) Resfriamento
   (4) Degrau abrupto de carga
   (5) Eclipse orbital
   (6) Incidência solar máxima (zênite + albedo + IR)
   (7) Contingência energética / saturação radiativa (P_budget < P_min,cores)
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tcas_core as C


def calcular_orcamento_analitico(xh, al, qm, t_cap=83.0, tau_horiz=240.0, par=C.NOM):
    """
    Calcula todos os termos do fechamento analítico do orçamento térmico de gov_tcas_dvfs,
    retornando dicionário detalhado com valores e unidades físicas verificadas.
    """
    N = 4
    Trad_est = float(xh[4])
    Tcap_K = float(t_cap + 273.15)
    RR = float(par["RR"])
    CRAD = float(par["CRAD"])

    Q_rad_emit = float(C.EPS * C.SIGMA * C.A_RAD * (Trad_est ** 4 - C.T_SPACE ** 4))
    Q_rad_liq = float(Q_rad_emit - qm)

    pd_min, pl_min = C.potencias(xh[:4], np.full(N, C.F_MIN), al)
    P_min_cores = float((pd_min + pl_min).sum())

    Trad_max = float(Tcap_K - (P_min_cores * RR) / 4.0)
    tau_rad = float(CRAD * RR / 4.0)

    if Trad_est < Trad_max:
        P_buffer = float((CRAD * (Trad_max - Trad_est) / tau_horiz) * (1.0 - np.exp(-tau_horiz / tau_rad)))
    else:
        P_buffer = 0.0

    P_cond_inst = float(4.0 * max(0.0, Tcap_K - Trad_est) / RR)
    alpha_h = float(tau_horiz / tau_rad)
    P_cond_coupled = float(max(0.0, (P_cond_inst + alpha_h * Q_rad_liq) / (1.0 + alpha_h)))
    P_cond_max = float(min(P_cond_inst, P_cond_coupled))
    P_rad_total = float(Q_rad_liq + P_buffer)
    P_budget_fisico = float(min(P_cond_max, P_rad_total))
    inviavel_fisico = bool(P_budget_fisico < P_min_cores)
    deficit_w = float(max(0.0, P_min_cores - P_budget_fisico))

    return {
        "Trad_est_K": Trad_est,
        "Tcap_K": Tcap_K,
        "Trad_max_K": Trad_max,
        "tau_rad_s": tau_rad,
        "tau_horiz_s": tau_horiz,
        "Q_rad_emit_W": Q_rad_emit,
        "qm_ext_W": float(qm),
        "Q_rad_liq_W": Q_rad_liq,
        "P_min_cores_W": P_min_cores,
        "P_buffer_W": P_buffer,
        "P_rad_mais_buffer_W": P_rad_total,
        "P_cond_inst_W": P_cond_inst,
        "P_cond_coupled_W": P_cond_coupled,
        "P_cond_max_W": P_cond_max,
        "P_budget_analitico_W": P_budget_fisico,
        "inviavel_fisico": inviavel_fisico,
        "deficit_W": deficit_w,
        "limitante_ativo": "conducao_fourier_acoplada" if P_cond_max < P_rad_total else "balanco_radiativo_buffer",
    }


def simular_horizonte_ode_injecao_constante(xh0, P_alvo_total, qext_serie, dt=0.5, par=C.NOM):
    """Integração Heun com injeção térmica constante e uniforme nos quatro dies.

    Esta função é um proxy térmico: não chama C.deriv() e não recalcula potência
    dinâmica/fuga em função da temperatura e da frequência. Não deve ser descrita
    como integração da planta eletrotérmica completa.
    """
    x = xh0.copy().astype(float)
    nt = len(qext_serie)
    p_per_die = np.full(4, P_alvo_total / 4.0)
    max_T_die = float(x[:4].max())

    def deriv_p_fixo(X, qext_val):
        T = X[:4]
        tr = X[4]
        qr = (T - tr) / par["RR"]
        d = np.empty(5)
        d[:4] = (p_per_die - par["LAP"] @ T - qr) / par["C"]
        d[4] = (qr.sum() + qext_val - C.EPS * C.SIGMA * C.A_RAD * (tr ** 4 - C.T_SPACE ** 4)) / par["CRAD"]
        return d

    for k in range(nt):
        qk = float(qext_serie[k])
        d1 = deriv_p_fixo(x, qk)
        x_pred = x + dt * d1
        qk_next = float(qext_serie[min(k + 1, nt - 1)])
        d2 = deriv_p_fixo(x_pred, qk_next)
        x = x + 0.5 * dt * (d1 + d2)
        max_T_die = max(max_T_die, float(x[:4].max()))

    return {
        "max_T_die_K": max_T_die,
        "final_T_die_mean_K": float(x[:4].mean()),
        "final_Trad_K": float(x[4]),
    }


def orcamento_numerico_ode_proxy(xh0, Trad_max_K, Tcap_K, qext_serie, dt=0.5, par=C.NOM):
    """
    Encontra por bisseção sobre a integração numérica não linear de Heun da EDO de 5 estados
    a potência total constante máxima P_ode_max [W] no horizonte de 240 s tal que:
      1) max_{t in [0, 240s]} max_i T_i(t) <= Tcap_K (restrição de condução/temperatura nos dies)
      2) T_rad(240 s) <= Trad_est + (Trad_max_K - Trad_est) * (1 - exp(-tau_horiz/tau_rad))
         (mesma meta de excursão máxima admitida para o radiador no horizonte de controle).
    Também calcula P_ode_cap_only (limitado apenas por max_i T_i(t) <= Tcap_K em 240 s).
    """
    tau_horiz = len(qext_serie) * dt
    tau_rad = float(par["CRAD"] * par["RR"] / 4.0)
    Trad_est = float(xh0[4])
    if Trad_est < Trad_max_K:
        Trad_target = Trad_est + (Trad_max_K - Trad_est) * (1.0 - np.exp(-tau_horiz / tau_rad))
    else:
        Trad_target = Trad_est

    # Bisseção 1: Respeitando simultaneamente T_die <= Tcap_K e T_rad(240s) <= Trad_target
    lo, hi = 0.0, 200.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        sim = simular_horizonte_ode_injecao_constante(xh0, mid, qext_serie, dt=dt, par=par)
        if sim["max_T_die_K"] <= Tcap_K and sim["final_Trad_K"] <= Trad_target:
            lo = mid
        else:
            hi = mid
    p_ode_acoplado = 0.5 * (lo + hi)

    # Bisseção 2: Respeitando apenas max_i T_i(t) <= Tcap_K no horizonte de 240 s
    lo2, hi2 = 0.0, 200.0
    for _ in range(45):
        mid = 0.5 * (lo2 + hi2)
        sim = simular_horizonte_ode_injecao_constante(xh0, mid, qext_serie, dt=dt, par=par)
        if sim["max_T_die_K"] <= Tcap_K:
            lo2 = mid
        else:
            hi2 = mid
    p_ode_tcap = 0.5 * (lo2 + hi2)

    return p_ode_acoplado, p_ode_tcap, Trad_target


def main():
    print("=" * 78)
    print("  VALIDAÇÃO FÍSICA DO ORÇAMENTO TÉRMICO (FASE 3: A09, A10)")
    print("=" * 78)

    dt = 0.5
    tau_horiz = 240.0
    nt_h = int(round(tau_horiz / dt))
    al_nom = np.array([0.85, 0.80, 0.75, 0.75])
    t_cap = 83.0
    Tcap_K = t_cap + 273.15

    # 7 Regimes exigidos em A10
    t_grid = np.arange(nt_h) * dt
    cenarios_7 = [
        {
            "id": "R1_regime_estacionario",
            "descricao": "Equilíbrio térmico quase-estacionário em insolação moderada (Trad = 325 K, T_dies = 342 K)",
            "xh": np.array([342.0, 342.0, 342.0, 342.0, 325.0]),
            "qext_serie": np.full(nt_h, 18.0),
            "al": al_nom,
        },
        {
            "id": "R2_aquecimento_transitorio",
            "descricao": "Aquecimento transitório rápido com rampa de fluxo externo (15 W -> 30 W) e dies a 349 K",
            "xh": np.array([349.5, 349.0, 348.5, 348.0, 332.0]),
            "qext_serie": np.linspace(15.0, 30.0, nt_h),
            "al": np.array([0.95, 0.90, 0.85, 0.80]),
        },
        {
            "id": "R3_resfriamento",
            "descricao": "Entrada em fase fria de resfriamento radiativo (Trad = 336 K, qext caindo de 18 W para 6.5 W)",
            "xh": np.array([346.0, 345.5, 345.0, 344.5, 336.0]),
            "qext_serie": np.linspace(18.0, 6.5, nt_h),
            "al": np.array([0.65, 0.65, 0.60, 0.60]),
        },
        {
            "id": "R4_degrau_abrupto_carga",
            "descricao": "Degrau abrupto de atividade computacional e carga ambiental no meio do horizonte",
            "xh": np.array([348.0, 347.5, 347.0, 346.5, 330.0]),
            "qext_serie": np.where(t_grid < 120.0, 12.0, 28.0),
            "al": np.array([0.95, 0.95, 0.90, 0.85]),
        },
        {
            "id": "R5_eclipse_orbital",
            "descricao": "Operação em umbra/eclipse LEO (apenas IR terrestre ~ 6.8 W, sem fluxo solar direto)",
            "xh": np.array([336.0, 335.5, 335.0, 334.5, 315.0]),
            "qext_serie": np.full(nt_h, 6.8),
            "al": al_nom,
        },
        {
            "id": "R6_incidencia_solar_zenite",
            "descricao": "Pico de insolação zênite + albedo + IR terrestre (~ 31.5 W) com radiador aquecido (340 K)",
            "xh": np.array([351.5, 351.0, 350.5, 350.0, 340.0]),
            "qext_serie": np.full(nt_h, 31.5),
            "al": al_nom,
        },
        {
            "id": "R7_contingencia_energetica_saturacao",
            "descricao": "Saturação térmica severa (Trad = 349.5 K > Trad_max), acionando contingência física P_budget < P_min",
            "xh": np.array([355.5, 355.2, 355.0, 354.8, 349.5]),
            "qext_serie": np.full(nt_h, 32.0),
            "al": np.array([0.95, 0.95, 0.95, 0.95]),
        },
    ]

    resultados_regimes = {}
    for c in cenarios_7:
        cid = c["id"]
        xh = c["xh"]
        qext_s = c["qext_serie"]
        qm = float(np.mean(qext_s))
        al = c["al"]

        ana = calcular_orcamento_analitico(xh, al, qm, t_cap=t_cap, tau_horiz=tau_horiz)
        p_ode_acop, p_ode_tcap, trad_target = orcamento_numerico_ode_proxy(
            xh, ana["Trad_max_K"], Tcap_K, qext_s, dt=dt
        )

        # Simula a EDO aplicando exatamente o orçamento analítico (ou P_min_cores em contingência)
        p_aplicado = ana["P_min_cores_W"] if ana["inviavel_fisico"] else ana["P_budget_analitico_W"]
        sim_ana = simular_horizonte_ode_injecao_constante(xh, p_aplicado, qext_s, dt=dt)

        erro_w = float(ana["P_budget_analitico_W"] - p_ode_acop)
        erro_rel_pct = float(100.0 * abs(erro_w) / max(p_ode_acop, 1e-6))
        margem_seguranca_tcap_w = float(p_ode_tcap - ana["P_budget_analitico_W"])

        resultados_regimes[cid] = {
            "descricao": c["descricao"],
            "analitico": ana,
            "numerico_ode_heun": {
                "P_ode_acoplado_W": float(p_ode_acop),
                "P_ode_limite_Tcap_240s_W": float(p_ode_tcap),
                "Trad_target_240s_K": float(trad_target),
                "erro_analitico_vs_ode_acoplado_W": erro_w,
                "erro_relativo_acoplado_pct": erro_rel_pct,
                "margem_conservadora_ate_violacao_Tcap_W": margem_seguranca_tcap_w,
                "simulacao_sob_orcamento_analitico": {
                    "P_aplicado_W": float(p_aplicado),
                    "max_T_die_240s_C": float(sim_ana["max_T_die_K"] - 273.15),
                    "final_Trad_240s_C": float(sim_ana["final_Trad_K"] - 273.15),
                    # A inviabilidade física não pode converter uma violação térmica em aprovação.
                    "operacao_viavel": bool(not ana["inviavel_fisico"]),
                    "violou_Tcap": bool(sim_ana["max_T_die_K"] > Tcap_K + 1e-3),
                    "respeitou_Tcap": bool(sim_ana["max_T_die_K"] <= Tcap_K + 1e-3),
                    "status_operacao_viavel": (
                        "not_applicable" if ana["inviavel_fisico"] else
                        ("pass" if sim_ana["max_T_die_K"] <= Tcap_K + 1e-3 else "fail")
                    ),
                },
            },
        }
        print(
            f"  [{cid:36s}] P_ana={ana['P_budget_analitico_W']:6.2f} W | "
            f"P_ODE_acop={p_ode_acop:6.2f} W | P_ODE_Tcap={p_ode_tcap:6.2f} W | "
            f"T_max_sim={sim_ana['max_T_die_K']-273.15:5.2f} C | inviavel={ana['inviavel_fisico']}"
        )

    derivacao_formal = {
        "equacao_1_conducao_fourier_quase_estatica": {
            "expressao": "T_rad,max = T_cap - (P_min,cores * R_r) / 4",
            "deducao": (
                "Em equilíbrio quase-estático uniforme entre os 4 dies e o radiador, o fluxo condutivo total é "
                "Q_cond = sum_{i=1}^4 (T_i - T_rad)/R_r. Para que nenhum die exceda T_cap quando operado na potência "
                "mínima admissível P_min,cores = sum_{i=1}^4 P_i(F_MIN, T_i), exige-se 4*(T_cap - T_rad)/R_r >= P_min,cores, "
                "resultando em T_rad <= T_cap - (P_min,cores * R_r)/4."
            ),
            "unidades": {"T_rad,max": "K", "T_cap": "K", "P_min,cores": "W", "R_r": "K/W"},
            "consistencia_dimensional": "K - (W * K/W) = K (Verificado)",
        },
        "equacao_2_constante_tempo_rc_radiador": {
            "expressao": "tau_rad = (C_rad * R_r) / 4",
            "valor_nominal_s": float(C.C_RAD * C.R_R / 4.0),
            "unidades": {"tau_rad": "s", "C_rad": "J/K", "R_r": "K/W"},
            "consistencia_dimensional": "(J/K) * (K/W) = J / (J/s) = s (Verificado)",
        },
        "equacao_3_buffer_entalpico_transiente": {
            "expressao": "P_buffer = (C_rad * max(0, T_rad,max - T_rad) / tau_horiz) * (1 - exp(-tau_horiz / tau_rad))",
            "deducao": (
                "A reserva entálpica disponível no radiador até atingir T_rad,max é Delta_H = C_rad * (T_rad,max - T_rad) [J]. "
                "Distribuída sobre o horizonte preditivo tau_horiz = 240 s e ponderada pelo fator de relaxação RC de 1ª ordem "
                "(1 - exp(-tau_horiz/tau_rad)) da resposta ao degrau térmico entre os 4 dies e o radiador, obtém-se a taxa "
                "média segura de absorção transiente P_buffer [W]."
            ),
            "unidades": {"P_buffer": "W", "C_rad": "J/K", "Delta_T": "K", "tau_horiz": "s"},
            "consistencia_dimensional": "(J/K * K / s) * 1 = J/s = W (Verificado)",
        },
        "equacao_4_fechamento_orcamento": {
            "expressao": "P_budget = min(4 * max(0, T_cap - T_rad) / R_r, Q_rad,emit(T_rad) - q_m + P_buffer)",
            "tratamento_contingencia": (
                "Se P_budget < P_min,cores, o estado é classificado explicitamente como 'physically_infeasible', "
                "incrementando n_deficit_orcamento e n_contingencia_inviavel e aplicando F_MIN sem mascarar o QP com piso artificial."
            ),
        },
    }

    out_dir = os.path.join(ROOT, "audit", "numerical_validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "thermal_budget_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fase": "Fase 3 - Fechamento Fisico do Orcamento Termico (A09, A10)",
                "A09_derivacao_e_unidades": derivacao_formal,
                "A10_validacao_7_regimes_vs_ode": resultados_regimes,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  [OK] Relatório salvo em: {out_path}")


if __name__ == "__main__":
    main()
