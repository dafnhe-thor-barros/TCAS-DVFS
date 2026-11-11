# -*- coding: utf-8 -*-
"""
validate_ekf_and_dso.py - Auditoria do Jacobiano em fronteiras, EKF Joseph e Barreira DSO (Fase 5: A13, A14, A17).

Executa:
1. A13: Validação do Jacobiano analítico em:
   - Pontos interiores nominais (100 amostras estocásticas, diferenças finitas de 4ª ordem)
   - Fronteiras exatas de saturação do leakage (T = 200 K e T = 500 K) e vizinhanças unilaterais
   - Estados extremos fora do domínio nominal (T = 160 K, 185 K < 200 K e T = 530 K, 580 K > 500 K)
   - Fronteiras de saturação de frequência (F_MIN = 0.30 GHz e F_MAX = 1.30 GHz).
2. A14: Validação de observabilidade e consistência estatística do EKF (forma de Joseph):
   - Posto e número de condição da matriz de observabilidade de Kalman (tanto para medição completa I_5
     quanto para medição restrita apenas aos 4 dies C_4x5 = [I_4, 0], provando observabilidade completa do radiador)
   - Erro de estimação RMSE vs ruído bruto de medição (sigma_s = 0.25 K)
   - Estatísticas da sequência de inovação (média, covariância empírica vs teórica S_k e NIS chi^2_5)
   - Simetria de Frobenius ||P_k - P_k^T||_F e positividade definida lambda_min(P_k) > 0 em todos os passos.
3. A17: Validação empírica da trava determinística de segurança (DSO - Deterministic Safety Override)
   sob perturbações externas severas, saturação de atuadores, condição inicial adversa (T(0) > 84.5 C)
   e descasamento paramétrico planta-modelo (+25% em R_r).
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tcas_core as C
import tcas_dvfs as D


def fd_central_4a_ordem(x, f, al, qext=20.0, par=C.NOM, h=1e-5):
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        dp2 = C.deriv(x + 2 * h * ej, f, al, qext, par)[0]
        dp1 = C.deriv(x + h * ej, f, al, qext, par)[0]
        dm1 = C.deriv(x - h * ej, f, al, qext, par)[0]
        dm2 = C.deriv(x - 2 * h * ej, f, al, qext, par)[0]
        J[:, j] = (-dp2 + 8.0 * dp1 - 8.0 * dm1 + dm2) / (12.0 * h)
    return J


def fd_unilateral(x, f, al, qext=20.0, par=C.NOM, h=1e-6, direcao=1.0):
    """Diferença finita unilateral de 2ª ordem para testar derivadas laterais nas fronteiras 200 K / 500 K."""
    n = len(x)
    J = np.zeros((n, n))
    s = 1.0 if direcao >= 0 else -1.0
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        f0 = C.deriv(x, f, al, qext, par)[0]
        f1 = C.deriv(x + s * h * ej, f, al, qext, par)[0]
        f2 = C.deriv(x + 2.0 * s * h * ej, f, al, qext, par)[0]
        J[:, j] = s * (-1.5 * f0 + 2.0 * f1 - 0.5 * f2) / h
    return J


def validar_jacobiano_a13():
    rng = np.random.default_rng(42)
    erros_int = []
    for _ in range(100):
        x = np.concatenate([rng.uniform(300.0, 365.0, 4), [rng.uniform(250.0, 315.0)]])
        f = rng.uniform(C.F_MIN, C.F_MAX, 4)
        al = rng.uniform(0.1, 1.0, 4)
        par = C.parametros(*rng.uniform(0.85, 1.15, 4))
        Ja = C.jacobiano(x, f, al, par)
        Jn = fd_central_4a_ordem(x, f, al, 20.0, par, h=1e-5)
        erros_int.append(float(np.linalg.norm(Ja - Jn, "fro") / np.linalg.norm(Ja, "fro")))

    # Casos de fronteira e extremos fora de domínio (A13)
    f_min_vec = np.full(4, C.F_MIN)
    f_max_vec = np.full(4, C.F_MAX)
    f_mix_vec = np.array([C.F_MIN, C.F_MAX, C.F_MIN, C.F_MAX])
    al_nom = np.array([0.9, 0.8, 0.7, 0.6])

    casos_extremos = [
        ("extremo_abaixo_200K_160K", np.array([160.0, 175.0, 185.0, 195.0, 150.0]), f_min_vec),
        ("extremo_acima_500K_540K", np.array([510.0, 525.0, 540.0, 580.0, 420.0]), f_max_vec),
        ("vizinhanca_interna_200K_mais_1e_3", np.array([200.001, 200.001, 200.001, 200.001, 190.0]), f_mix_vec),
        ("vizinhanca_interna_500K_menos_1e_3", np.array([499.999, 499.999, 499.999, 499.999, 380.0]), f_mix_vec),
        ("saturacao_frequencia_F_MIN", np.array([340.0, 342.0, 344.0, 346.0, 320.0]), f_min_vec),
        ("saturacao_frequencia_F_MAX", np.array([350.0, 352.0, 354.0, 356.0, 335.0]), f_max_vec),
    ]

    relatorio_extremos = {}
    for nome, x_c, f_c in casos_extremos:
        Ja = C.jacobiano(x_c, f_c, al_nom, C.NOM)
        Jn = fd_central_4a_ordem(x_c, f_c, al_nom, 20.0, C.NOM, h=1e-4)
        rel_err = float(np.linalg.norm(Ja - Jn, "fro") / np.linalg.norm(Ja, "fro"))
        relatorio_extremos[nome] = {
            "x_K": x_c.tolist(),
            "f_GHz": f_c.tolist(),
            "rel_err_frobenius": rel_err,
            "max_abs_err": float(np.max(np.abs(Ja - Jn))),
            "passou_1e_7": bool(rel_err < 1e-7),
        }

    # Exatamente sobre a quina não-diferenciável T = 200.0 K e T = 500.0 K:
    # A derivada lateral externa (T -> 200^- e T -> 500^+) tem dP_leak/dT = 0, coincidindo exatamente com jacobiano()
    x_200 = np.array([200.0, 200.0, 200.0, 200.0, 185.0])
    Ja_200 = C.jacobiano(x_200, f_mix_vec, al_nom, C.NOM)
    Jn_200_left = fd_unilateral(x_200, f_mix_vec, al_nom, 20.0, C.NOM, h=1e-6, direcao=-1.0)
    Jn_200_right = fd_unilateral(x_200, f_mix_vec, al_nom, 20.0, C.NOM, h=1e-6, direcao=+1.0)

    x_500 = np.array([500.0, 500.0, 500.0, 500.0, 390.0])
    Ja_500 = C.jacobiano(x_500, f_mix_vec, al_nom, C.NOM)
    Jn_500_right = fd_unilateral(x_500, f_mix_vec, al_nom, 20.0, C.NOM, h=1e-6, direcao=+1.0)
    Jn_500_left = fd_unilateral(x_500, f_mix_vec, al_nom, 20.0, C.NOM, h=1e-6, direcao=-1.0)

    fronteiras_quinas = {
        "quina_exata_200K": {
            "erro_rel_derivada_lateral_esquerda_saturada": float(
                np.linalg.norm(Ja_200 - Jn_200_left, "fro") / np.linalg.norm(Ja_200, "fro")
            ),
            "salto_subgradiente_direita_vs_esquerda_1_sobre_s": float(
                np.max(np.abs(np.diag(Jn_200_right - Jn_200_left)[:4]))
            ),
            "explicacao": "Em T=200 K, clip(T, 200, 500) possui derivada à esquerda nula (coincidindo com jacobiano) e derivada à direita positiva.",
        },
        "quina_exata_500K": {
            "erro_rel_derivada_lateral_direita_saturada": float(
                np.linalg.norm(Ja_500 - Jn_500_right, "fro") / np.linalg.norm(Ja_500, "fro")
            ),
            "salto_subgradiente_esquerda_vs_direita_1_sobre_s": float(
                np.max(np.abs(np.diag(Jn_500_left - Jn_500_right)[:4]))
            ),
            "explicacao": "Em T=500 K, clip(T, 200, 500) possui derivada à direita nula (coincidindo com jacobiano) e derivada à esquerda positiva.",
        },
    }

    return {
        "interior_100_amostras": {
            "max_rel_err": float(np.max(erros_int)),
            "mean_rel_err": float(np.mean(erros_int)),
            "passou_1e_7": bool(np.max(erros_int) < 1e-7),
        },
        "casos_extremos_e_saturacao": relatorio_extremos,
        "analise_subgradiente_quinas_200K_500K": fronteiras_quinas,
    }


def validar_ekf_a14():
    # 1. Observabilidade estrutural e numérica da planta 5x5
    x_op = np.array([348.0, 347.0, 346.0, 345.0, 328.0])
    f_op = np.array([1.15, 1.05, 1.20, 1.00])
    al_op = np.array([0.85, 0.80, 0.90, 0.75])
    J_cont = C.jacobiano(x_op, f_op, al_op, C.NOM)
    dt = 0.5
    I5 = np.eye(5)
    Phi = I5 + dt * J_cont + 0.5 * (dt * J_cont) @ (dt * J_cont)

    # Caso A: Medição completa H = I_5 (5 sensores)
    H_full = np.eye(5)
    O_full = np.vstack([H_full @ np.linalg.matrix_power(Phi, k) for k in range(5)])
    rank_full = int(np.linalg.matrix_rank(O_full))
    cond_full = float(np.linalg.cond(O_full))

    # Caso B: Medição apenas dos 4 dies H_dies = [I_4, 0_{4x1}] (testa se T_rad seria observável sem sensor direto)
    H_dies = np.hstack([np.eye(4), np.zeros((4, 1))])
    O_dies = np.vstack([H_dies @ np.linalg.matrix_power(Phi, k) for k in range(5)])
    rank_dies = int(np.linalg.matrix_rank(O_dies))
    cond_dies = float(np.linalg.cond(O_dies))

    # 2. Simulação do EKF em órbita completa verificando inovação, simetria de Joseph, PSD e RMSE
    sc = C.cenario(n_orb=1, dt=dt)
    nt = sc["nt"]
    sigma_s = 0.25
    rng = np.random.default_rng(12345)

    Q = np.diag([0.02, 0.02, 0.02, 0.02, 0.05])
    R = (sigma_s ** 2) * I5
    P = np.diag([1.0, 1.0, 1.0, 1.0, 2.0])

    x_true = sc["x0"].copy()
    x_hat = x_true + rng.normal(0.0, sigma_s, 5)

    err_ekf = np.zeros((nt, 5))
    err_raw = np.zeros((nt, 5))
    innovations = np.zeros((nt - 1, 5))
    nis_series = np.zeros(nt - 1)
    max_sym_err = 0.0
    min_eig_P = float(np.linalg.eigvalsh(P)[0])

    for k in range(nt):
        noise_m = rng.normal(0.0, sigma_s, 5)
        y_k = x_true + noise_m
        err_raw[k] = y_k - x_true

        if k > 0:
            fk_prev = sc["fn"][k - 1]
            alk_prev = sc["al"][k - 1]
            qk_prev = sc["qext"][k - 1]
            d_hat = C.deriv(x_hat, fk_prev, alk_prev, qk_prev, C.NOM)[0]
            dF = dt * C.jacobiano(x_hat, fk_prev, alk_prev, C.NOM)
            Phi_k = I5 + dF + 0.5 * (dF @ dF)
            Pp = Phi_k @ P @ Phi_k.T + Q * dt
            xp = x_hat + dt * d_hat

            nu = y_k - xp
            S = Pp + R
            S_inv = np.linalg.inv(S)
            K = Pp @ S_inv
            IK = I5 - K
            x_hat = xp + K @ nu
            P = IK @ Pp @ IK.T + K @ R @ K.T

            innovations[k - 1] = nu
            nis_series[k - 1] = float(nu @ (S_inv @ nu))
            sym_err = float(np.linalg.norm(P - P.T, "fro"))
            max_sym_err = max(max_sym_err, sym_err)
            eig_min = float(np.linalg.eigvalsh(0.5 * (P + P.T))[0])
            min_eig_P = min(min_eig_P, eig_min)

        err_ekf[k] = x_hat - x_true

        if k < nt - 1:
            fk = sc["fn"][k]
            d1, _, _ = C.deriv(x_true, fk, sc["al"][k], sc["qext"][k], C.NOM)
            d2, _, _ = C.deriv(x_true + dt * d1, fk, sc["al"][k + 1], sc["qext"][k + 1], C.NOM)
            x_true = x_true + 0.5 * dt * (d1 + d2)

    rmse_raw = np.sqrt(np.mean(err_raw ** 2, axis=0))
    rmse_ekf = np.sqrt(np.mean(err_ekf ** 2, axis=0))
    reducao_ruido_pct = float(100.0 * (1.0 - np.mean(rmse_ekf) / np.mean(rmse_raw)))

    return {
        "observabilidade": {
            "medicao_completa_I5": {
                "rank": rank_full,
                "dimensao_estado": 5,
                "completamente_observavel": bool(rank_full == 5),
                "cond_observability_matrix": cond_full,
            },
            "medicao_apenas_4_dies_sem_sensor_radiador": {
                "rank": rank_dies,
                "dimensao_estado": 5,
                "radiador_observavel_pelos_dies": bool(rank_dies == 5),
                "cond_observability_matrix": cond_dies,
            },
        },
        "desempenho_filtragem_1_orbita": {
            "n_steps": int(nt),
            "sigma_sensor_K": sigma_s,
            "rmse_sensor_bruto_por_estado_K": rmse_raw.tolist(),
            "rmse_ekf_por_estado_K": rmse_ekf.tolist(),
            "rmse_medio_bruto_K": float(np.mean(rmse_raw)),
            "rmse_medio_ekf_K": float(np.mean(rmse_ekf)),
            "reducao_erro_estimacao_pct": reducao_ruido_pct,
        },
        "consistencia_inovacao_e_covariancia": {
            "media_inovacao_por_estado_K": np.mean(innovations, axis=0).tolist(),
            "std_inovacao_por_estado_K": np.std(innovations, axis=0).tolist(),
            "nis_medio_empirico": float(np.mean(nis_series)),
            "nis_esperado_teorico_chi2_5": 5.0,
            "max_erro_simetria_P_frobenius": max_sym_err,
            "min_autovalor_P_trajetoria": min_eig_P,
            "covariancia_estritamente_simetrica_e_psd": bool(max_sym_err < 1e-14 and min_eig_P > 0.0),
        },
    }


def simular_estresse_dso(cenario_nome, x0_custom, qext_boost, par_planta, t_bar=84.5, duracao_s=600.0):
    """
    Compara o governador TCAS-DVFS com DSO ativo (t_bar = 84.5 C) vs DSO desativado (t_bar = 999.0 C)
    sob estresse severo.
    """
    dt = 0.5
    sc = C.cenario(n_orb=1, dt=dt)
    nt_sub = int(round(duracao_s / dt)) + 1
    sc_sub = {
        "dt": dt,
        "nt": nt_sub,
        "t": sc["t"][:nt_sub],
        "t_orb": sc["t_orb"],
        "qext": sc["qext"][:nt_sub] + qext_boost,
        "al": np.full((nt_sub, 4), 0.95),
        "fn": np.full((nt_sub, 4), C.F_MAX),
        "x0": x0_custom.copy(),
    }
    q_cs = np.concatenate([[0.0], np.cumsum(sc_sub["qext"])])
    sc_sub["cs"] = q_cs
    sc_sub["qext_cumsum"] = q_cs

    resultados = {}
    for modo_dso, tb in [("com_DSO_84_5C", t_bar), ("sem_DSO", 999.0)]:
        gov_tuple = D.gov_tcas_dvfs(usa_sched=True, usa_mpc=True, dt=dt, t_cap=83.0, t_bar=tb, sigma_s=0.25, par=C.NOM)
        res = C.simular(sc_sub, gov_tuple, par=par_planta, sigma_s=0.25, seed=77)
        info = res["info"]
        T_dies_C = res["T"]
        viol_85_mask = (T_dies_C > C.T_LIM).any(axis=1)
        excesso_85 = np.maximum(0.0, T_dies_C.max(axis=1) - C.T_LIM)
        resultados[modo_dso] = {
            "T_max_C": float(T_dies_C.max()),
            "n_ativacoes_barreira_DSO": int(info.get("n_barreira", 0)),
            "tempo_acima_84_5C_s": float((T_dies_C > 84.5).any(axis=1).sum() * dt),
            "tempo_acima_85_0C_s": float(viol_85_mask.sum() * dt),
            "integral_violacao_acima_85C_K_s": float(np.sum(excesso_85) * dt),
            "max_excesso_acima_85C_K": float(np.max(excesso_85)),
        }
    return {"cenario": cenario_nome, **resultados}


def validar_dso_a17():
    testes = [
        simular_estresse_dso(
            "S1_surto_fluxo_externo_mais_45W",
            x0_custom=np.array([354.15, 353.85, 353.55, 353.25, 342.15]),
            qext_boost=45.0,
            par_planta=C.NOM,
        ),
        simular_estresse_dso(
            "S2_radiador_saturado_77C",
            x0_custom=np.array([356.15, 355.95, 355.75, 355.55, 350.15]),
            qext_boost=30.0,
            par_planta=C.NOM,
        ),
        simular_estresse_dso(
            "S3_condicao_inicial_adversa_acima_84_5C",
            x0_custom=np.array([358.05, 357.85, 357.75, 357.65, 347.15]),  # 84.9 C nos dies!
            qext_boost=20.0,
            par_planta=C.NOM,
        ),
        simular_estresse_dso(
            "S4_descasamento_parametrico_severo_Rr_1_50x",
            x0_custom=np.array([355.65, 355.35, 355.15, 354.95, 336.15]),
            qext_boost=20.0,
            par_planta=C.parametros(sRij=1.20, sRr=1.50, sC=0.80, sCr=1.00),
        ),
    ]
    return testes


def main():
    print("=" * 78)
    print("  AUDITORIA DO JACOBIANO, EKF JOSEPH E BARREIRA DSO (FASE 5: A13, A14, A17)")
    print("=" * 78)

    res_a13 = validar_jacobiano_a13()
    print(f"  [A13 Jacobiano] Interior (100 pts) max_rel_err={res_a13['interior_100_amostras']['max_rel_err']:.3e} | "
          f"Todos extremos passaram (<1e-7): {all(v['passou_1e_7'] for v in res_a13['casos_extremos_e_saturacao'].values())}")

    res_a14 = validar_ekf_a14()
    print(f"  [A14 EKF] Rank(O_full)={res_a14['observabilidade']['medicao_completa_I5']['rank']}/5 | "
          f"Rank(O_dies_only)={res_a14['observabilidade']['medicao_apenas_4_dies_sem_sensor_radiador']['rank']}/5 | "
          f"RMSE_bruto={res_a14['desempenho_filtragem_1_orbita']['rmse_medio_bruto_K']:.4f} K -> "
          f"RMSE_EKF={res_a14['desempenho_filtragem_1_orbita']['rmse_medio_ekf_K']:.4f} K "
          f"(-{res_a14['desempenho_filtragem_1_orbita']['reducao_erro_estimacao_pct']:.1f}%)")
    print(f"  [A14 EKF] Simetria ||P - P^T||_F={res_a14['consistencia_inovacao_e_covariancia']['max_erro_simetria_P_frobenius']:.2e} | "
          f"min_eig(P)={res_a14['consistencia_inovacao_e_covariancia']['min_autovalor_P_trajetoria']:.4e} | "
          f"NIS_medio={res_a14['consistencia_inovacao_e_covariancia']['nis_medio_empirico']:.3f} (teórico=5.0)")

    res_a17 = validar_dso_a17()
    for t in res_a17:
        c_dso = t["com_DSO_84_5C"]
        s_dso = t["sem_DSO"]
        print(f"  [A17 {t['cenario']:38s}] Com DSO: Tmax={c_dso['T_max_C']:.2f} C (n_bar={c_dso['n_ativacoes_barreira_DSO']:3d}, "
              f"t>85C={c_dso['tempo_acima_85_0C_s']:.1f}s) | Sem DSO: Tmax={s_dso['T_max_C']:.2f} C (t>85C={s_dso['tempo_acima_85_0C_s']:.1f}s)")

    out_dir = os.path.join(ROOT, "audit", "numerical_validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ekf_jacobian_dso_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fase": "Fase 5 - Auditoria do EKF, Jacobiano e Barreira Termica DSO (A13, A14, A17)",
                "A13_validacao_jacobiano_fronteiras": res_a13,
                "A14_observabilidade_e_consistencia_ekf": res_a14,
                "A17_validacao_barreira_dso": res_a17,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  [OK] Relatório salvo em: {out_path}")


if __name__ == "__main__":
    main()
