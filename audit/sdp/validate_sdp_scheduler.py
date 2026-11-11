# -*- coding: utf-8 -*-
"""
validate_sdp_scheduler.py - Auditoria formal e validação numérica do escalonador SDP / ADMM (Fase 4: A11, A12).

Executa:
1. A11: Documentação formal do problema de relaxação semidefinida (Shor/Lifted QAP),
   variáveis primais (Y, Z1, Z2) e duais escalonadas (U1, U2), equações de atualização
   ADMM de consenso em 2 blocos, projeção do resíduo dual sobre ker(E) e tolerâncias
   combinadas absolutas/relativas.
2. A12: Validação empírica em 50 instâncias de matrizes de diafonia térmica Q = (p p^T) (x) Psi - mu diag(x_prev),
   separando estritamente:
   - Viabilidade cônica semidefinida: lambda_min(Z1) >= -1e-14 e lambda_min(Y)
   - Viabilidade afim exata: ||E vec(Y) - e||_inf <= 1e-12
   - Viabilidade de não-negatividade: min(Z2) >= 0
   - Convergência numérica ADMM: r_pri <= eps_pri e r_dua <= eps_dua
   - Qualidade heurística pós-arredondamento contra enumeração exata das 24 permutações
     (heuristic_suboptimality_ratio e exact_hit_rate, sem reivindicar gap dual certificado).
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tcas_core as C
import tcas_dvfs as D


def main():
    print("=" * 78)
    print("  AUDITORIA E VALIDAÇÃO DO ESCALONADOR SDP / ADMM (FASE 4: A11, A12)")
    print("=" * 78)

    N = 4
    par = C.NOM
    Phi = np.linalg.inv(par["LAP"] + np.eye(N) / par["RR"])
    Hc_ = np.eye(N) - np.ones((N, N)) / N
    Psi = Phi @ Hc_ @ Phi
    mu_mig = 1.0

    rng = np.random.default_rng(20261111)
    sdp_warm = D.fabrica_sdp(N=4, iters=120, rho=1.0, n_round=24, seed=42)
    sdp_cold = D.fabrica_sdp(N=4, iters=120, rho=1.0, n_round=24, seed=42)
    sdp_exato = D.fabrica_sdp(N=4, iters=120, rho=1.0, n_round=24, seed=42)

    n_instances = 50
    perm_prev = np.arange(N)

    warm_records = []
    cold_records = []

    for idx in range(n_instances):
        # Gera vetor de potências das 4 tarefas heterogêneas (entre 3 W e 28 W por die)
        phat = rng.uniform(3.0, 28.0, size=N)
        Xprev = np.zeros((N, N))
        Xprev[perm_prev, np.arange(N)] = 1.0
        Q = np.kron(np.outer(phat, phat), Psi) + np.diag(-mu_mig * Xprev.reshape(-1))

        pm_ex, dg_ex = sdp_exato(Q, modo="exato")
        pm_w, dg_w = sdp_warm(Q, modo="sdp", warm=True)
        pm_c, dg_c = sdp_cold(Q, modo="sdp", warm=False)

        perm_prev = pm_w.copy()

        warm_records.append({
            "instance": idx,
            "J_exact_min": dg_ex["Jex"],
            "J_sdp_rounded": dg_w["J"],
            "exact_hit": dg_w["hit"],
            "heuristic_suboptimality_ratio": dg_w["heuristic_suboptimality_ratio"],
            "it": dg_w["it"],
            "converged": dg_w["converged"],
            "r_pri": dg_w["r_pri"],
            "r_dua": dg_w["r_dua"],
            "eps_pri": dg_w["eps_pri"],
            "eps_dua": dg_w["eps_dua"],
            "affine_viol_inf": dg_w["affine_viol_inf"],
            "lambda_min_Z1": dg_w["lambda_min_Z1"],
            "lambda_min_Y": dg_w["lambda_min_Y"],
            "min_Z2": dg_w["min_Z2"],
        })

        cold_records.append({
            "instance": idx,
            "J_exact_min": dg_c["Jex"],
            "J_sdp_rounded": dg_c["J"],
            "exact_hit": dg_c["hit"],
            "heuristic_suboptimality_ratio": dg_c["heuristic_suboptimality_ratio"],
            "it": dg_c["it"],
            "converged": dg_c["converged"],
            "r_pri": dg_c["r_pri"],
            "r_dua": dg_c["r_dua"],
            "affine_viol_inf": dg_c["affine_viol_inf"],
            "lambda_min_Z1": dg_c["lambda_min_Z1"],
            "lambda_min_Y": dg_c["lambda_min_Y"],
            "min_Z2": dg_c["min_Z2"],
        })

    def resumir(recs):
        return {
            "n_instances": len(recs),
            "viabilidade_semidefinida_Z1": {
                "min_lambda_min_Z1": float(min(r["lambda_min_Z1"] for r in recs)),
                "max_lambda_min_Z1": float(max(r["lambda_min_Z1"] for r in recs)),
                "psd_strictly_satisfied_tol_1e_14": bool(all(r["lambda_min_Z1"] >= -1e-14 for r in recs)),
            },
            "viabilidade_espaco_Y_consenso": {
                "min_lambda_min_Y": float(min(r["lambda_min_Y"] for r in recs)),
                "mean_lambda_min_Y": float(np.mean([r["lambda_min_Y"] for r in recs])),
            },
            "viabilidade_afim_EY_eq_e": {
                "max_affine_viol_inf": float(max(r["affine_viol_inf"] for r in recs)),
                "mean_affine_viol_inf": float(np.mean([r["affine_viol_inf"] for r in recs])),
                "affine_satisfied_tol_1e_12": bool(all(r["affine_viol_inf"] <= 1e-12 for r in recs)),
            },
            "viabilidade_nao_negatividade_Z2": {
                "min_entry_Z2": float(min(r["min_Z2"] for r in recs)),
                "nonnegative_satisfied": bool(all(r["min_Z2"] >= 0.0 for r in recs)),
            },
            "convergencia_numerica_admm": {
                "convergence_rate_pct": float(100.0 * np.mean([r["converged"] for r in recs])),
                "mean_iterations": float(np.mean([r["it"] for r in recs])),
                "mean_r_pri": float(np.mean([r["r_pri"] for r in recs])),
                "max_r_pri": float(max(r["r_pri"] for r in recs)),
                "mean_r_dua": float(np.mean([r["r_dua"] for r in recs])),
                "max_r_dua": float(max(r["r_dua"] for r in recs)),
            },
            "qualidade_heuristica_pos_arredondamento": {
                "exact_hit_rate_pct": float(100.0 * np.mean([r["exact_hit"] for r in recs])),
                "mean_heuristic_suboptimality_ratio": float(np.mean([r["heuristic_suboptimality_ratio"] for r in recs])),
                "max_heuristic_suboptimality_ratio": float(max(r["heuristic_suboptimality_ratio"] for r in recs)),
            },
        }

    sum_warm = resumir(warm_records)
    sum_cold = resumir(cold_records)

    print(f"  [Warm Start (50 inst.)] PSD Z1 OK: {sum_warm['viabilidade_semidefinida_Z1']['psd_strictly_satisfied_tol_1e_14']} "
          f"(min eig={sum_warm['viabilidade_semidefinida_Z1']['min_lambda_min_Z1']:.2e}) | "
          f"Viol. Afim max={sum_warm['viabilidade_afim_EY_eq_e']['max_affine_viol_inf']:.2e}")
    print(f"  [Warm Start (50 inst.)] Conv. ADMM={sum_warm['convergencia_numerica_admm']['convergence_rate_pct']:.1f}% "
          f"(it_med={sum_warm['convergencia_numerica_admm']['mean_iterations']:.1f}) | "
          f"Hit Exato={sum_warm['qualidade_heuristica_pos_arredondamento']['exact_hit_rate_pct']:.1f}% | "
          f"Subotim. Heurística={sum_warm['qualidade_heuristica_pos_arredondamento']['mean_heuristic_suboptimality_ratio']:.4e}")
    print(f"  [Cold Start (50 inst.)] Conv. ADMM={sum_cold['convergencia_numerica_admm']['convergence_rate_pct']:.1f}% "
          f"(it_med={sum_cold['convergencia_numerica_admm']['mean_iterations']:.1f}) | "
          f"Hit Exato={sum_cold['qualidade_heuristica_pos_arredondamento']['exact_hit_rate_pct']:.1f}%")

    formulacao_a11 = {
        "problema_otimizacao": (
            "min_{Y in S^17} <Cm, Y> s.a. E vec(Y) = e, Z1 = Y in S_+^17 (PSD), Z2 = Y >= 0 (duplamente não-negativo)"
        ),
        "dimensoes": {"N_dies": 4, "n_binarias": 16, "d_matriz_levantada": 17, "n_restricoes_afins": 289},
        "variaveis_primais": {
            "Y": "Matriz levantada 17x17 satisfazendo exatamente o subespaço afim E vec(Y) = e",
            "Z1": "Projeção de consenso no cone semidefinido positivo S_+^17 via decomposição espectral",
            "Z2": "Projeção de consenso no ortante não-negativo R_+^(17x17)",
        },
        "variaveis_duais": {
            "U1": "Multiplicador dual escalonado associado ao consenso Y - Z1 = 0",
            "U2": "Multiplicador dual escalonado associado ao consenso Y - Z2 = 0",
        },
        "equacoes_atualizacao_admm": [
            "1. Passo afim (Y): v_tg = vec(0.5*((Z1 - U1) + (Z2 - U2)) - Cm/(2*rho)); vec(Y) = v_tg - E^+ (E v_tg - e)",
            "2. Passo cônico PSD (Z1): Z1 = V diag(max(lambda(Y + U1), 0)) V^T",
            "3. Passo cônico não-negativo (Z2): Z2 = max(Y + U2, 0)",
            "4. Passo dual (U1, U2): U1 <- U1 + Y - Z1; U2 <- U2 + Y - Z2",
        ],
        "residuos_e_tolerancias": {
            "r_pri": "sqrt(||Y - Z1||_F^2 + ||Y - Z2||_F^2)",
            "r_dua": "rho * ||(I - E^+ E) vec((Z1 - Z1_old) + (Z2 - Z2_old))||_2",
            "eps_pri": "1e-4 + 1e-3 * max(sqrt(2)*||Y||_F, sqrt(||Z1||_F^2 + ||Z2||_F^2))",
            "eps_dua": "1e-4 + 1e-3 * rho * ||(I - E^+ E) vec(U1 + U2)||_2",
        },
        "distincao_conceitual_estrita_A12": {
            "viabilidade": "Medida por ||E vec(Y) - e||_inf, lambda_min(Z1) e min(Z2)",
            "convergencia_numerica": "Medida por r_pri <= eps_pri e r_dua <= eps_dua",
            "qualidade_heuristica": "Medida por exact_hit e heuristic_suboptimality_ratio = (J_round - J_exact)/(J_mean - J_exact)",
            "otimalidade_certificada": (
                "Para N=4, o ótimo global exato J_exact é certificado diretamente pela enumeração exata das 4! = 24 "
                "permutações em O(N!), eliminando qualquer necessidade de reivindicar limite dual aproximado do ADMM."
            ),
        },
    }

    out_dir = os.path.join(ROOT, "audit", "sdp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sdp_validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fase": "Fase 4 - Auditoria Formal e Validacao do SDP (A11, A12)",
                "A11_formulacao_matematica": formulacao_a11,
                "A12_resumo_warm_start_50_instancias": sum_warm,
                "A12_resumo_cold_start_50_instancias": sum_cold,
                "A12_amostras_detalhadas_warm": warm_records[:10],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  [OK] Relatório salvo em: {out_path}")


if __name__ == "__main__":
    main()
