# -*- coding: utf-8 -*-
"""
validate_reliability_metrics.py - Validação ASTM E1049-85 de Rainflow e Delimitação Física de Black / Coffin-Manson (Fase 6: A15, A16).

Executa:
1. A15: Validação analítica e normativa do algoritmo Rainflow de 3 pontos (ASTM E1049-85, Seção 5.4.4):
   - Onda triangular exata (2 ciclos completos, range Delta_T = 10.0 K)
   - Onda senoidal (3 ciclos, amplitude 5.0 K, range pico-a-pico Delta_T = 10.0 K)
   - Sequência canônica ASTM E1049-85 Seção 5.4.4: [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0],
     verificando cada ciclo fechado (peso 1.0) e meio-ciclo residual (peso 0.5) e a soma exata de 5.0 ciclos.
   - Verificação de invariância de escala na razão adimensional de Coffin-Manson:
     (Delta_T)^q vs (Delta_T / 2)^q cancela exatamente o fator 2^q na razão D_base / D_prop.
2. A16: Documentação e verificação numérica da delimitação física dos modelos de Black (eletromigração)
   e Coffin-Manson (fadiga termomecânica), comprovando que as métricas reportadas (Black_razao_soma,
   Black_razao_pior_no, q21_soma, q19_soma) são índices comparativos relativos adimensionais invariantes
   às constantes geométricas/materiais pré-exponenciais (A_Black, A_interconnect, B_CM), não constituindo
   predição absoluta de MTTF em horas.
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tcas_core as C


def validar_rainflow_a15():
    # 1. Onda triangular [0, 10, 0, 10, 0]
    t_tri = [0.0, 10.0, 0.0, 10.0, 0.0]
    cic_tri = C.rainflow(t_tri, th=0.1)
    tri_ok = bool(all(np.isclose(r, 10.0) for r, _ in cic_tri) and np.isclose(sum(w for _, w in cic_tri), 2.0))

    # 2. Onda senoidal (3 ciclos, amplitude 5 K -> range 10 K)
    theta = np.linspace(0.0, 6.0 * np.pi, 600)
    y_sin = 80.0 + 5.0 * np.sin(theta)
    cic_sin = C.rainflow(y_sin, th=0.05)
    completos_sin = [r for r, w in cic_sin if w == 1.0]
    sin_ok = bool(len(completos_sin) == 2 and all(np.isclose(r, 10.0, atol=0.05) for r in completos_sin))

    # 3. Vetor canônico ASTM E1049-85 Seção 5.4.4
    pts_astm = [0.0, -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0, 0.0]
    cic_astm = C.rainflow(pts_astm, th=0.01)
    contagem_ranges = {}
    lista_detalhada = []
    for r, w in cic_astm:
        if r < 1e-6:
            continue
        contagem_ranges[float(r)] = contagem_ranges.get(float(r), 0.0) + float(w)
        lista_detalhada.append({"delta_T_range_K": float(r), "semi_amplitude_K": float(r / 2.0), "peso_ciclos": float(w)})

    esperado_astm = {2.0: 1.0, 3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}
    astm_ok = bool(
        len(contagem_ranges) == len(esperado_astm)
        and all(k in contagem_ranges and np.isclose(contagem_ranges[k], v) for k, v in esperado_astm.items())
    )

    return {
        "onda_triangular": {
            "entrada": t_tri,
            "ciclos_extraidos": [{"delta_T_K": float(r), "peso": float(w)} for r, w in cic_tri],
            "total_ciclos": float(sum(w for _, w in cic_tri)),
            "passou": tri_ok,
        },
        "onda_senoidal": {
            "amplitude_declarada_K": 5.0,
            "range_pico_a_pico_esperado_K": 10.0,
            "ciclos_completos_extraidos_K": [float(r) for r in completos_sin],
            "total_ciclos_equivalentes": float(sum(w for _, w in cic_sin)),
            "passou": sin_ok,
        },
        "astm_e1049_85_secao_5_4_4": {
            "sequencia_referencia": pts_astm,
            "ciclos_individuais_extraidos": lista_detalhada,
            "multiconjunto_agregado_por_range": {str(k): v for k, v in sorted(contagem_ranges.items())},
            "multiconjunto_prescrito_norma": {str(k): v for k, v in sorted(esperado_astm.items())},
            "soma_total_ciclos": float(sum(contagem_ranges.values())),
            "conformidade_exata_astm": astm_ok,
        },
    }


def validar_delimitacao_black_coffin_manson_a16():
    """
    Verifica numericamente a invariância de escala das razões adimensionais de dano
    (Black e Coffin-Manson) frente a constantes desconhecidas de geometria de interconexão
    e coeficientes materiais absolutos.
    """
    rng = np.random.default_rng(101)
    nt = 1000
    dt = 0.5
    # Dois perfis sintéticos de teste (Baseline mais quente vs Proposto mais frio)
    rb = {
        "T": 78.0 + 6.0 * np.sin(np.linspace(0, 8 * np.pi, nt))[:, None] + rng.normal(0, 0.2, (nt, 4)),
        "f": np.full((nt, 4), 1.20),
        "Pn": np.full((nt, 4), 18.0),
    }
    rp = {
        "T": 71.0 + 3.2 * np.sin(np.linspace(0, 8 * np.pi, nt))[:, None] + rng.normal(0, 0.2, (nt, 4)),
        "f": np.full((nt, 4), 1.05),
        "Pn": np.full((nt, 4), 13.5),
    }
    sc_mock = {"dt": dt}
    rel_nom = C.confiabilidade(rb, rp, sc_mock, skip_h=0.0, limiares=(0.25, 1.0, 2.0))

    # Verifica que multiplicar jref (ou a área efetiva A_eff da malha de potência) por um fator arbitrário alpha > 0
    # não altera em nada Black_razao_soma nem Black_razao_pior_no
    jref_1 = float((rb["Pn"] / C.tensao(rb["f"])).mean())
    jref_scaled = 3.75 * jref_1
    b_b1 = C.dano_black(rb, jref_1, dt)
    b_p1 = C.dano_black(rp, jref_1, dt)
    b_b2 = C.dano_black(rb, jref_scaled, dt)
    b_p2 = C.dano_black(rp, jref_scaled, dt)
    invariancia_area_black = float(abs((b_b1.sum() / b_p1.sum()) - (b_b2.sum() / b_p2.sum())))

    # Verifica que usar semi-amplitude (Delta_T / 2)^q vs range completo (Delta_T)^q produz exatamente a mesma razão CM
    cic_b = C.rainflow(rb["T"][:, 0], 1.0)
    cic_p = C.rainflow(rp["T"][:, 0], 1.0)
    razao_range_q21 = sum(n * (dT ** 2.1) for dT, n in cic_b) / sum(n * (dT ** 2.1) for dT, n in cic_p)
    razao_amp_q21 = sum(n * ((dT / 2.0) ** 2.1) for dT, n in cic_b) / sum(n * ((dT / 2.0) ** 2.1) for dT, n in cic_p)
    invariancia_range_vs_amplitude_cm = float(abs(razao_range_q21 - razao_amp_q21))

    return {
        "delimitacao_metodologica": {
            "natureza_das_metricas": "Índices comparativos relativos adimensionais (D_baseline / D_proposto)",
            "proibicao_interpretativa": (
                "Os valores reportados NÃO representam tempo absoluto até falha (MTTF em horas ou anos), "
                "pois as constantes geométricas e metalúrgicas específicas do encapsulamento (A_Black, A_interconnect, B_CM) "
                "cancelam-se na razão relativa."
            ),
            "hipotese_proxy_corrente_black": (
                "A corrente de alimentação de cada domínio de core é modelada por I_i(t) = P_i(t) / V(f_i(t)) [A]. "
                "Assumindo seção transversal condutiva efetiva constante A_eff nas trilhas de alimentação VDD de cada die, "
                "a densidade de corrente normalizada J_i / J_ref = I_i / I_ref é exata."
            ),
            "parametros_adotados": {
                "black_expoente_corrente_n": 1.8,
                "black_energia_ativacao_Ea_eV": 0.70,
                "coffin_manson_expoentes_q": [1.9, 2.1],
                "rainflow_limiares_histerese_K": [0.25, 1.0, 2.0],
            },
        },
        "verificacao_invariancia_algebrica": {
            "erro_invariancia_escala_area_black": invariancia_area_black,
            "erro_invariancia_range_vs_semiamplitude_cm": invariancia_range_vs_amplitude_cm,
            "invariancia_exata_verificada": bool(
                invariancia_area_black < 1e-14 and invariancia_range_vs_amplitude_cm < 1e-14
            ),
            "exemplo_metricas_relativas": rel_nom,
        },
    }


def main():
    print("=" * 78)
    print("  VALIDAÇÃO DE RAINFLOW E DELIMITAÇÃO DE CONFIABILIDADE (FASE 6: A15, A16)")
    print("=" * 78)

    res_a15 = validar_rainflow_a15()
    res_a16 = validar_delimitacao_black_coffin_manson_a16()

    assert res_a15["onda_triangular"]["passou"]
    assert res_a15["onda_senoidal"]["passou"]
    assert res_a15["astm_e1049_85_secao_5_4_4"]["conformidade_exata_astm"]
    assert res_a16["verificacao_invariancia_algebrica"]["invariancia_exata_verificada"]

    print(f"  [A15 Rainflow] Triangular OK: {res_a15['onda_triangular']['passou']} | "
          f"Senoidal OK: {res_a15['onda_senoidal']['passou']} | "
          f"ASTM E1049-85 Seção 5.4.4 OK: {res_a15['astm_e1049_85_secao_5_4_4']['conformidade_exata_astm']} "
          f"(soma ciclos = {res_a15['astm_e1049_85_secao_5_4_4']['soma_total_ciclos']})")
    print(f"  [A16 Black/CM] Invariância algébrica verificada: "
          f"err_Black={res_a16['verificacao_invariancia_algebrica']['erro_invariancia_escala_area_black']:.2e}, "
          f"err_CM={res_a16['verificacao_invariancia_algebrica']['erro_invariancia_range_vs_semiamplitude_cm']:.2e}")

    out_dir = os.path.join(ROOT, "audit", "numerical_validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reliability_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fase": "Fase 6 - Validacao de Rainflow e Delimitacao de Confiabilidade (A15, A16)",
                "A15_validacao_rainflow_astm_e1049": res_a15,
                "A16_delimitacao_black_e_coffin_manson": res_a16,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  [OK] Relatório salvo em: {out_path}")


if __name__ == "__main__":
    main()
