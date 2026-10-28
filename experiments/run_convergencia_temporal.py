# -*- coding: utf-8 -*-
"""
run_convergencia_temporal.py - Estudo de Convergência Temporal do Integrador (Heun / RK2).

Avalia o impacto do passo de discretizacao temporal da planta (dt in [1.0, 0.5, 0.25, 0.125] s)
mantendo a taxa de controle do MPC invariante em dt_MPC = 2.0 s (ksub = round(2.0 / dt)).
Registra o erro relativo e absoluto de T_max, delta_T_peak e Energia frente a referencia mais fina (dt = 0.125 s).
"""

import sys, os
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
import os
import sys
import csv
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import tcas_core as C
import tcas_dvfs as D


def avaliar_convergencia(n_orb=1, semente=0):
    passos_dt = [1.0, 0.5, 0.25, 0.125]
    resultados = []
    
    print("======================================================================")
    print(f"  ESTUDO DE CONVERGÊNCIA TEMPORAL DA PLANTA (Heun RK2, {n_orb} Órbita(s))")
    print("======================================================================")
    print(f"  Passos avaliados: {passos_dt} segundos")
    print(f"  Taxa de controle MPC fixada em dt_MPC = 2.0 s")
    
    for dt in passos_dt:
        ksub = int(round(2.0 / dt))
        sc = C.cenario(n_orb=n_orb, dt=dt)
        gov = D.gov_tcas_dvfs(ksub=ksub)
        
        t0 = time.time()
        r = C.simular(sc, gov, seed=semente)
        t_exec = time.time() - t0
        
        m = C.metricas(r, sc)
        
        res_item = {
            "dt_s": dt,
            "ksub": ksub,
            "n_steps": sc["nt"],
            "T_max": m["Tmax"],
            "T_med": m["Tmed"],
            "dT_pico": m["dT_pico"],
            "dT_med": m["dT_med"],
            "t_viol_s": m["t_viol_s"],
            "FPS": m["FPS"],
            "E_kJ": m["E_kJ"],
            "EDP": m["EDP"],
            "t_exec_s": t_exec
        }
        resultados.append(res_item)
        print(f"  dt = {dt:5.3f} s (N={sc['nt']:6d}) -> T_max = {m['Tmax']:.4f} °C, dT_pico = {m['dT_pico']:.4f} K, E = {m['E_kJ']:.2f} kJ ({t_exec:.2f} s)")

    # Calcular erros relativos e absolutos frente à referência mais fina (dt = 0.125 s)
    ref = resultados[-1] # dt = 0.125 s
    print("\n----------------------------------------------------------------------")
    print(f"  ERROS FRENTE À REFERÊNCIA MAIS FINA (dt = {ref['dt_s']} s):")
    print("----------------------------------------------------------------------")
    print(f"  {'dt (s)':<8} {'|Err T_max| (K)':<18} {'|Err dT_pico| (K)':<18} {'Err Rel Energia (%)':<20}")
    
    for item in resultados:
        err_tmax = abs(item["T_max"] - ref["T_max"])
        err_dt_pico = abs(item["dT_pico"] - ref["dT_pico"])
        err_rel_e = abs(item["E_kJ"] - ref["E_kJ"]) / ref["E_kJ"] * 100.0
        
        item["err_Tmax_abs_K"] = err_tmax
        item["err_dTpico_abs_K"] = err_dt_pico
        item["err_rel_E_pct"] = err_rel_e
        
        print(f"  {item['dt_s']:<8.3f} {err_tmax:<18.4f} {err_dt_pico:<18.4f} {err_rel_e:<20.4f}%")
        
    return resultados


def salvar_resultados(resultados, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_csv = os.path.join(pasta_saida, "convergencia_temporal.csv")
    caminho_json = os.path.join(pasta_saida, "convergencia_temporal.json")
    
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(resultados)
        
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
        
    print(f"\n  Resultados salvos com sucesso em:")
    print(f"  - {caminho_csv}")
    print(f"  - {caminho_json}")


if __name__ == "__main__":
    pasta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pasta_resultados = os.path.join(pasta_raiz, "results")
    
    res = avaliar_convergencia(n_orb=1, semente=0)
    salvar_resultados(res, pasta_resultados)
    print("======================================================================")
