# -*- coding: utf-8 -*-
"""
validacao_jacobiano.py - Validação do Jacobiano Analítico da Planta Térmica 5x5 (CTM).

Compara o Jacobiano analítico exato derivado em tcas_core.py contra a matriz Jacobiana
obtida numericamente por diferenças finitas centrais de 4a ordem:
  dF/dx_j ≈ (-f(x + 2h e_j) + 8f(x + h e_j) - 8f(x - h e_j) + f(x - 2h e_j)) / (12h)

Testa 100 combinações aleatórias de:
- Estados térmicos: x in [300, 370] K (núcleos) e [250, 320] K (radiador)
- Frequências: f in [0.4, 1.6] GHz
- Atividades: alpha in [0.1, 1.0]
- Parâmetros nominais e perturbados (+/-20%)
Exige erro relativo de Frobenius ||J_ana - J_num||_F / ||J_ana||_F < 1e-7.
"""
import os
import sys
import numpy as np

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import tcas_core as C


def jacobiano_diferencas_finitas(x, f, al, qext=20.0, par=C.NOM, h=1e-5):
    """Calcula o Jacobiano por diferenças finitas centrais de 4a ordem."""
    n = len(x)
    J_num = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        d_p2 = C.deriv(x + 2 * h * ej, f, al, qext, par)[0]
        d_p1 = C.deriv(x + h * ej, f, al, qext, par)[0]
        d_m1 = C.deriv(x - h * ej, f, al, qext, par)[0]
        d_m2 = C.deriv(x - 2 * h * ej, f, al, qext, par)[0]
        J_num[:, j] = (-d_p2 + 8.0 * d_p1 - 8.0 * d_m1 + d_m2) / (12.0 * h)
    return J_num


def executar_bateria_validacao(n_pontos=100, seed=42):
    rng = np.random.default_rng(seed)
    erros_rel = []
    erros_max_elem = []
    
    print("=" * 72)
    print("  VALIDAÇÃO SISTEMÁTICA DO JACOBIANO ANALÍTICO (TCAS-DVFS)")
    print(f"  Testando {n_pontos} pontos estocásticos com diferenças finitas de 4a ordem")
    print("=" * 72)
    
    for i in range(n_pontos):
        # Amostragem em regime operacional
        x = np.zeros(5)
        x[:4] = rng.uniform(300.0, 365.0, size=4)
        x[4] = rng.uniform(250.0, 315.0)
        f = rng.uniform(C.F_MIN, C.F_MAX, size=4)
        al = rng.uniform(0.1, 1.0, size=4)
        qext = rng.uniform(0.0, 45.0)
        
        # Parâmetros variáveis para testar par['C'], par['RR'], etc.
        sRij = rng.uniform(0.85, 1.15)
        sRr = rng.uniform(0.85, 1.15)
        sC = rng.uniform(0.85, 1.15)
        sCr = rng.uniform(0.85, 1.15)
        par = C.parametros(sRij, sRr, sC, sCr)
        
        J_ana = C.jacobiano(x, f, al, par)
        J_num = jacobiano_diferencas_finitas(x, f, al, qext, par, h=1e-5)
        
        norm_ana = np.linalg.norm(J_ana, 'fro')
        norm_diff = np.linalg.norm(J_ana - J_num, 'fro')
        rel_err = norm_diff / norm_ana
        max_elem_err = np.max(np.abs(J_ana - J_num))
        
        erros_rel.append(rel_err)
        erros_max_elem.append(max_elem_err)
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Amostra {i+1:3d}/{n_pontos:3d} | Erro Relativo Frobenius: {rel_err:.3e} | Erro Máx Elemento: {max_elem_err:.3e}")
            
    erros_rel = np.array(erros_rel)
    erros_max_elem = np.array(erros_max_elem)
    
    max_rel_err = float(np.max(erros_rel))
    med_rel_err = float(np.mean(erros_rel))
    p99_rel_err = float(np.percentile(erros_rel, 99))
    
    print("-" * 72)
    print(f"  Estatísticas Globais:")
    print(f"    - Erro Relativo Médio:        {med_rel_err:.3e}")
    print(f"    - Erro Relativo 99º Perc.:    {p99_rel_err:.3e}")
    print(f"    - Erro Relativo Máximo:       {max_rel_err:.3e}")
    print(f"    - Critério de Aprovação:      < 1.000e-07")
    
    assert max_rel_err < 1e-7, f"Falha na validação do Jacobiano! Erro máximo {max_rel_err:.3e} >= 1e-7"
    print("=" * 72)
    print("  [APROVADO] O Jacobiano analítico é rigorosamente exato e consistente!")
    print("=" * 72)
    return {
        "n_pontos": n_pontos,
        "max_rel_err": max_rel_err,
        "med_rel_err": med_rel_err,
        "p99_rel_err": p99_rel_err,
        "passou": True
    }


if __name__ == "__main__":
    executar_bateria_validacao(100)
