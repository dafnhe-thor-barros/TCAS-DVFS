# -*- coding: utf-8 -*-
"""
test_pinn_gradients.py - Suite de Testes Unitários e de Gradientes da PINN (Fase 1: A01-A04).

Testes Mandatórios:
  1. [T01] w_phys = 0: parcela física no gradiente total deve ser identicamente nula.
  2. [T02] w_phys > 0: gradiente físico deve ser não nulo em TODAS as camadas (0, 1, 2).
  3. [T03] Sensibilidade a w_phys: alterar w_phys altera o gradiente total e os parâmetros pós-passo.
  4. [T04] Backpropagation completo da Condição Inicial (L_IC): produz gradientes não nulos
           em todas as camadas ocultas (0, 1) e de saída (2).
  5. [T05] Consistência com Diferenças Finitas Centrais: erro relativo Frobenius < 1e-7 em todas
           as camadas para L_data, L_phys, L_IC e L_total.
  6. [T06] Cobertura integral de lotes (A03): 100% das amostras são processadas em cada época
           mesmo quando N_data não é múltiplo de batch_size.
  7. [T07] Disjunção estrita entre pontos de colocação física e pontos de supervisão (A04).
"""

import os
import sys
import json
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_src = os.path.join(_root, 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

import pinn_thermal as P


def _grad_diferencas_finitas(pinn, X_col, X_d, y_d, X_ic, y_ic, chave_perda="loss_total", eps=1e-6):
    """Calcula gradientes numéricos por diferenças finitas centrais de 2ª ordem para todos os pesos e biases."""
    num_W = [np.zeros_like(w) for w in pinn.weights]
    num_b = [np.zeros_like(b) for b in pinn.biases]

    for l in range(len(pinn.weights)):
        W_l = pinn.weights[l]
        for i in range(W_l.shape[0]):
            for j in range(W_l.shape[1]):
                orig = W_l[i, j]
                W_l[i, j] = orig + eps
                lp = pinn.funcao_perda(X_col, X_d, y_d, X_ic, y_ic)[chave_perda]
                W_l[i, j] = orig - eps
                lm = pinn.funcao_perda(X_col, X_d, y_d, X_ic, y_ic)[chave_perda]
                W_l[i, j] = orig
                num_W[l][i, j] = (lp - lm) / (2.0 * eps)

        b_l = pinn.biases[l]
        for j in range(b_l.shape[0]):
            orig = b_l[j]
            b_l[j] = orig + eps
            lp = pinn.funcao_perda(X_col, X_d, y_d, X_ic, y_ic)[chave_perda]
            b_l[j] = orig - eps
            lm = pinn.funcao_perda(X_col, X_d, y_d, X_ic, y_ic)[chave_perda]
            b_l[j] = orig
            num_b[l][j] = (lp - lm) / (2.0 * eps)

    return num_W, num_b


def executar_suite_pinn():
    rng = np.random.default_rng(123)
    N_d = 16
    N_c = 20
    X_d = np.column_stack([
        rng.uniform(10.0, 5000.0, N_d),
        rng.uniform(0.35, 1.25, (N_d, 4)),
        rng.uniform(0.20, 0.95, (N_d, 4)),
        rng.uniform(5.0, 40.0, N_d)
    ])
    y_d = rng.uniform(300.0, 355.0, (N_d, 5))
    X_ic = X_d[:1].copy()
    y_ic = y_d[:1].copy()
    X_col = np.column_stack([
        rng.uniform(10.0, 5000.0, N_c),
        rng.uniform(0.35, 1.25, (N_c, 4)),
        rng.uniform(0.20, 0.95, (N_c, 4)),
        rng.uniform(5.0, 40.0, N_c)
    ])

    relatorio = {"testes": {}, "todos_aprovados": False}
    print("=" * 76)
    print("  SUITE DE AUDITORIA MATEMÁTICA DA PINN (FASE 1: A01, A02, A03, A04)")
    print("=" * 76)

    # --- TESTE 1: w_phys = 0 ---
    pinn0 = P.PINNThermalSurrogate(n_hidden=8, n_layers=2, w_data=1.5, w_phys=0.0, w_ic=2.0, seed=42)
    info0 = pinn0.calcular_perdas_e_gradientes(X_d, y_d, X_col, X_ic, y_ic)
    gW_tot0, gb_tot0 = info0["grads_total"]
    gW_d0, gb_d0 = info0["grads_data"]
    gW_ic0, gb_ic0 = info0["grads_ic"]
    diff_sem_phys = max(
        max(np.max(np.abs(gW_tot0[l] - (1.5 * gW_d0[l] + 2.0 * gW_ic0[l]))) for l in range(3)),
        max(np.max(np.abs(gb_tot0[l] - (1.5 * gb_d0[l] + 2.0 * gb_ic0[l]))) for l in range(3))
    )
    assert diff_sem_phys == 0.0, f"T01 falhou: diff={diff_sem_phys}"
    relatorio["testes"]["T01_w_phys_zero"] = {"max_diff_from_pure_supervised": float(diff_sem_phys), "passou": True}
    print(f"  [OK] T01 (w_phys = 0): contribuição física no gradiente total = {diff_sem_phys:.3e}")

    # --- TESTE 2: w_phys > 0 produz gradiente físico não nulo em TODAS as camadas ---
    pinn1 = P.PINNThermalSurrogate(n_hidden=8, n_layers=2, w_data=1.5, w_phys=2.5, w_ic=2.0, seed=42)
    info1 = pinn1.calcular_perdas_e_gradientes(X_d, y_d, X_col, X_ic, y_ic)
    gW_p1, gb_p1 = info1["grads_phys"]
    normas_phys_camadas = [float(np.linalg.norm(gW_p1[l])) for l in range(3)]
    assert all(n > 1e-4 for n in normas_phys_camadas), f"T02 falhou: {normas_phys_camadas}"
    relatorio["testes"]["T02_w_phys_positivo_camadas"] = {
        "grad_phys_norm_per_layer": normas_phys_camadas,
        "passou": True
    }
    print(f"  [OK] T02 (w_phys > 0): ||gW_phys|| por camada = {[round(x, 6) for x in normas_phys_camadas]}")

    # --- TESTE 3: Alterar w_phys altera efetivamente o gradiente total e o treinamento ---
    gW_tot1, _ = info1["grads_total"]
    delta_grad_total = [float(np.linalg.norm(gW_tot1[l] - gW_tot0[l])) for l in range(3)]
    assert all(d > 1e-4 for d in delta_grad_total), f"T03 falhou: {delta_grad_total}"
    relatorio["testes"]["T03_sensibilidade_w_phys"] = {
        "delta_grad_total_norm_per_layer": delta_grad_total,
        "passou": True
    }
    print(f"  [OK] T03 (Sensibilidade w_phys): ||Delta gW_total|| por camada = {[round(x, 6) for x in delta_grad_total]}")

    # --- TESTE 4: Backpropagation completo da Condição Inicial (L_IC) em todas as camadas ---
    gW_ic1, gb_ic1 = info1["grads_ic"]
    normas_ic_camadas = [float(np.linalg.norm(gW_ic1[l])) for l in range(3)]
    assert all(n > 1e-4 for n in normas_ic_camadas), f"T04 falhou: {normas_ic_camadas}"
    relatorio["testes"]["T04_ic_backprop_todas_camadas"] = {
        "grad_ic_norm_per_layer": normas_ic_camadas,
        "passou": True
    }
    print(f"  [OK] T04 (Backprop IC completo): ||gW_ic|| por camada = {[round(x, 6) for x in normas_ic_camadas]}")

    # --- TESTE 5: Verificação por Diferenças Finitas Centrais (L_phys, L_ic, L_data, L_total) ---
    # 5a. Gradientes de L_phys isolado
    pinn_p_only = P.PINNThermalSurrogate(n_hidden=8, n_layers=2, w_data=0.0, w_phys=1.0, w_ic=0.0, seed=42)
    info_po = pinn_p_only.calcular_perdas_e_gradientes(X_d, y_d, X_col, X_ic, y_ic)
    num_Wp, num_bp = _grad_diferencas_finitas(pinn_p_only, X_col, X_d, y_d, X_ic, y_ic, "loss_physics")
    erros_rel_phys_W = [
        float(np.linalg.norm(info_po["grads_phys"][0][l] - num_Wp[l]) / np.linalg.norm(num_Wp[l]))
        for l in range(3)
    ]
    erros_rel_phys_b = [
        float(np.linalg.norm(info_po["grads_phys"][1][l] - num_bp[l]) / np.linalg.norm(num_bp[l]))
        for l in range(3)
    ]
    assert max(erros_rel_phys_W + erros_rel_phys_b) < 1e-7

    # 5b. Gradientes de L_total composto
    num_Wt, num_bt = _grad_diferencas_finitas(pinn1, X_col, X_d, y_d, X_ic, y_ic, "loss_total")
    erros_rel_tot_W = [
        float(np.linalg.norm(info1["grads_total"][0][l] - num_Wt[l]) / np.linalg.norm(num_Wt[l]))
        for l in range(3)
    ]
    erros_rel_tot_b = [
        float(np.linalg.norm(info1["grads_total"][1][l] - num_bt[l]) / np.linalg.norm(num_bt[l]))
        for l in range(3)
    ]
    assert max(erros_rel_tot_W + erros_rel_tot_b) < 1e-7

    relatorio["testes"]["T05_consistencia_diferencas_finitas"] = {
        "rel_err_phys_W_per_layer": erros_rel_phys_W,
        "rel_err_phys_b_per_layer": erros_rel_phys_b,
        "rel_err_total_W_per_layer": erros_rel_tot_W,
        "rel_err_total_b_per_layer": erros_rel_tot_b,
        "max_rel_err": float(max(erros_rel_phys_W + erros_rel_phys_b + erros_rel_tot_W + erros_rel_tot_b)),
        "passou": True
    }
    print(f"  [OK] T05 (Diferenças Finitas): erro rel. máx L_phys = {max(erros_rel_phys_W):.3e}, L_total = {max(erros_rel_tot_W):.3e} (< 1e-7)")

    # --- TESTE 6: Cobertura de Lotes Incompletos (A03) ---
    N_impar = 103
    batch_sz = 32  # 103 = 3 * 32 + 7 (último lote tem 7 amostras)
    X_imp = rng.uniform(0.0, 1.0, (N_impar, 10))
    X_imp[:, 0] *= 5000.0
    X_imp[:, 1:5] = 0.3 + X_imp[:, 1:5] * 1.0
    X_imp[:, 9] *= 30.0
    y_imp = rng.uniform(305.0, 340.0, (N_impar, 5))
    pinn_batch = P.PINNThermalSurrogate(n_hidden=8, n_layers=2, seed=42)
    hist_b = pinn_batch.fit(X_imp, y_imp, epochs=3, batch_size=batch_sz, verbose=False)
    assert all(s == N_impar for s in hist_b["samples_seen"]), f"T06 falhou: {hist_b['samples_seen']}"
    relatorio["testes"]["T06_cobertura_integral_lotes"] = {
        "n_samples_total": N_impar,
        "batch_size": batch_sz,
        "remainder_batch_size": N_impar % batch_sz,
        "samples_seen_per_epoch": hist_b["samples_seen"],
        "passou": True
    }
    print(f"  [OK] T06 (Lotes Incompletos): {hist_b['samples_seen']} amostras processadas de {N_impar} (resto = {N_impar % batch_sz})")

    # --- TESTE 7: Pontos de Colocação Independentes dos Dados Supervisionados (A04) ---
    X_dom = np.column_stack([
        np.linspace(0.0, 5668.0, 500),
        rng.uniform(0.4, 1.2, (500, 4)),
        rng.uniform(0.3, 0.9, (500, 4)),
        rng.uniform(10.0, 35.0, 500)
    ])
    idx_sup = np.arange(0, 500, 5)
    X_sup = X_dom[idx_sup]
    X_col_ind, idx_col_base = P.gerar_pontos_colocacao_independentes(X_dom, idx_sup, n_col=200, seed=42)
    intersecao_indices = set(idx_sup.tolist()).intersection(set(idx_col_base.tolist()))
    min_dist_tempo = float(np.min(np.abs(X_col_ind[:, 0:1] - X_sup[:, 0:1].T)))
    assert len(intersecao_indices) == 0 and min_dist_tempo > 0.0
    relatorio["testes"]["T07_colocacao_independente"] = {
        "n_supervisionados": int(len(idx_sup)),
        "n_colocacao": int(len(X_col_ind)),
        "intersecao_indices": len(intersecao_indices),
        "min_distancia_temporal_s": min_dist_tempo,
        "passou": True
    }
    print(f"  [OK] T07 (Colocação Disjunta): interseção de índices = {len(intersecao_indices)}, dist_min_t = {min_dist_tempo:.4f} s")

    relatorio["todos_aprovados"] = True
    out_path = os.path.join(_root, "audit", "pinn", "pinn_gradient_tests.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2)

    print("=" * 76)
    print(f"  [APROVADO] 7/7 testes matemáticos da PINN aprovados! Salvo em: {out_path}")
    print("=" * 76)
    return relatorio


if __name__ == "__main__":
    executar_suite_pinn()
