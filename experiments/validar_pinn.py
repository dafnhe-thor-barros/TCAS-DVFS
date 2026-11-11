# -*- coding: utf-8 -*-
"""
validar_pinn.py - Validação Experimental da PINN vs MLP Supervisionada (Fase 1: A01-A05).

Executa:
1. Geração de 3 órbitas da planta térmica não linear (Heun 2ª ordem):
   - Órbita 0: spin-up para regime periódico orbital;
   - Órbita 1: domínio de treinamento com supervisão esparsa ruidosa (60 sensores, sigma = 0.6 K)
     e 500 pontos de colocação física estritamente disjuntos (A04);
   - Órbita 2: conjunto de teste estritamente Out-of-Sample (2.835 pontos).
2. Comparação sob idênticas condições iniciais, dados e orçamento de treinamento (A05)
   ao longo da fronteira de Pareto físico-empírica:
     - Modelo A: MLP Supervisionada Pura (w_phys = 0.0 s^2)
     - Modelo B: PINN Equilibrada (w_phys = 20.0 s^2)
     - Modelo C: PINN Fortemente Física (w_phys = 80.0 s^2)
3. Registro individual de loss_data, loss_phys, loss_ic, normas de gradientes de cada
   parcela, evolução dos parâmetros e tempo de treinamento.
"""

import os
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_src = os.path.join(_root, 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

import tcas_core as C
import pinn_thermal as P


def executar_validacao_pinn(n_epochs=120, salvar_figura=True):
    print("=" * 76)
    print("  VALIDAÇÃO CIENTÍFICA PINN vs MLP SUPERVISIONADA (FASE 1: A01-A05)")
    print("=" * 76)

    dt = 2.0
    sc = C.cenario(n_orb=3, dt=dt)
    nt = sc["nt"]
    t = sc["t"]
    t_orb = sc["t_orb"]
    nt_orb = int(round(t_orb / dt))

    acao, _ = C.gov_tca()
    x_gt = np.zeros((nt, 5))
    x_gt[0] = sc["x0"]
    f_hist = np.zeros((nt, 4))
    al_hist = sc["al"].copy()

    for k in range(nt - 1):
        f_k = acao(k, x_gt[k], sc)
        f_hist[k] = f_k
        d1 = C.deriv(x_gt[k], f_k, al_hist[k], sc["qext"][k], C.NOM)[0]
        x_e = x_gt[k] + dt * d1
        d2 = C.deriv(x_e, f_k, al_hist[k + 1], sc["qext"][k + 1], C.NOM)[0]
        x_gt[k + 1] = x_gt[k] + 0.5 * dt * (d1 + d2)
    f_hist[-1] = f_hist[-2]

    t_local = np.mod(t, t_orb)
    X_full = np.column_stack([t_local, f_hist, al_hist, sc["qext"]])

    # Órbita 1 (regime periódico) para treino; Órbita 2 para teste out-of-sample
    X_orb1 = X_full[nt_orb:2 * nt_orb]
    y_orb1_gt = x_gt[nt_orb:2 * nt_orb]
    X_orb2 = X_full[2 * nt_orb:]
    y_orb2_gt = x_gt[2 * nt_orb:]

    # Supervisão esparsa na Órbita 1 (60 observações com ruído sigma = 0.6 K)
    rng = np.random.default_rng(100)
    idx_sup = np.linspace(0, len(X_orb1) - 1, 60, dtype=int)
    X_train_sup = X_orb1[idx_sup]
    y_train_sup = y_orb1_gt[idx_sup] + rng.normal(0.0, 0.6, size=(len(idx_sup), 5))
    X_ic = X_orb1[:1].copy()
    y_ic = y_orb1_gt[:1].copy()

    # Pontos de colocação física estritamente disjuntos dos pontos supervisionados (A04)
    X_col_ind, idx_col_base = P.gerar_pontos_colocacao_independentes(
        X_orb1, idx_sup, n_col=500, seed=42
    )
    intersecao = set(idx_sup.tolist()).intersection(set(idx_col_base.tolist()))
    assert len(intersecao) == 0, "Falha A04: pontos de colocação interceptam supervisão!"

    print(f"  Supervisão Esparsa (Órbita 1): {len(X_train_sup)} amostras (sigma = 0.60 K)")
    print(f"  Colocação Física Independente: {len(X_col_ind)} pontos disjuntos (interseção = {len(intersecao)})")
    print(f"  Teste Out-of-Sample (Órbita 2): {len(X_orb2)} amostras")

    # 1. MLP Supervisionada Pura (w_phys = 0.0)
    print("\n  [1/3] Treinando MLP Supervisionada Pura (w_phys = 0.0 s^2)...")
    mlp = P.criar_pinn_surrogate(
        n_hidden=32, n_layers=2, lr=2e-3,
        w_data=1.0, w_phys=0.0, w_ic=5.0,
        t_orb_ref=t_orb, seed=42
    )
    hist_mlp = mlp.fit(
        X_train_sup, y_train_sup, X_colocacao=X_col_ind, X_ic=X_ic, y_ic=y_ic,
        epochs=n_epochs, batch_size=32, col_batch_size=125, verbose=True
    )

    # 2. PINN Equilibrada (w_phys = 20.0)
    print("\n  [2/3] Treinando PINN Equilibrada (w_phys = 20.0 s^2)...")
    pinn_bal = P.criar_pinn_surrogate(
        n_hidden=32, n_layers=2, lr=2e-3,
        w_data=1.0, w_phys=20.0, w_ic=5.0,
        t_orb_ref=t_orb, seed=42
    )
    hist_bal = pinn_bal.fit(
        X_train_sup, y_train_sup, X_colocacao=X_col_ind, X_ic=X_ic, y_ic=y_ic,
        epochs=n_epochs, batch_size=32, col_batch_size=125, verbose=True
    )

    # 3. PINN Fortemente Física (w_phys = 80.0)
    print("\n  [3/3] Treinando PINN Fortemente Física (w_phys = 80.0 s^2)...")
    pinn_str = P.criar_pinn_surrogate(
        n_hidden=32, n_layers=2, lr=2e-3,
        w_data=1.0, w_phys=80.0, w_ic=5.0,
        t_orb_ref=t_orb, seed=42
    )
    hist_str = pinn_str.fit(
        X_train_sup, y_train_sup, X_colocacao=X_col_ind, X_ic=X_ic, y_ic=y_ic,
        epochs=n_epochs, batch_size=32, col_batch_size=125, verbose=True
    )

    # Avaliações comparativas
    ev_mlp_1 = mlp.avaliar(X_orb1, y_orb1_gt)
    ev_bal_1 = pinn_bal.avaliar(X_orb1, y_orb1_gt)
    ev_str_1 = pinn_str.avaliar(X_orb1, y_orb1_gt)

    ev_mlp_2 = mlp.avaliar(X_orb2, y_orb2_gt)
    ev_bal_2 = pinn_bal.avaliar(X_orb2, y_orb2_gt)
    ev_str_2 = pinn_str.avaliar(X_orb2, y_orb2_gt)

    mse_col_mlp = float(np.mean(mlp.calcular_residuo_fisico(X_col_ind) ** 2))
    mse_col_bal = float(np.mean(pinn_bal.calcular_residuo_fisico(X_col_ind) ** 2))
    mse_col_str = float(np.mean(pinn_str.calcular_residuo_fisico(X_col_ind) ** 2))

    red_bal_pct = (1.0 - mse_col_bal / mse_col_mlp) * 100.0
    red_str_pct = (1.0 - mse_col_str / mse_col_mlp) * 100.0

    # Diferença nos parâmetros finais aprendidos ||theta_PINN - theta_MLP||_2
    diff_param_bal = float(np.linalg.norm(pinn_bal.vetor_parametros() - mlp.vetor_parametros()))
    diff_param_str = float(np.linalg.norm(pinn_str.vetor_parametros() - mlp.vetor_parametros()))

    print("\n" + "=" * 86)
    print("  COMPARAÇÃO HEAD-TO-HEAD: MLP SUPERVISIONADA vs PINN (FASE 1: A05)")
    print("=" * 86)
    print(f"  Métrica                                | MLP (w_p=0)  | PINN (w_p=20) | PINN (w_p=80)")
    print(f"  ---------------------------------------+--------------+---------------+--------------")
    print(f"  MSE Físico Colocação Disjunta [K²/s²]  | {mse_col_mlp:12.4e} | {mse_col_bal:13.4e} | {mse_col_str:12.4e}")
    print(f"  Redução do Resíduo Físico (Colocação)  |     0.00%    |    {red_bal_pct:8.2f}%  |   {red_str_pct:8.2f}%")
    print(f"  Resíduo Físico RMS (Órbita 2) [K/s]    | {ev_mlp_2['res_phys_rms_K_s']:12.4e} | {ev_bal_2['res_phys_rms_K_s']:13.4e} | {ev_str_2['res_phys_rms_K_s']:12.4e}")
    print(f"  RMSE Temperatura (Órbita 1) [K]        | {ev_mlp_1['rmse_K']:12.3f} | {ev_bal_1['rmse_K']:13.3f} | {ev_str_1['rmse_K']:12.3f}")
    print(f"  RMSE Temperatura (Órbita 2) [K]        | {ev_mlp_2['rmse_K']:12.3f} | {ev_bal_2['rmse_K']:13.3f} | {ev_str_2['rmse_K']:12.3f}")
    print(f"  MAE Temperatura (Órbita 2) [K]         | {ev_mlp_2['mae_K']:12.3f} | {ev_bal_2['mae_K']:13.3f} | {ev_str_2['mae_K']:12.3f}")
    print(f"  Deslocamento ||theta - theta_MLP||_2   | {0.0:12.4f} | {diff_param_bal:13.4f} | {diff_param_str:12.4f}")
    print("=" * 86)

    assert mse_col_str < mse_col_bal < mse_col_mlp, "Monotonia física violada!"
    assert diff_param_bal > 0.1 and diff_param_str > 0.1, "Parâmetros não diferiram da MLP!"

    audit_pinn_dir = os.path.join(_root, "audit", "pinn")
    os.makedirs(audit_pinn_dir, exist_ok=True)

    csv_path = os.path.join(audit_pinn_dir, "pinn_training_curves.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "epoch",
            "mlp_loss_data", "mlp_loss_phys", "mlp_loss_ic",
            "pinn20_loss_total", "pinn20_loss_data", "pinn20_loss_phys", "pinn20_loss_ic",
            "pinn20_grad_norm_data", "pinn20_grad_norm_phys", "pinn20_grad_norm_ic",
            "pinn80_loss_total", "pinn80_loss_data", "pinn80_loss_phys", "pinn80_loss_ic",
            "pinn80_grad_norm_data", "pinn80_grad_norm_phys", "pinn80_grad_norm_ic",
            "pinn80_param_norm", "pinn80_param_delta_norm", "pinn80_epoch_time_s", "pinn80_samples_seen"
        ])
        for ep in range(n_epochs):
            writer.writerow([
                ep + 1,
                hist_mlp["loss_data"][ep], hist_mlp["loss_phys"][ep], hist_mlp["loss_ic"][ep],
                hist_bal["loss_total"][ep], hist_bal["loss_data"][ep], hist_bal["loss_phys"][ep], hist_bal["loss_ic"][ep],
                hist_bal["grad_norm_data"][ep], hist_bal["grad_norm_phys"][ep], hist_bal["grad_norm_ic"][ep],
                hist_str["loss_total"][ep], hist_str["loss_data"][ep], hist_str["loss_phys"][ep], hist_str["loss_ic"][ep],
                hist_str["grad_norm_data"][ep], hist_str["grad_norm_phys"][ep], hist_str["grad_norm_ic"][ep],
                hist_str["param_norm"][ep], hist_str["param_delta_norm"][ep], hist_str["epoch_time_s"][ep], hist_str["samples_seen"][ep]
            ])

    comp_data = {
        "training_config": {
            "n_epochs": n_epochs,
            "batch_size": 32,
            "col_batch_size": 125,
            "lr": 2e-3,
            "n_supervised_samples": int(len(X_train_sup)),
            "n_collocation_samples": int(len(X_col_ind)),
            "n_test_samples_orbit2": int(len(X_orb2)),
            "disjoint_collocation_verified": True,
            "w_data": 1.0,
            "w_ic": 5.0
        },
        "mlp_w_phys_0": {
            "orbit1_in_sample": ev_mlp_1,
            "orbit2_out_of_sample": ev_mlp_2,
            "collocation_phys_mse_K2_s2": mse_col_mlp,
            "train_time_s": float(mlp.total_train_time_s)
        },
        "pinn_balanced_w_phys_20": {
            "orbit1_in_sample": ev_bal_1,
            "orbit2_out_of_sample": ev_bal_2,
            "collocation_phys_mse_K2_s2": mse_col_bal,
            "physics_residual_reduction_pct": float(red_bal_pct),
            "param_diff_from_mlp_l2": diff_param_bal,
            "train_time_s": float(pinn_bal.total_train_time_s)
        },
        "pinn_strong_w_phys_80": {
            "orbit1_in_sample": ev_str_1,
            "orbit2_out_of_sample": ev_str_2,
            "collocation_phys_mse_K2_s2": mse_col_str,
            "physics_residual_reduction_pct": float(red_str_pct),
            "param_diff_from_mlp_l2": diff_param_str,
            "train_time_s": float(pinn_str.total_train_time_s)
        }
    }
    json_path = os.path.join(audit_pinn_dir, "pinn_vs_mlp_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)

    if salvar_figura:
        T_pred_bal = pinn_bal.predict(X_orb2)
        t_min_test = np.arange(len(X_orb2)) * dt / 60.0

        fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=False)
        cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        labels = ["Die 1", "Die 2", "Die 3", "Die 4", "Radiator"]

        for i in [0, 2, 4]:
            axes[0].plot(t_min_test, y_orb2_gt[:, i] - 273.15, color=cores[i], lw=1.3, label=f"{labels[i]} (Heun GT)")
            axes[0].plot(t_min_test, T_pred_bal[:, i] - 273.15, color=cores[i], ls="--", lw=1.2, label=f"{labels[i]} (PINN)")
        axes[0].set_xlabel("Out-of-Sample Orbit Time (min)")
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title(
            f"Out-of-Sample Validation (Orbit 2) | PINN RMSE = {ev_bal_2['rmse_K']:.2f} K, MAE = {ev_bal_2['mae_K']:.2f} K"
        )
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncol=3, fontsize=7.5)

        ep_axis = np.arange(1, n_epochs + 1)
        axes[1].semilogy(ep_axis, hist_mlp["loss_phys"], color="#d62728", ls=":", lw=1.5, label="Supervised MLP ($w_{phys}=0$) Collocation PDE MSE")
        axes[1].semilogy(ep_axis, hist_bal["loss_phys"], color="#ff7f0e", ls="--", lw=1.5, label=f"PINN ($w_{{phys}}=20$) PDE MSE (-{red_bal_pct:.1f}%)")
        axes[1].semilogy(ep_axis, hist_str["loss_phys"], color="#1f77b4", lw=1.6, label=f"PINN ($w_{{phys}}=80$) PDE MSE (-{red_str_pct:.1f}%)")
        axes[1].set_xlabel("Training Epoch")
        axes[1].set_ylabel("Collocation PDE Residual MSE ($K^2/s^2$)")
        axes[1].set_title("Active Physics Residual Minimization on Disjoint Collocation Points")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        for out_fig in [
            os.path.join(_root, "results", "fig_validacao_pinn.png"),
            os.path.join(_root, "figuras", "fig_validacao_pinn.png"),
            os.path.join(audit_pinn_dir, "fig_pinn_vs_mlp.png")
        ]:
            os.makedirs(os.path.dirname(out_fig), exist_ok=True)
            fig.savefig(out_fig, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Artefatos e figuras salvos em: {audit_pinn_dir}")

    return comp_data


if __name__ == "__main__":
    executar_validacao_pinn(n_epochs=120)
