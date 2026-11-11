# -*- coding: utf-8 -*-
"""
pinn_thermal.py - Physics-Informed Neural Network (PINN) para Modelagem Térmica de Satélites.

Reconstrução Matemática Completa (Fase 1 — A01 a A05):
  - Função objetivo composta:
      L(theta) = w_data * L_data(theta) + w_phys * L_phys(theta) + w_ic * L_IC(theta)
  - Diferenciação analítica exata de 1ª e 2ª ordem mista:
      * dT_hat/dt via regra da cadeia direta através das camadas tanh;
      * Retropropagação completa (backpropagation) de L_phys para TODOS os pesos e biases
        de todas as camadas, acoplando d(dT_hat/dt)/d(theta) e o Jacobiano analítico
        da planta térmica df_CTM/dT;
      * Retropropagação completa de L_IC para todas as camadas ocultas e de saída;
  - Processamento integral de lotes (sem descarte do último mini-lote incompleto);
  - Separação estrita entre pontos de supervisão observados (X_data) e pontos de colocação física (X_colocacao);
  - Rastreamento diagnóstico de perdas, normas de gradientes por parcela, evolução de parâmetros e tempo.
"""

import math
import time
import numpy as np

try:
    from tcas_core import (NOM, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX,
                           EA, KB, T0, tensao, potencias, deriv, jacobiano)
except ImportError:
    from .tcas_core import (NOM, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX,
                            EA, KB, T0, tensao, potencias, deriv, jacobiano)


def gerar_pontos_colocacao_independentes(X_dominio, indices_supervisionados, n_col=600, seed=42):
    """
    Gera pontos de colocação física estritamente disjuntos dos pontos usados na supervisão (A04).
    Além de selecionar índices não observados, perturba continuamente os instantes de tempo
    para pontos intermediários da malha orbital.
    """
    N_total = len(X_dominio)
    sup_set = set(int(i) for i in indices_supervisionados)
    candidatos = np.array([i for i in range(N_total - 1) if i not in sup_set], dtype=int)
    rng = np.random.default_rng(seed)
    if len(candidatos) >= n_col:
        escolhidos = np.sort(rng.choice(candidatos, size=n_col, replace=False))
    else:
        escolhidos = candidatos
    # Interpolação em pontos intermediários (t + tau * dt) para garantir disjunção contínua
    tau = rng.uniform(0.15, 0.85, size=(len(escolhidos), 1))
    X_col = (1.0 - tau) * X_dominio[escolhidos] + tau * X_dominio[escolhidos + 1]
    return X_col, escolhidos


class PINNThermalSurrogate:
    """
    Rede Neural Informada pela Física (PINN) com retropropagação analítica exata
    do resíduo diferencial CTM 5x5 e da condição inicial em todas as camadas.
    """
    def __init__(self, n_hidden=32, n_layers=2, lr=2e-3, w_phys=1.0, w_data=1.0, w_ic=5.0,
                 t_orb_ref=5668.144369, par=NOM, seed=42):
        self.n_inputs = 10            # [t, f1..f4, al1..al4, qext]
        self.n_outputs = 5            # [T1, T2, T3, T4, Trad]
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        self.lr = float(lr)
        self.w_phys = float(w_phys)
        self.w_data = float(w_data)
        self.w_ic = float(w_ic)
        self.t_orb_ref = float(t_orb_ref)
        self.par = par
        self.seed = int(seed)

        # Escalas de normalização física
        self.t_scale = float(t_orb_ref)
        self.f_min, self.f_max = float(F_MIN), float(F_MAX)
        self.q_scale = 50.0
        self.T_mean = 320.0
        self.T_std = 30.0
        self.alpha_t = 2.0 / self.t_scale  # dx_n0 / dt

        # Inicialização Xavier/Glorot para ativação tanh
        rng = np.random.default_rng(self.seed)
        self.weights = []
        self.biases = []
        dims = [self.n_inputs] + [self.n_hidden] * self.n_layers + [self.n_outputs]
        for i in range(len(dims) - 1):
            lim = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights.append(rng.normal(0.0, lim, size=(dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1], dtype=float))

        # Cópia inicial para medir evolução dos parâmetros ||theta - theta_0||_2
        self._init_vec = self.vetor_parametros().copy()

        # Estado do otimizador Adam
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t_step = 0

        self.history = {
            "loss_total": [],
            "loss_data": [],
            "loss_phys": [],
            "loss_ic": [],
            "grad_norm_total": [],
            "grad_norm_data": [],
            "grad_norm_phys": [],
            "grad_norm_ic": [],
            "param_norm": [],
            "param_delta_norm": [],
            "epoch_time_s": [],
            "samples_seen": []
        }

    def vetor_parametros(self):
        """Concatena todos os pesos e biases em um vetor 1D."""
        return np.concatenate([w.ravel() for w in self.weights] + [b.ravel() for b in self.biases])

    @staticmethod
    def _norma_grad(gW, gb):
        s = sum(float(np.sum(w * w)) for w in gW) + sum(float(np.sum(b * b)) for b in gb)
        return math.sqrt(s)

    def normalizar_entradas(self, X):
        """Mapeia entradas físicas para [-1, 1]."""
        X = np.asarray(X, dtype=float)
        X_n = np.empty_like(X)
        X_n[:, 0] = (X[:, 0] / self.t_scale) * 2.0 - 1.0
        X_n[:, 1:5] = (X[:, 1:5] - self.f_min) / (self.f_max - self.f_min) * 2.0 - 1.0
        X_n[:, 5:9] = X[:, 5:9] * 2.0 - 1.0
        X_n[:, 9] = (X[:, 9] / self.q_scale) * 2.0 - 1.0
        return X_n

    def forward_with_cache(self, X):
        """
        Propagação direta e propagação tangente temporal (dT_pred/dt analítico).
        Retorna T_pred [K], dT_dt [K/s] e cache completo para retropropagação de 1ª e 2ª ordem.
        """
        X_n = self.normalizar_entradas(X)
        N = X_n.shape[0]
        a_list = [X_n]
        da_list = [np.zeros_like(X_n)]
        da_list[0][:, 0] = self.alpha_t

        z_list, dz_list, sp_list, spp_list = [], [], [], []
        n_hidden_layers = len(self.weights) - 1

        for l in range(n_hidden_layers):
            z = a_list[l] @ self.weights[l] + self.biases[l]
            dz = da_list[l] @ self.weights[l]
            al1 = np.tanh(z)
            sp = 1.0 - al1 * al1
            spp = -2.0 * al1 * sp
            dal1 = sp * dz

            z_list.append(z)
            dz_list.append(dz)
            sp_list.append(sp)
            spp_list.append(spp)
            a_list.append(al1)
            da_list.append(dal1)

        y_n = a_list[-1] @ self.weights[-1] + self.biases[-1]
        dy_n = da_list[-1] @ self.weights[-1]

        T_pred = self.T_mean + self.T_std * y_n
        dT_dt = self.T_std * dy_n

        cache = {
            "a": a_list,
            "da": da_list,
            "z": z_list,
            "dz": dz_list,
            "sp": sp_list,
            "spp": spp_list,
            "y_n": y_n,
            "dy_n": dy_n,
            "T_pred": T_pred,
            "dT_dt": dT_dt
        }
        return T_pred, dT_dt, cache

    def forward(self, X):
        """Retorna predição de temperatura em Kelvin [K]."""
        T_pred, _, _ = self.forward_with_cache(X)
        return T_pred

    def _ctm_deriv_jacobiano_batch(self, T_pred, F, AL, QEXT):
        """
        Avalia f_CTM(T, f, al, qext) e o Jacobiano analítico df_CTM/dT em lote,
        mantendo equivalência exata com tcas_core.deriv e tcas_core.jacobiano.
        """
        N = T_pred.shape[0]
        T = T_pred[:, :4]
        Tr = T_pred[:, 4]
        pd, pl = potencias(T, F, AL)
        p = pd + pl

        c = float(self.par["C"])
        rr = float(self.par["RR"])
        crad = float(self.par["CRAD"])
        Lp = np.asarray(self.par["LAP"], dtype=float)

        qr = (T - Tr[:, None]) / rr
        f_ctm = np.empty((N, 5), dtype=float)
        f_ctm[:, :4] = (p - (T @ Lp.T) - qr) / c
        f_ctm[:, 4] = (qr.sum(axis=1) + QEXT - EPS * SIGMA * A_RAD * (Tr ** 4 - T_SPACE ** 4)) / crad

        Tc = np.clip(T, 200.0, 500.0)
        dentro = (T > 200.0) & (T < 500.0)
        dpl_dT = np.where(dentro, pl * (2.0 / Tc + EA / (KB * (Tc ** 2))), 0.0)

        J = np.zeros((N, 5, 5), dtype=float)
        J[:, :4, :4] = ((-Lp - np.eye(4) / rr) / c)[None, :, :]
        idx4 = np.arange(4)
        J[:, idx4, idx4] += dpl_dT / c
        J[:, :4, 4] = 1.0 / (rr * c)
        J[:, 4, :4] = 1.0 / (crad * rr)
        J[:, 4, 4] = -4.0 / (rr * crad) - 4.0 * EPS * SIGMA * A_RAD * (Tr ** 3) / crad
        return f_ctm, J

    def calcular_residuo_fisico(self, X_col, T_pred=None, dT_dt=None):
        """Calcula o resíduo físico R(t) = dT_pred/dt - f_CTM(T_pred, f, al, qext) [K/s]."""
        if T_pred is None or dT_dt is None:
            T_pred, dT_dt, _ = self.forward_with_cache(X_col)
        f_ctm, _ = self._ctm_deriv_jacobiano_batch(T_pred, X_col[:, 1:5], X_col[:, 5:9], X_col[:, 9])
        return dT_dt - f_ctm

    def _backprop_mse(self, cache, err_K):
        """
        Retropropaga uma perda MSE de temperatura L = mean(err_K^2) através de TODAS as camadas.
        Usado tanto para L_data quanto para L_IC (garantindo A02).
        """
        n_layers_total = len(self.weights)
        gW = [np.zeros_like(w) for w in self.weights]
        gb = [np.zeros_like(b) for b in self.biases]

        N = err_K.shape[0]
        bar_yn = (2.0 / (5.0 * N)) * err_K * self.T_std

        gW[-1] = cache["a"][-1].T @ bar_yn
        gb[-1] = bar_yn.sum(axis=0)
        bar_a = bar_yn @ self.weights[-1].T

        for l in range(n_layers_total - 2, -1, -1):
            bar_z = bar_a * cache["sp"][l]
            gW[l] = cache["a"][l].T @ bar_z
            gb[l] = bar_z.sum(axis=0)
            if l > 0:
                bar_a = bar_z @ self.weights[l].T

        return gW, gb

    def _backprop_physics(self, X_col, cache):
        """
        Retropropaga a perda física L_phys = mean((dT_dt - f_CTM)^2) para TODOS os pesos e biases
        de todas as camadas da rede (garantindo A01).
        """
        n_layers_total = len(self.weights)
        gW = [np.zeros_like(w) for w in self.weights]
        gb = [np.zeros_like(b) for b in self.biases]

        T_pred = cache["T_pred"]
        dT_dt = cache["dT_dt"]
        N_c = T_pred.shape[0]

        f_ctm, J_all = self._ctm_deriv_jacobiano_batch(T_pred, X_col[:, 1:5], X_col[:, 5:9], X_col[:, 9])
        R = dT_dt - f_ctm
        loss_phys = float(np.mean(R * R))

        G_R = (2.0 / (5.0 * N_c)) * R
        bar_dyn = self.T_std * G_R
        bar_yn = -self.T_std * np.einsum('ki,kij->kj', G_R, J_all)

        # Camada de saída
        gW[-1] = cache["a"][-1].T @ bar_yn + cache["da"][-1].T @ bar_dyn
        gb[-1] = bar_yn.sum(axis=0)
        bar_a = bar_yn @ self.weights[-1].T
        bar_da = bar_dyn @ self.weights[-1].T

        # Camadas ocultas (derivadas de 1ª e 2ª ordem da ativação tanh)
        for l in range(n_layers_total - 2, -1, -1):
            sp = cache["sp"][l]
            spp = cache["spp"][l]
            dz = cache["dz"][l]

            bar_dz = bar_da * sp
            bar_z = bar_a * sp + bar_da * dz * spp

            gW[l] = cache["a"][l].T @ bar_z + cache["da"][l].T @ bar_dz
            gb[l] = bar_z.sum(axis=0)
            if l > 0:
                bar_a = bar_z @ self.weights[l].T
                bar_da = bar_dz @ self.weights[l].T

        return loss_phys, gW, gb

    def calcular_perdas_e_gradientes(self, X_data, y_data, X_colocacao, X_ic=None, y_ic=None):
        """
        Calcula L_data, L_phys, L_ic, L_total e os gradientes analíticos exatos de cada parcela
        em todas as camadas.
        """
        n_layers_total = len(self.weights)

        # 1. Parcela de Dados Supervisionados (L_data)
        T_d, _, cache_d = self.forward_with_cache(X_data)
        err_d = T_d - y_data
        loss_data = float(np.mean(err_d * err_d))
        gW_d, gb_d = self._backprop_mse(cache_d, err_d)

        # 2. Parcela de Condição Inicial (L_IC) com backpropagation completo até a camada 0
        if X_ic is None or y_ic is None:
            X_ic = X_data[:1]
            y_ic = y_data[:1]
        T_ic, _, cache_ic = self.forward_with_cache(X_ic)
        err_ic = T_ic - y_ic
        loss_ic = float(np.mean(err_ic * err_ic))
        gW_ic, gb_ic = self._backprop_mse(cache_ic, err_ic)

        # 3. Parcela Física (L_phys) nos pontos de colocação
        _, _, cache_col = self.forward_with_cache(X_colocacao)
        loss_phys, gW_p, gb_p = self._backprop_physics(X_colocacao, cache_col)

        # 4. Combinação ponderada L_total = w_data * L_data + w_phys * L_phys + w_ic * L_ic
        loss_total = self.w_data * loss_data + self.w_phys * loss_phys + self.w_ic * loss_ic
        gW_tot = [
            self.w_data * gW_d[l] + self.w_phys * gW_p[l] + self.w_ic * gW_ic[l]
            for l in range(n_layers_total)
        ]
        gb_tot = [
            self.w_data * gb_d[l] + self.w_phys * gb_p[l] + self.w_ic * gb_ic[l]
            for l in range(n_layers_total)
        ]

        return {
            "loss_total": loss_total,
            "loss_data": loss_data,
            "loss_phys": loss_phys,
            "loss_ic": loss_ic,
            "grads_total": (gW_tot, gb_tot),
            "grads_data": (gW_d, gb_d),
            "grads_phys": (gW_p, gb_p),
            "grads_ic": (gW_ic, gb_ic),
            "grad_norm_total": self._norma_grad(gW_tot, gb_tot),
            "grad_norm_data": self._norma_grad(gW_d, gb_d),
            "grad_norm_phys": self._norma_grad(gW_p, gb_p),
            "grad_norm_ic": self._norma_grad(gW_ic, gb_ic)
        }

    def funcao_perda(self, X_colocacao, X_dados, y_dados, X_ic=None, y_ic=None):
        """Avalia apenas os escalares de perda (útil para verificação por diferenças finitas)."""
        T_d, _, _ = self.forward_with_cache(X_dados)
        loss_data = float(np.mean((T_d - y_dados) ** 2))

        if X_ic is None or y_ic is None:
            X_ic = X_dados[:1]
            y_ic = y_dados[:1]
        T_ic, _, _ = self.forward_with_cache(X_ic)
        loss_ic = float(np.mean((T_ic - y_ic) ** 2))

        T_col, dT_dt_col, _ = self.forward_with_cache(X_colocacao)
        res_phys = self.calcular_residuo_fisico(X_colocacao, T_col, dT_dt_col)
        loss_phys = float(np.mean(res_phys ** 2))

        loss_total = self.w_data * loss_data + self.w_phys * loss_phys + self.w_ic * loss_ic
        return {
            "loss_total": loss_total,
            "loss_physics": loss_phys,
            "loss_data": loss_data,
            "loss_ic": loss_ic
        }

    def _step_adam(self, gW, gb, grad_clip=50.0):
        """Atualiza todos os pesos e biases usando Adam com clipping defensivo de norma."""
        norma = self._norma_grad(gW, gb)
        if norma > grad_clip:
             escala = grad_clip / norma
             gW = [w * escala for w in gW]
             gb = [b * escala for b in gb]

        self.t_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - beta1 ** self.t_step
        bc2 = 1.0 - beta2 ** self.t_step

        for l in range(len(self.weights)):
            self.m_w[l] = beta1 * self.m_w[l] + (1.0 - beta1) * gW[l]
            self.v_w[l] = beta2 * self.v_w[l] + (1.0 - beta2) * (gW[l] * gW[l])
            mw_hat = self.m_w[l] / bc1
            vw_hat = self.v_w[l] / bc2
            self.weights[l] -= self.lr * mw_hat / (np.sqrt(vw_hat) + eps)

            self.m_b[l] = beta1 * self.m_b[l] + (1.0 - beta1) * gb[l]
            self.v_b[l] = beta2 * self.v_b[l] + (1.0 - beta2) * (gb[l] * gb[l])
            mb_hat = self.m_b[l] / bc1
            vb_hat = self.v_b[l] / bc2
            self.biases[l] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def fit(self, X_data, y_data, X_colocacao=None, X_ic=None, y_ic=None,
            epochs=150, batch_size=128, col_batch_size=256, verbose=True):
        """
        Treina a PINN garantindo:
          - Retropropagação de L_data, L_phys e L_IC em todas as camadas (A01, A02);
          - Processamento de 100% das amostras a cada época sem descartar o último lote (A03);
          - Pontos de colocação independentes dos dados observados (A04).
        """
        X_data = np.asarray(X_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        N_data = len(X_data)

        if X_ic is None or y_ic is None:
            X_ic = X_data[:1].copy()
            y_ic = y_data[:1].copy()

        if X_colocacao is None:
            # Gera pontos de colocação em instantes intermediários disjuntos de X_data
            rng_col = np.random.default_rng(self.seed + 100)
            tau = rng_col.uniform(0.2, 0.8, size=(max(1, N_data - 1), 1))
            X_colocacao = (1.0 - tau) * X_data[:-1] + tau * X_data[1:]
        else:
            X_colocacao = np.asarray(X_colocacao, dtype=float)

        N_col = len(X_colocacao)
        rng = np.random.default_rng(self.seed)
        t_start_total = time.perf_counter()

        for ep in range(epochs):
            t_ep0 = time.perf_counter()
            perm_d = rng.permutation(N_data)
            perm_c = rng.permutation(N_col)

            # Lotes completos incluindo o último lote residual (A03)
            starts_d = list(range(0, N_data, batch_size))
            n_batches = len(starts_d)

            ep_loss_tot = 0.0
            ep_loss_d = 0.0
            ep_loss_p = 0.0
            ep_loss_ic = 0.0
            ep_gn_tot = 0.0
            ep_gn_d = 0.0
            ep_gn_p = 0.0
            ep_gn_ic = 0.0
            samples_seen_ep = 0

            for b_idx, start_d in enumerate(starts_d):
                end_d = min(start_d + batch_size, N_data)
                idx_d = perm_d[start_d:end_d]
                samples_seen_ep += len(idx_d)

                # Seleciona mini-lote de colocação física independente
                start_c = (b_idx * col_batch_size) % N_col
                end_c = min(start_c + col_batch_size, N_col)
                idx_c = perm_c[start_c:end_c]
                if len(idx_c) < min(col_batch_size, N_col):
                    faltam = min(col_batch_size, N_col) - len(idx_c)
                    idx_c = np.concatenate([idx_c, perm_c[:faltam]])

                info_step = self.calcular_perdas_e_gradientes(
                    X_data[idx_d], y_data[idx_d], X_colocacao[idx_c], X_ic, y_ic
                )
                gW_tot, gb_tot = info_step["grads_total"]
                self._step_adam(gW_tot, gb_tot)

                peso_lote = len(idx_d) / float(N_data)
                ep_loss_tot += info_step["loss_total"] * peso_lote
                ep_loss_d += info_step["loss_data"] * peso_lote
                ep_loss_p += info_step["loss_phys"] * peso_lote
                ep_loss_ic += info_step["loss_ic"] * peso_lote
                ep_gn_tot += info_step["grad_norm_total"] / n_batches
                ep_gn_d += info_step["grad_norm_data"] / n_batches
                ep_gn_p += info_step["grad_norm_phys"] / n_batches
                ep_gn_ic += info_step["grad_norm_ic"] / n_batches

            assert samples_seen_ep == N_data, f"Erro A03: processou {samples_seen_ep} != {N_data}"

            cur_vec = self.vetor_parametros()
            p_norm = float(np.linalg.norm(cur_vec))
            p_delta = float(np.linalg.norm(cur_vec - self._init_vec))
            ep_time = time.perf_counter() - t_ep0

            self.history["loss_total"].append(float(ep_loss_tot))
            self.history["loss_data"].append(float(ep_loss_d))
            self.history["loss_phys"].append(float(ep_loss_p))
            self.history["loss_ic"].append(float(ep_loss_ic))
            self.history["grad_norm_total"].append(float(ep_gn_tot))
            self.history["grad_norm_data"].append(float(ep_gn_d))
            self.history["grad_norm_phys"].append(float(ep_gn_p))
            self.history["grad_norm_ic"].append(float(ep_gn_ic))
            self.history["param_norm"].append(p_norm)
            self.history["param_delta_norm"].append(p_delta)
            self.history["epoch_time_s"].append(float(ep_time))
            self.history["samples_seen"].append(int(samples_seen_ep))

            if verbose and ((ep + 1) % 25 == 0 or ep == 0):
                print(
                    f"  Época {ep+1:3d}/{epochs:3d} | L_tot: {ep_loss_tot:.4e} "
                    f"| L_data: {ep_loss_d:.4e} | L_phys: {ep_loss_p:.4e} | L_ic: {ep_loss_ic:.4e} "
                    f"| ||g_p||: {ep_gn_p:.3e} | ||g_ic||: {ep_gn_ic:.3e}"
                )

        self.total_train_time_s = time.perf_counter() - t_start_total
        return self.history

    def predict(self, X):
        """Predição em Kelvin [K]."""
        return self.forward(X)

    def avaliar(self, X_test, y_test):
        """Avalia métricas térmicas e resíduo diferencial físico no conjunto de teste."""
        T_pred, dT_dt, _ = self.forward_with_cache(X_test)
        err = T_pred - y_test
        rmse = float(np.sqrt(np.mean(err * err)))
        mae = float(np.mean(np.abs(err)))
        max_err = float(np.max(np.abs(err)))

        res = self.calcular_residuo_fisico(X_test, T_pred, dT_dt)
        res_phys_mse = float(np.mean(res * res))
        res_phys_rms = float(np.sqrt(res_phys_mse))
        res_phys_mean = float(np.mean(np.abs(res)))
        res_phys_max = float(np.max(np.abs(res)))

        return {
            "rmse_K": rmse,
            "mae_K": mae,
            "max_err_K": max_err,
            "res_phys_mse_K2_s2": res_phys_mse,
            "res_phys_rms_K_s": res_phys_rms,
            "res_phys_mean_K_s": res_phys_mean,
            "res_phys_max_K_s": res_phys_max
        }


def criar_pinn_surrogate(**kwargs):
    return PINNThermalSurrogate(**kwargs)
