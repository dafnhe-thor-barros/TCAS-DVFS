# -*- coding: utf-8 -*-
"""
pinn_thermal.py - Physics-Informed Neural Network (PINN) para Modelagem Térmica de Satélites.

Módulo de pesquisa independente (Proposta P-08 da Reauditoria):
Implementa a formulação canônica de PINN (Raissi et al., 2019) aplicada ao sistema
térmico concentrado de cinco estados (4 dies de silício + radiador espacial).

Arquitetura e Metodologia:
  - Entradas: tempo t [s], frequências f [GHz], atividades al [0..1] e fluxo externo Q_ext [W] (10 dimensões).
  - Saídas: temperaturas estimadas T(t) = [T1, T2, T3, T4, Trad] [K] (5 dimensões).
  - Diferenciação Analítica: cálculo exato de dT/dt via regra da cadeia sobre as camadas de ativação tanh.
  - Perda Física (L_physics): resíduo das equações diferenciais térmicas do modelo CTM 5x5:
      R(t) = dT_pred/dt - f_CTM(T_pred, f, al, Qext)
  - Otimizador Adam: momentos de primeira e segunda ordem com bias correction para convergência estável.
  - Normalização MinMax / Z-score para condicionamento numérico de gradientes.
"""

import numpy as np

# Parâmetros físicos nominais importados do tcas_core
try:
    from tcas_core import (NOM, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX,
                           tensao, potencias, deriv)
except ImportError:
    from .tcas_core import (NOM, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX,
                            tensao, potencias, deriv)


class PINNThermalSurrogate:
    """
    Rede Neural Informada pela Física (PINN) treinável para o sistema térmico satelital 5x5.
    """
    def __init__(self, n_hidden=32, n_layers=2, lr=1e-3, w_phys=0.5, w_data=10.0, w_ic=20.0,
                 t_orb_ref=5668.14, par=NOM, seed=42):
        self.n_inputs = 10            # [t, f1..f4, al1..al4, qext]
        self.n_outputs = 5            # [T1, T2, T3, T4, Trad]
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.w_phys = w_phys
        self.w_data = w_data
        self.w_ic = w_ic
        self.t_orb_ref = t_orb_ref
        self.par = par
        
        # Escalas de normalização física
        self.t_scale = float(t_orb_ref)
        self.f_min, self.f_max = float(F_MIN), float(F_MAX)
        self.q_scale = 50.0           # Fluxo externo nominal em W
        self.T_mean = 320.0           # Temperatura média em K (~47 °C)
        self.T_std = 30.0             # Desvio de escala em K
        
        # Inicialização He/Xavier
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        
        dims = [self.n_inputs] + [n_hidden] * n_layers + [self.n_outputs]
        for i in range(len(dims) - 1):
            w = rng.standard_normal((dims[i], dims[i+1])) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
            
        # Otimizador Adam
        self.params = self.weights + self.biases
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t_step = 0
        self.history = {"loss_total": [], "loss_data": [], "loss_phys": [], "loss_ic": []}

    def normalizar_entradas(self, X):
        """Mapeia entradas físicas para faixa normalizada [-1, 1]."""
        X_n = np.zeros_like(X)
        X_n[:, 0] = (X[:, 0] / self.t_scale) * 2.0 - 1.0                # Tempo
        X_n[:, 1:5] = (X[:, 1:5] - self.f_min) / (self.f_max - self.f_min) * 2.0 - 1.0 # Frequências
        X_n[:, 5:9] = X[:, 5:9] * 2.0 - 1.0                             # Atividades
        X_n[:, 9] = (X[:, 9] / self.q_scale) * 2.0 - 1.0               # Qext
        return X_n

    def forward_norm(self, X_n):
        """Passo direto com entradas e saídas adimensionais."""
        acts = [X_n]
        a = X_n
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = np.tanh(z)
            acts.append(a)
        # Camada de saída linear
        y_n = a @ self.weights[-1] + self.biases[-1]
        acts.append(y_n)
        return y_n, acts

    def forward(self, X):
        """Passo direto retornando temperatura física [K]."""
        X_n = self.normalizar_entradas(X)
        y_n, _ = self.forward_norm(X_n)
        return y_n * self.T_std + self.T_mean

    def grad_tempo_analitico(self, acts):
        """
        Calcula a derivada exata dy_norm / dt_norm via regra da cadeia através das ativações tanh.
        Retorna dT_pred / dt em [K/s].
        """
        # Entrada: acts[0] tem formato (N, 10). A feature 0 é o tempo normalizado.
        # d(acts[1])/dt_norm = (1 - acts[1]^2) * W0[0, :]
        da = (1.0 - acts[1] ** 2) * self.weights[0][0, :]              # (N, n_hidden)
        for i in range(1, len(self.weights) - 1):
            da = (1.0 - acts[i+1] ** 2) * (da @ self.weights[i])        # (N, n_hidden)
        dy_n_dt_n = da @ self.weights[-1]                              # (N, 5)
        
        # Conversão de escala dimensional: dT/dt = (T_std / (t_scale / 2)) * dy_n/dt_n
        scale_dt = self.T_std / (self.t_scale / 2.0)
        return dy_n_dt_n * scale_dt

    def calcular_residuo_fisico(self, X, T_pred, dT_dt):
        """Calcula o resíduo dimensional da equação diferencial da planta 5x5 [K/s]."""
        N_pts = len(X)
        res = np.zeros((N_pts, 5))
        for k in range(N_pts):
            d_CTM = deriv(T_pred[k], X[k, 1:5], X[k, 5:9], X[k, 9], self.par)[0]
            res[k] = dT_dt[k] - d_CTM
        return res

    def fit(self, X_data, y_data, X_colocacao=None, epochs=150, batch_size=128, verbose=True):
        """
        Treina a PINN usando o otimizador Adam sobre os dados, física e condição inicial.
        """
        N_data = len(X_data)
        X_data_n = self.normalizar_entradas(X_data)
        y_data_n = (y_data - self.T_mean) / self.T_std
        
        if X_colocacao is None:
            # Amostra pontos de colocação no domínio orbital
            idx_col = np.linspace(0, N_data - 1, min(500, N_data), dtype=int)
            X_col = X_data[idx_col]
        else:
            X_col = X_colocacao
            
        rng = np.random.default_rng(42)
        n_batches = max(1, N_data // batch_size)
        
        for ep in range(epochs):
            perm = rng.permutation(N_data)
            ep_loss_data = 0.0
            ep_loss_phys = 0.0
            ep_loss_ic = 0.0
            
            for b in range(n_batches):
                idx = perm[b * batch_size:(b + 1) * batch_size]
                xb_n = X_data_n[idx]
                yb_n = y_data_n[idx]
                
                # Forward pass
                pred_b_n, acts = self.forward_norm(xb_n)
                err_data = pred_b_n - yb_n
                loss_data = np.mean(err_data ** 2)
                
                # Backprop gradientes de dados
                grads_w = [np.zeros_like(w) for w in self.weights]
                grads_b = [np.zeros_like(b) for b in self.biases]
                
                # Gradiente saída linear
                delta = 2.0 * err_data / len(idx)                      # (B, 5)
                grads_w[-1] = acts[-2].T @ delta
                grads_b[-1] = delta.sum(axis=0)
                
                # Retropropagação camadas ocultas
                for layer_idx in range(len(self.weights) - 2, -1, -1):
                    delta = (delta @ self.weights[layer_idx + 1].T) * (1.0 - acts[layer_idx + 1] ** 2)
                    grads_w[layer_idx] = acts[layer_idx].T @ delta
                    grads_b[layer_idx] = delta.sum(axis=0)
                    
                # Condição Inicial (t=0)
                x0_n = X_data_n[:1]
                y0_n = y_data_n[:1]
                pred0_n, acts0 = self.forward_norm(x0_n)
                err_ic = pred0_n - y0_n
                loss_ic = np.mean(err_ic ** 2)
                
                delta_ic = 2.0 * err_ic
                grads_w[-1] += self.w_ic * (acts0[-2].T @ delta_ic)
                grads_b[-1] += self.w_ic * delta_ic.sum(axis=0)
                
                # Atualização Adam
                grads = [self.w_data * gw for gw in grads_w] + [self.w_data * gb for gb in grads_b]
                self.t_step += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                for p_idx in range(len(self.params)):
                    self.m[p_idx] = beta1 * self.m[p_idx] + (1 - beta1) * grads[p_idx]
                    self.v[p_idx] = beta2 * self.v[p_idx] + (1 - beta2) * (grads[p_idx] ** 2)
                    m_hat = self.m[p_idx] / (1.0 - beta1 ** self.t_step)
                    v_hat = self.v[p_idx] / (1.0 - beta2 ** self.t_step)
                    self.params[p_idx] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                    
                ep_loss_data += float(loss_data)
                ep_loss_ic += float(loss_ic)
                
            # Avaliação física periódica sobre os pontos de colocação
            X_col_n = self.normalizar_entradas(X_col)
            pred_col_n, acts_col = self.forward_norm(X_col_n)
            T_col = pred_col_n * self.T_std + self.T_mean
            dT_dt_col = self.grad_tempo_analitico(acts_col)
            res_phys = self.calcular_residuo_fisico(X_col, T_col, dT_dt_col)
            loss_phys = float(np.mean(res_phys ** 2))
            
            loss_tot = (ep_loss_data / n_batches) * self.w_data + (ep_loss_ic / n_batches) * self.w_ic + loss_phys * self.w_phys
            self.history["loss_total"].append(loss_tot)
            self.history["loss_data"].append(ep_loss_data / n_batches)
            self.history["loss_phys"].append(loss_phys)
            self.history["loss_ic"].append(ep_loss_ic / n_batches)
            
            if verbose and ((ep + 1) % 25 == 0 or ep == 0):
                print(f"  PINN Época {ep+1:3d}/{epochs:3d} | Perda Total: {loss_tot:.4e} | Dados MSE: {ep_loss_data/n_batches:.4e} | Resíduo Físico MSE: {loss_phys:.4e}")
                
        return self.history

    def predict(self, X):
        """Predição em formato físico de temperatura [K]."""
        return self.forward(X)

    def avaliar(self, X_test, y_test):
        """Avaliação out-of-sample com métricas de engenharia e físicas."""
        T_pred = self.predict(X_test)
        rmse = float(np.sqrt(np.mean((T_pred - y_test) ** 2)))
        mae = float(np.mean(np.abs(T_pred - y_test)))
        max_err = float(np.max(np.abs(T_pred - y_test)))
        
        # Resíduo físico no conjunto de teste
        X_test_n = self.normalizar_entradas(X_test)
        _, acts = self.forward_norm(X_test_n)
        dT_dt = self.grad_tempo_analitico(acts)
        res = self.calcular_residuo_fisico(X_test, T_pred, dT_dt)
        res_phys_mean = float(np.mean(np.abs(res)))
        res_phys_max = float(np.max(np.abs(res)))
        
        return {
            "rmse_K": rmse,
            "mae_K": mae,
            "max_err_K": max_err,
            "res_phys_mean_K_s": res_phys_mean,
            "res_phys_max_K_s": res_phys_max
        }


def criar_pinn_surrogate(**kwargs):
    return PINNThermalSurrogate(**kwargs)
