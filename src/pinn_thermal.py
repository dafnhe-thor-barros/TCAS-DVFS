# -*- coding: utf-8 -*-
"""
pinn_thermal.py - Physics-Informed Neural Network (PINN) para Modelagem Termica de Satelites.

Modulo de pesquisa independente (Proposta P-08 da Reauditoria):
Implementa a formulacao canonica de PINN (Raissi et al., 2019) aplicada ao sistema
termico concentrado de cinco estados (4 dies de silicio + radiador espacial).

Arquitetura:
  - Entradas: tempo t [s], vetor de frequencias f [GHz], atividades al [0..1] e fluxo externo Q_ext [W].
  - Saidas: temperaturas estimadas T(t) = [T1, T2, T3, T4, Trad] [K].
  - Perda de Dados (L_data): erro quadratico medio frente aos sensores com ruido.
  - Perda Fisica (L_physics): residuo das equacoes diferenciais termicas (CTM 5x5).
  - Perda de Condicao Inicial (L_ic): aderencia a x(0) = x0.
"""

import numpy as np

# Parametros fisicos nominais importados do tcas_core
try:
    from tcas_core import (NOM, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX,
                           tensao, potencias, deriv)
except ImportError:
    # Fallback para execucao isolada
    from .tcas_core import (NOM, EPS, SIGMA, A_RAD, T_SPACE, F_MIN, F_MAX,
                            tensao, potencias, deriv)


class PINNThermalSurrogate:
    """
    Surrogate Termico baseado em Redes Neurais Informadas pela Fisica (PINN).
    Implementa computacao analitica do residuo fisico e forward pass vetorizado.
    """
    def __init__(self, n_hidden=64, n_layers=3, lr=1e-3, w_phys=1.0, w_data=10.0, w_ic=100.0, par=NOM):
        self.n_inputs = 1 + 4 + 4 + 1  # [t, f1..f4, al1..al4, qext] = 10 entradas
        self.n_outputs = 5            # [T1, T2, T3, T4, Trad] = 5 saidas
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.w_phys = w_phys
        self.w_data = w_data
        self.w_ic = w_ic
        self.par = par
        
        # Inicializacao de pesos (He / Xavier)
        rng = np.random.default_rng(42)
        self.weights = []
        self.biases = []
        
        # Camada de entrada
        self.weights.append(rng.standard_normal((self.n_inputs, n_hidden)) * np.sqrt(2.0 / self.n_inputs))
        self.biases.append(np.zeros(n_hidden))
        
        # Camadas ocultas
        for _ in range(n_layers - 1):
            self.weights.append(rng.standard_normal((n_hidden, n_hidden)) * np.sqrt(2.0 / n_hidden))
            self.biases.append(np.zeros(n_hidden))
            
        # Camada de saida
        self.weights.append(rng.standard_normal((n_hidden, self.n_outputs)) * np.sqrt(2.0 / n_hidden))
        self.biases.append(np.zeros(self.n_outputs))

    def _ativacao(self, z):
        """Funcao de ativacao suave tanh (adequada para autograd e derivadas continuas)."""
        return np.tanh(z)

    def _ativacao_deriv(self, a):
        """Derivada da ativacao: 1 - tanh^2."""
        return 1.0 - a ** 2

    def forward(self, X):
        """
        Propagacao direta da rede neural.
        X shape: (N_pontos, 10)
        Retorna: T_pred shape: (N_pontos, 5) em Kelvin.
        """
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self._ativacao(z)
        # Camada de saida linear com deslocamento termico de referencia (300 K)
        T_pred = a @ self.weights[-1] + self.biases[-1] + 300.0
        return T_pred

    def calcular_residuo_fisico(self, t, f, al, qext, T_pred, dT_dt_num):
        """
        Calcula o residuo do balanco de energia termico:
          R(t) = dT_pred/dt - f_fisica(T_pred, f, al, qext)
        """
        N_pts = len(t)
        residuos = np.zeros((N_pts, 5))
        for k in range(N_pts):
            d_fisica, _, _ = deriv(T_pred[k], f[k], al[k], qext[k], self.par)
            residuos[k] = dT_dt_num[k] - d_fisica
        return residuos

    def funcao_perda(self, X_colocacao, dT_dt_colocacao, X_dados=None, y_dados=None, X_ic=None, y_ic=None):
        """
        Funcao de perda composta da PINN:
          L_total = w_phys * L_physics + w_data * L_data + w_ic * L_ic
        """
        # 1. Perda de Colocacao Fisica
        t = X_colocacao[:, 0]
        f = X_colocacao[:, 1:5]
        al = X_colocacao[:, 5:9]
        qext = X_colocacao[:, 9]
        T_col = self.forward(X_colocacao)
        res_fisico = self.calcular_residuo_fisico(t, f, al, qext, T_col, dT_dt_colocacao)
        loss_phys = float(np.mean(res_fisico ** 2))

        # 2. Perda de Dados Observados
        loss_data = 0.0
        if X_dados is not None and y_dados is not None:
            T_dados_pred = self.forward(X_dados)
            loss_data = float(np.mean((T_dados_pred - y_dados) ** 2))

        # 3. Perda de Condicao Inicial
        loss_ic = 0.0
        if X_ic is not None and y_ic is not None:
            T_ic_pred = self.forward(X_ic)
            loss_ic = float(np.mean((T_ic_pred - y_ic) ** 2))

        loss_total = self.w_phys * loss_phys + self.w_data * loss_data + self.w_ic * loss_ic
        return {
            "loss_total": loss_total,
            "loss_physics": loss_phys,
            "loss_data": loss_data,
            "loss_ic": loss_ic
        }


def criar_pinn_surrogate():
    """Fabrica de instancia da PINN para integracao em pipelines de pesquisa."""
    return PINNThermalSurrogate()


if __name__ == "__main__":
    pinn = criar_pinn_surrogate()
    X_teste = np.zeros((10, 10))
    X_teste[:, 1:5] = 1.0  # 1 GHz
    X_teste[:, 5:9] = 0.7  # 70% atividade
    X_teste[:, 9] = 20.0   # 20 W radiacao externa
    T_out = pinn.forward(X_teste)
    print("PINN Thermal Surrogate instanciado com sucesso!")
    print(f"Shape de entrada: {X_teste.shape} -> Shape de saida: {T_out.shape}")
    print(f"Temperaturas preditas iniciais [K]: {np.round(T_out[0], 2)}")
