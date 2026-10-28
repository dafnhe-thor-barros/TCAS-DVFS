# -*- coding: utf-8 -*-
"""
gerar_metadata.py - Gera metadata.json e configs/nominal.json para conformidade e reprodutibilidade (Seção 19).
"""

import os
import sys
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import matplotlib

def obter_git_hash(pasta):
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        res = subprocess.run(cmd, cwd=pasta, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unversioned_local_build_20261028"

def coletar_metadados(pasta_raiz):
    git_hash = obter_git_hash(pasta_raiz)
    
    # Informações de Hardware e SO
    sistema_operacional = f"{platform.system()} {platform.release()} ({platform.version()})"
    processador = platform.processor() or platform.machine()
    
    metadados = {
        "project": "TCAS-DVFS: Thermal-Crosstalk-Aware Scheduling and Model Predictive DVFS for Spaceborne AI",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "operating_system": sistema_operacional,
            "architecture": platform.architecture()[0],
            "processor": processador,
        },
        "version_control": {
            "vcs": "git",
            "git_commit_hash": git_hash,
            "repository": "https://github.com/dafnhe-thor-barros/TCAS-DVFS"
        },
        "simulation_parameters": {
            "orbit_scenario": "500-km circular Sun-Synchronous Orbit (synthetic)",
            "altitude_km": 500.0,
            "orbital_period_s": 5668.144369,
            "total_orbits_nominal": 3,
            "total_simulation_duration_s": 17004.0,
            "time_samples": 34009,
            "dt_plant_s": 0.5,
            "dt_mpc_s": 2.0,
            "ksub": 4,
            "dt_sched_s": 120.0,
            "prediction_horizon_steps": 30,
            "control_horizon_steps": 8,
            "thermal_cap_tcap_degC": 83.0,
            "thermal_barrier_tbar_degC": 84.5,
            "noise_seeds": list(range(10)),
            "lhs_samples": 30,
            "lhs_perturbation_pct": 20.0
        },
        "execution_commands": {
            "primary_campaign": "python experiments/run_experimentos.py --orbitas 3 --orbitas-ablacao 3 --sementes 10 --saida results",
            "uncertainty_campaign": "python experiments/run_incerteza.py --amostras 30 --orbitas 3 --saida results",
            "rainflow_validation": "python experiments/validacao_rainflow.py",
            "temporal_convergence": "python experiments/run_convergencia_temporal.py"
        },
        "data_manifest": {
            "results/tabela1.csv": "Desempenho consolidado das 8 políticas de controle em 3 órbitas (semente 0)",
            "results/tabela_sementes.csv": "Estatísticas de dispersão entre 10 sementes estocásticas de ruído de sensor",
            "results/ablacao.csv": "Estudo de ablação funcional arquitetural (3 órbitas, semente 0)",
            "results/amostras.csv": "Matriz de 30 amostras de amostragem por hipercubo latino (LHS, +-20%)",
            "results/convergencia_temporal.csv": "Avaliação do erro numérico com dt in [1.0, 0.5, 0.25, 0.125] s",
            "results/metadata.json": "Metadados de reprodutibilidade e ambiente computacional"
        }
    }
    return metadados

def salvar_configs(pasta_raiz):
    pasta_configs = os.path.join(pasta_raiz, "configs")
    os.makedirs(pasta_configs, exist_ok=True)
    
    config_nominal = {
        "orbit": {
            "altitude_km": 500.0,
            "period_s": 5668.144369,
            "solar_constant_w_m2": 1361.0,
            "albedo": 0.30,
            "earth_ir_base_k": 255.0,
            "earth_ir_amplitude_k": 15.0,
            "solar_absorptivity": 0.12,
            "ir_emissivity": 0.88,
            "radiator_area_m2": 0.12,
            "radiator_capacitance_j_k": 1299.2,
            "view_factor": 0.42
        },
        "cluster": {
            "num_dies": 4,
            "die_capacitance_j_k": 18.5,
            "die_to_radiator_resistance_k_w": 2.10,
            "inter_die_resistances_k_w": {
                "R12_R34": 1.45,
                "R13_R24": 1.52,
                "R14_R23": 3.20
            },
            "voltage_range_v": [0.65, 1.05],
            "frequency_range_ghz": [0.30, 1.30],
            "effective_capacitance_nf": 24.5,
            "leakage_ref_power_w": 0.80,
            "activation_energy_ev": 0.20
        },
        "controller": {
            "dt_plant_s": 0.5,
            "dt_mpc_s": 2.0,
            "prediction_horizon_hp": 30,
            "control_horizon_hc": 8,
            "temperature_cap_c": 83.0,
            "safety_backstop_threshold_c": 84.5,
            "weights": {
                "workload_tracking_w": 1.0,
                "slew_rate_penalty_rw": 0.2,
                "spatial_gradient_penalty_rho": 0.02,
                "thermal_slack_penalty_lambda": 200.0
            },
            "admm_solver": {
                "max_iterations": 60,
                "over_relaxation_alpha": 1.6,
                "regularization_sigma": 1e-6,
                "step_size_rho": 1.0,
                "tolerance": 2e-3
            }
        },
        "observer": {
            "type": "Extended Kalman Filter (Joseph Form)",
            "sensor_noise_sigma_k": 0.25,
            "q_covariance_diagonal": [0.02, 0.02, 0.02, 0.02, 0.05],
            "p0_covariance_diagonal": [1.0, 1.0, 1.0, 1.0, 2.0]
        }
    }
    
    caminho_nominal = os.path.join(pasta_configs, "nominal.json")
    with open(caminho_nominal, "w", encoding="utf-8") as f:
        json.dump(config_nominal, f, indent=2)
    print(f"  [OK] Configuração nominal gravada em: {caminho_nominal}")


if __name__ == "__main__":
    pasta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pasta_results = os.path.join(pasta_raiz, "results")
    os.makedirs(pasta_results, exist_ok=True)
    
    meta = coletar_metadados(pasta_raiz)
    caminho_meta = os.path.join(pasta_results, "metadata.json")
    with open(caminho_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"  [OK] Metadados de reprodutibilidade gravados em: {caminho_meta}")
    salvar_configs(pasta_raiz)
