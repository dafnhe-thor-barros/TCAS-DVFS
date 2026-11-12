# -*- coding: utf-8 -*-
"""
generate_provenance.py - Gera e sincroniza metadados de reprodutibilidade e rastreabilidade forense (Fase 7: A18).

Atualiza:
- results/metadata.json
- audit/reproducibility/campaign_provenance.json
com o SHA exato do commit de código utilizado na geração dos artefatos (`code_commit_sha`),
versões de ambiente, parâmetros de simulação, comandos exatos e hashes SHA-256 de todos os
arquivos de código, auditoria e resultados.
"""
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import numpy as np
import scipy
import matplotlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_rev_parse(ref="HEAD"):
    out = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT)
    return out.decode("utf-8").strip()


def main():
    code_sha = git_rev_parse("HEAD")
    baseline_sha = git_rev_parse("audit-baseline-a8027ee")
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    rel_paths = [
        "src/tcas_core.py",
        "src/tcas_dvfs.py",
        "src/pinn_thermal.py",
        "experiments/run_experimentos.py",
        "experiments/run_incerteza.py",
        "experiments/run_convergencia_temporal.py",
        "experiments/validar_pinn.py",
        "experiments/validacao_jacobiano.py",
        "experiments/validacao_rainflow.py",
        "experiments/gerar_figuras.py",
        "results/tabela1.csv",
        "results/tabela_sementes.csv",
        "results/ablacao.csv",
        "results/resultados.json",
        "results/amostras.csv",
        "results/resumo_incerteza.csv",
        "results/spearman.csv",
        "results/incerteza.json",
        "results/convergencia_temporal.csv",
        "results/convergencia_temporal.json",
        "audit/pinn/pinn_gradient_tests.json",
        "audit/pinn/pinn_vs_mlp_comparison.json",
        "audit/pinn/pinn_training_curves.csv",
        "audit/mpc/mpc_validation_report.json",
        "audit/mpc/mpc_nonlinear_prediction_error.csv",
        "audit/numerical_validation/thermal_budget_validation.json",
        "audit/sdp/sdp_validation_report.json",
        "audit/numerical_validation/ekf_jacobian_dso_report.json",
        "audit/numerical_validation/reliability_validation.json",
    ]

    checksums = {}
    for rp in rel_paths:
        ap = os.path.join(ROOT, rp)
        if os.path.exists(ap):
            checksums[rp] = {
                "sha256": sha256_file(ap),
                "size_bytes": os.path.getsize(ap),
            }

    metadata = {
        "project": "TCAS-DVFS: Thermal-Crosstalk-Aware Scheduling and Model Predictive DVFS for Spaceborne AI",
        "timestamp_utc": now_utc,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "matplotlib_version": matplotlib.__version__,
            "operating_system": f"{platform.system()} {platform.release()} ({platform.version()})",
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
        },
        "version_control": {
            "vcs": "git",
            "repository": "https://github.com/dafnhe-thor-barros/TCAS-DVFS",
            "baseline_frozen_tag": "audit-baseline-a8027ee",
            "baseline_frozen_commit": baseline_sha,
            "code_commit_hash": code_sha,
            "git_commit_hash": code_sha,
            "provenance_note": (
                "code_commit_hash e git_commit_hash registram exatamente o commit do código-fonte e dos testes "
                "de auditoria (Fases 0 a 6) com o qual todos os artefatos experimentais desta campanha foram gerados."
            ),
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
            "decision_variables_mpc": 152,
            "spatial_temporal_slacks": 120,
            "thermal_cap_tcap_degC": 83.0,
            "thermal_barrier_tbar_degC": 84.5,
            "noise_seeds": list(range(10)),
            "lhs_samples": 30,
            "lhs_perturbation_pct": 20.0,
        },
        "execution_commands": {
            "pinn_gradient_unit_tests": "python audit/tests/test_pinn_gradients.py",
            "pinn_validation": "python experiments/validar_pinn.py",
            "mpc_admm_validation": "python audit/mpc/validate_mpc_admm.py",
            "thermal_budget_validation": "python audit/numerical_validation/validate_thermal_budget.py",
            "sdp_scheduler_validation": "python audit/sdp/validate_sdp_scheduler.py",
            "ekf_jacobian_dso_validation": "python audit/numerical_validation/validate_ekf_and_dso.py",
            "reliability_rainflow_validation": "python audit/numerical_validation/validate_reliability_metrics.py",
            "jacobian_validation": "python experiments/validacao_jacobiano.py",
            "rainflow_validation": "python experiments/validacao_rainflow.py",
            "temporal_convergence": "python experiments/run_convergencia_temporal.py",
            "primary_campaign": "python experiments/run_experimentos.py --orbitas 3 --orbitas-ablacao 1 --sementes 10 --saida results --workers 8",
            "uncertainty_campaign": "python experiments/run_incerteza.py --amostras 30 --orbitas 3 --saida results --workers 8",
            "figure_generation": "python experiments/gerar_figuras.py",
        },
        "data_manifest": {
            "results/tabela1.csv": "Desempenho consolidado das 8 políticas de controle em 3 órbitas (semente 0)",
            "results/tabela_sementes.csv": "Estatísticas de dispersão entre 10 sementes estocásticas de ruído de sensor",
            "results/ablacao.csv": "Estudo de ablação funcional arquitetural (1 órbita, semente 0)",
            "results/resultados.json": "Resultados consolidados da campanha principal, ablação, confiabilidade e multi-semente",
            "results/amostras.csv": "Matriz de 30 amostras de amostragem por hipercubo latino (LHS, +-20%)",
            "results/resumo_incerteza.csv": "Intervalos de confiança t-Student e bootstrap (B=4000) da campanha de incerteza",
            "results/spearman.csv": "Correlações de postos de Spearman entre parâmetros térmicos e métricas",
            "results/incerteza.json": "Resultados completos da campanha de incerteza paramétrica e Clopper-Pearson",
            "results/convergencia_temporal.csv": "Avaliação do erro numérico com dt in [1.0, 0.5, 0.25, 0.125] s",
            "results/convergencia_temporal.json": "Métricas detalhadas de convergência temporal com dt_MPC=2.0s e dt_SDP=120.0s invariantes",
            "results/fig_validacao_pinn.png": "Curvas de validação out-of-sample da PINN térmica vs MLP supervisionada e ground truth Heun",
            "results/metadata.json": "Metadados de reprodutibilidade, commit de código e manifesto SHA-256",
        },
        "artifact_checksums_sha256": checksums,
    }

    meta_path = os.path.join(ROOT, "results", "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    prov_path = os.path.join(ROOT, "audit", "reproducibility", "campaign_provenance.json")
    os.makedirs(os.path.dirname(prov_path), exist_ok=True)
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] Metadados de proveniência atualizados com code_commit_hash = {code_sha}")


if __name__ == "__main__":
    main()
