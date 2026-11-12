import numpy as np

from audit.numerical_validation.validate_thermal_budget import (
    frequencia_uniforme_para_potencia,
    simular_horizonte_planta_completa,
)
import src.tcas_core as C


def test_full_plant_heun_recalculates_power_and_returns_finite_state():
    x0 = np.array([330.0, 331.0, 332.0, 333.0, 300.0])
    activity = np.array([0.8, 0.8, 0.8, 0.8])
    frequency = np.full(4, C.F_MIN)
    result = simular_horizonte_planta_completa(
        x0, frequency, activity, np.array([5.0, 5.0]), dt=0.1
    )
    assert np.isfinite(result["max_T_die_K"])
    assert np.isfinite(result["P_media_W"])
    assert result["duracao_s"] == 0.2
    assert result["P_media_W"] > 0.0


def test_frequency_inverse_respects_initial_power_budget():
    x0 = np.array([330.0, 331.0, 332.0, 333.0, 300.0])
    activity = np.array([0.8, 0.8, 0.8, 0.8])
    pd_min, pl_min = C.potencias(x0[:4], np.full(4, C.F_MIN), activity)
    pd_max, pl_max = C.potencias(x0[:4], np.full(4, C.F_MAX), activity)
    p_min = float(np.sum(pd_min + pl_min))
    p_max = float(np.sum(pd_max + pl_max))
    target = 0.5 * (p_min + p_max)
    frequency = frequencia_uniforme_para_potencia(x0, activity, target)
    pd, pl = C.potencias(x0[:4], np.full(4, frequency), activity)
    achieved = float(np.sum(pd + pl))
    assert p_min <= achieved <= target + 1e-8
    assert abs(achieved - target) < 1e-5
