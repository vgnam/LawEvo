import numpy as np

from lawevo.pid import (
    ControllerStructure,
    PIDBounds,
    PIDGains,
    evaluate_gains,
    generate_scenarios,
    simulate_pid,
    tune_cem,
)


def test_pid_reaches_a_simple_goal() -> None:
    gains = PIDGains(1.2, 0.0, 0.1, 2.5, 0.0, 0.15)
    scenario = generate_scenarios(1, seed=42)[0]
    trajectory = simulate_pid(gains, scenario)
    assert trajectory.success
    assert trajectory.final_distance <= 0.12
    assert np.all(np.isfinite(trajectory.controls))


def test_pid_evaluation_is_deterministic() -> None:
    gains = PIDGains(1.1, 0.05, 0.1, 2.5, 0.03, 0.15)
    scenarios = generate_scenarios(3, seed=7)
    assert evaluate_gains(gains, scenarios) == evaluate_gains(gains, scenarios)
    assert PIDBounds().contains(gains)


def test_structure_cem_tunes_finite_gains() -> None:
    structure = ControllerStructure(
        "test", ("error", "tanh_derivative"), ("error", "tanh_derivative")
    )
    gains, metrics, history = tune_cem(
        structure, generate_scenarios(2, seed=9), iterations=1, population_size=4, seed=3
    )
    assert gains.shape == (4,)
    assert np.isfinite(metrics.score)
    assert len(history) == 2
