from types import SimpleNamespace

import numpy as np
import pytest

from lawevo.pid import (
    ADAPTERS,
    CONTROLLED_VARIANT_ADAPTERS,
    PANDA_GYM_ADAPTERS,
    evaluate_gym_structure,
    inverted_pendulum_lqr,
    run_episode,
)
from lawevo.pid.gym_benchmark import strict_upper_violation

SUITE = {
    **CONTROLLED_VARIANT_ADAPTERS,
    **{key: ADAPTERS[key] for key in ("inverted_pendulum", "reacher")},
    **{key: PANDA_GYM_ADAPTERS[key] for key in ("panda_reach", "panda_push", "panda_slide")},
}


@pytest.mark.parametrize("key", tuple(SUITE))
def test_all_ten_have_finite_sg_in_rollouts_and_aggregate_metrics(key):
    adapter = SUITE[key]
    law = adapter.classical[0]
    metrics, episodes = evaluate_gym_structure(
        adapter, law, np.zeros(law.parameter_count), [3, 11]
    )
    assert adapter.success_gap_spec
    assert metrics.sg is not None and np.isfinite(metrics.sg)
    assert metrics.sg == pytest.approx(np.mean([e.sg for e in episodes]))
    assert metrics.q is not None and np.isfinite(metrics.q)
    assert metrics.q == pytest.approx(np.mean([e.q for e in episodes]))
    for episode in episodes:
        assert 0 <= episode.sg <= 1
        assert episode.constraint_violations
        assert (episode.sg == 0.0) == episode.success
        assert episode.sg == pytest.approx(np.mean(list(episode.constraint_violations.values())))
        assert 0.0 <= episode.q <= 1.0
        assert episode.progress_predicates
        assert episode.q == pytest.approx(np.mean(list(episode.progress_predicates.values())))
        if key in ("inverted_pendulum", "inverted_pendulum_pulse", "reacher", "reacher_payload"):
            assert (episode.q == 1.0) == episode.success


@pytest.mark.parametrize("key", ("inverted_pendulum", "inverted_pendulum_pulse"))
def test_lqr_success_has_zero_sg_and_actual_horizon(key):
    adapter = SUITE[key]
    law, gains = inverted_pendulum_lqr()
    episode = run_episode(adapter, law, gains, seed=90000)
    assert episode.success
    assert episode.sg == 0.0
    assert episode.q == 1.0
    assert episode.diagnostics["steps"] == adapter.horizon
    assert episode.constraint_violations == {"survival": 0.0, "final_angle": 0.0}


@pytest.mark.parametrize("steps,angle,terminated,success", [
    (500, 0.0, False, True),
    (500, np.nextafter(0.1, 0.0), False, True),
    (500, 0.1, False, False),
    (500, -0.1, False, False),
    (500, np.nextafter(0.1, 1.0), False, False),
    (500, 0.0, True, False),  # termination exactly on the last control step
    (499, 0.0, False, False),  # truncation before the required horizon
    (100, 0.3, True, False),
    (500, float("nan"), True, False),
])
def test_cartpole_sg_agrees_with_original_success_at_boundaries(steps, angle, terminated, success):
    adapter = ADAPTERS["inverted_pendulum"]
    tracker = adapter.make_tracker()
    for step in range(steps):
        tracker.update(None, [0.0, angle, 0.0, 0.0], terminated and step == steps - 1)
    violations = adapter.success_constraints(tracker)
    actual = adapter.success(None, [0.0, angle, 0.0, 0.0], steps, terminated)
    assert bool(actual) == success
    assert all(v == 0 for v in violations.values()) == success
    assert all(np.isfinite(v) and 0 <= v <= 1 for v in violations.values())


def test_cartpole_gap_rewards_longer_survival_and_smaller_final_angle():
    adapter = ADAPTERS["inverted_pendulum"]

    def gap(steps, angle):
        tracker = adapter.make_tracker()
        for step in range(steps):
            tracker.update(None, [0, angle, 0, 0], step == steps - 1)
        return np.mean(list(adapter.success_constraints(tracker).values()))

    assert gap(100, 0.2) > gap(400, 0.2)
    assert gap(400, 0.2) > gap(400, 0.15)
    assert all(v > 0 for v in adapter.success_constraints(adapter.make_tracker()).values())


@pytest.mark.parametrize("distance", [0.0, 0.049, 0.05, 0.051, 0.15, 1.0])
def test_reacher_tracker_uses_world_xy_and_preserves_strict_success(distance):
    adapter = ADAPTERS["reacher"]
    positions = {
        "fingertip": SimpleNamespace(xpos=np.array([distance, 0.0, 9.0])),
        "target": SimpleNamespace(xpos=np.array([0.0, 0.0, 0.0])),
    }
    env = SimpleNamespace(unwrapped=SimpleNamespace(data=SimpleNamespace(body=positions.get)))
    tracker = adapter.make_tracker()
    tracker.update(env, None, False)
    violation = adapter.success_constraints(tracker)["goal_position"]
    assert (violation == 0) == adapter.success(env, None, 1, False)
    assert tracker.diagnostics["final_distance"] == distance
    if distance > 0.05:
        assert violation == pytest.approx(min(1, (distance - 0.05) / 0.20))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_distance_is_failure(value):
    assert strict_upper_violation(value, 0.05, 0.2) == 1.0


@pytest.mark.parametrize("key", (
    "panda_reach", "panda_push", "panda_slide", "panda_push_low_friction",
    "panda_slide_friction_shift",
))
def test_existing_panda_inclusive_distance_threshold_is_preserved(key):
    adapter = SUITE[key]
    tracker = adapter.make_tracker()
    tolerance = tracker.distance_threshold
    for distance, expected in ((tolerance, True), (tolerance + 0.01, False)):
        tracker.steps = [{"ee": np.array([distance, 0, 0]),
                          "obj": np.array([distance, 0, 0]), "goal": np.zeros(3)}]
        assert all(v == 0 for v in adapter.success_constraints(tracker).values()) == expected


def test_moving_slow_requires_both_tracking_rate_and_final_distance():
    adapter = SUITE["panda_reach_moving_slow"]
    for rate, distance, expected in ((0.6, 0.05, True), (0.59, 0.05, False),
                                      (0.6, 0.051, False), (1.0, 0.0, True)):
        tracker = SimpleNamespace(tracking_rate=rate, final_distance=distance)
        assert all(v == 0 for v in adapter.success_constraints(tracker).values()) == expected
