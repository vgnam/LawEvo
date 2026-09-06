from dataclasses import replace

import numpy as np
import pytest

from lawevo.pid import ADAPTERS, CONTROLLED_VARIANT_ADAPTERS
from lawevo.pid.gym_benchmark import GymMetrics


@pytest.mark.parametrize("key", ("reacher", "reacher_payload"))
@pytest.mark.parametrize("distances,expected", [
    ([], (False, False, False)),
    ([0.3, 0.2], (False, False, False)),
    ([0.2, 0.08, 0.06], (True, False, False)),
    ([0.2, 0.04, 0.07], (True, True, False)),
    ([0.2, 0.04], (True, True, True)),
    ([0.1], (False, False, False)),
    ([0.05], (True, False, False)),
    ([np.nextafter(0.05, 0.0)], (True, True, True)),
    ([float("nan"), float("inf")], (False, False, False)),
])
def test_reacher_progress_distinguishes_approach_transient_contact_and_final_success(
    key, distances, expected,
):
    adapter = {**ADAPTERS, **CONTROLLED_VARIANT_ADAPTERS}[key]
    tracker = adapter.make_tracker()
    tracker.steps = [{"distance": distance} for distance in distances]
    assert tuple(adapter.progress_predicates(tracker).values()) == expected
    success = all(v == 0 for v in adapter.success_constraints(tracker).values())
    assert all(expected) == success
    tracker.reset()
    assert not any(adapter.progress_predicates(tracker).values())


@pytest.mark.parametrize("key", ("inverted_pendulum", "inverted_pendulum_pulse"))
@pytest.mark.parametrize("count,angle,terminated,expected", [
    (0, 0.0, False, (False, False, False)),
    (100, 0.2, True, (False, False, False)),
    (250, 0.2, True, (False, False, False)),
    (250, 0.2, False, (True, False, False)),
    (499, 0.2, True, (True, False, False)),
    (500, 0.0, True, (True, False, True)),
    (500, 0.1, False, (True, True, False)),
    (500, float("nan"), False, (True, True, False)),
    (500, 0.09, False, (True, True, True)),
])
def test_cartpole_progress_preserves_survival_milestones_and_strict_final_angle(
    key, count, angle, terminated, expected,
):
    adapter = {**ADAPTERS, **CONTROLLED_VARIANT_ADAPTERS}[key]
    tracker = adapter.make_tracker()
    for step in range(count):
        tracker.update(None, [0, angle, 0, 0], terminated and step == count - 1)
    assert tuple(adapter.progress_predicates(tracker).values()) == expected
    success = all(v == 0 for v in adapter.success_constraints(tracker).values())
    assert all(expected) == success


def test_q_does_not_change_selection_order():
    metrics = GymMetrics(0.4, -8.0, 0.5, 2.0, 100.0, 5, 0.2, 0.0)
    assert metrics.selection_key == replace(metrics, q=1.0).selection_key
    assert metrics.selection_key == replace(metrics, q=None).selection_key


@pytest.mark.parametrize("base,variant", [
    ("reacher", "reacher_payload"), ("inverted_pendulum", "inverted_pendulum_pulse"),
])
def test_pair_shares_milestone_spec_and_prompt_contains_it(base, variant):
    from experiments.gymnasium_classical_benchmarks import prompt

    adapter = CONTROLLED_VARIANT_ADAPTERS[variant]
    assert adapter.progress_spec == ADAPTERS[base].progress_spec
    text = prompt(variant, adapter.allowed_terms, [], [], 2, 1)
    for milestone in adapter.progress_predicates(adapter.make_tracker()):
        assert milestone in text
    assert "diagnostic only, never used in selection" in text
