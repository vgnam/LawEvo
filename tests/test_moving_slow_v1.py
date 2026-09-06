import numpy as np
import pytest

from lawevo.pid.controlled_variants import (
    PandaReachMovingSlowAdapter,
    PandaReachMovingSlowV1Adapter,
)


def trajectory(adapter, distances):
    tracker = adapter.make_tracker()
    tracker.steps = [{"ee": np.array([d, 0, 0]), "goal": np.zeros(3)} for d in distances]
    return tracker


@pytest.mark.parametrize("distance,success", [(0.009, True), (0.01, True), (0.010001, False)])
def test_radius_applies_to_sr_sg_q(distance, success):
    adapter = PandaReachMovingSlowV1Adapter()
    tracker = trajectory(adapter, [distance]*150)
    gaps = adapter.success_constraints(tracker)
    assert all(v == 0 for v in gaps.values()) == success
    assert adapter.progress_predicates(tracker)["finish_inside_radius"] == success
    assert tracker.tracking_rate == float(success)
    assert adapter.success_gap_spec["radius_m"] == 0.01
    assert adapter.progress_spec["radius_m"] == 0.01


def test_holding_orbit_center_no_longer_passes():
    t = np.arange(1, 151)*0.04
    offsets = 0.025*np.column_stack((np.sin(2*np.pi*t/5), 0.5*np.sin(4*np.pi*t/5), np.zeros(150)))
    for adapter, expected in ((PandaReachMovingSlowAdapter(), True), (PandaReachMovingSlowV1Adapter(), False)):
        tracker = adapter.make_tracker()
        tracker.steps = [{"ee": np.zeros(3), "goal": p} for p in offsets]
        assert all(v == 0 for v in adapter.success_constraints(tracker).values()) == expected


def test_v1_only_changes_success_radius_in_simulation():
    old, new = PandaReachMovingSlowAdapter(), PandaReachMovingSlowV1Adapter()
    a, b = old.make_env(), new.make_env()
    try:
        first, _ = a.reset(seed=31)
        second, _ = b.reset(seed=31)
        assert a.unwrapped.task.distance_threshold == 0.05
        assert b.unwrapped.task.distance_threshold == 0.01
        for name in first:
            np.testing.assert_allclose(first[name], second[name])
        for _ in range(20):
            first, r1, term1, trunc1, _ = a.step(np.array([0.1, 0, 0]))
            second, r2, term2, trunc2, _ = b.step(np.array([0.1, 0, 0]))
            for name in first:
                np.testing.assert_allclose(first[name], second[name], atol=1e-12)
            assert r1 == pytest.approx(r2)
            assert not (term1 or term2 or trunc1 or trunc2)
    finally:
        a.close()
        b.close()
