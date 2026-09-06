from copy import copy
from dataclasses import asdict

import numpy as np
import pytest

from lawevo.pid import ADAPTERS, CONTROLLED_VARIANT_ADAPTERS, PANDA_GYM_ADAPTERS
from lawevo.pid.gym_benchmark import run_episode
from lawevo.verify.controller import (
    SUPPORTED_ENVIRONMENTS,
    ControllerBarrierConfig,
    ControllerBarrierVerifier,
    barrier_spec,
)

ALL = {a.env_id: a for a in {**ADAPTERS, **CONTROLLED_VARIANT_ADAPTERS, **PANDA_GYM_ADAPTERS}.values()}


@pytest.mark.parametrize("env_id", sorted(SUPPORTED_ENVIRONMENTS))
def test_all_ten_audit_real_rollouts_without_changing_metrics(env_id):
    adapter = copy(ALL[env_id])
    adapter.horizon = 5
    law = adapter.classical[0]
    gains = np.ones(law.parameter_count)
    plain = run_episode(adapter, law, gains, 11)
    audited = run_episode(adapter, law, gains, 11, verify_barriers=True)
    report = audited.barrier_verification
    assert report is not None
    assert report["steps"] > 0
    assert report["certified_between_samples"] is False
    assert report["numeric_failures"] == 0
    assert report["spec"] == barrier_spec(env_id)
    assert report["minimum_margin"] is not None
    a, b = asdict(plain), asdict(audited)
    a.pop("barrier_verification")
    b.pop("barrier_verification")
    assert a == b


class ScalarAudit(ControllerBarrierVerifier):
    def margins(self, env, observation):
        return {"example": float(observation)}


def test_barrier_residual_failure_does_not_mean_envelope_exit():
    audit = ScalarAudit("Reacher-v5", None, 1.0, 0.1, 123)
    audit.update(None, 0.1)
    report = audit.report()
    assert report["envelope_satisfied"]
    assert report["negative_residual_count"] == 1
    assert report["status"] == "failed_sampled_checks"
    assert report["first_failure"]["seed"] == 123
    assert report["first_failure"]["step"] == 1


def test_initial_unsafe_state_and_nonfinite_action_are_not_hidden():
    audit = ScalarAudit("Reacher-v5", None, -0.1, 0.1, 5)
    audit.check_action(np.array([np.nan]), np.array([-1]), np.array([1]))
    audit.update(None, 0.1)
    report = audit.report()
    assert report["outside_envelope_count"] == 1
    assert report["numeric_failures"] == 1
    assert report["first_failure"]["step"] == 0
    assert not report["envelope_satisfied"]


def test_constant_safe_barrier_passes_and_clipping_is_reported_separately():
    audit = ScalarAudit("Reacher-v5", None, 0.5, 0.1, 5)
    audit.check_action(np.array([2]), np.array([-1]), np.array([1]))
    audit.update(None, 0.5)
    assert audit.report()["status"] == "passed_sampled_checks"
    assert audit.report()["saturated_steps"] == 1


@pytest.mark.parametrize("kwargs", [{"decay_rate": 0}, {"tolerance": -1}, {"reacher_speed_limit": np.nan}])
def test_invalid_audit_config(kwargs):
    with pytest.raises(ValueError):
        ControllerBarrierConfig(**kwargs)


def test_raw_expression_exposes_numeric_failure():
    from lawevo.pid.expression import SymbolicExpression

    law = SymbolicExpression("overflow", "K1*x")
    with np.errstate(all="ignore"):
        raw = law.evaluate_raw({"x": np.array([1e308])}, [1e308])
        safe = law.evaluate({"x": np.array([1e308])}, [1e308])
    assert not np.isfinite(raw).all()
    assert np.isfinite(safe).all()


def test_unsupported_profile_is_explicit():
    assert barrier_spec("RobosuiteWipe-v0") is None


def test_scalar_broadcast_is_valid():
    audit = ScalarAudit("Reacher-v5", None, 0.5, 0.1, 5)
    audit.check_action(0.0, np.full(2, -1), np.ones(2))
    audit.update(None, 0.5)
    assert audit.report()["numeric_failures"] == 0


def test_saved_controller_cli_without_training(tmp_path, monkeypatch):
    import json
    import sys

    from experiments.verify_controller import main

    adapter = ALL["Reacher-v5"]
    monkeypatch.setattr(adapter, "horizon", 4)
    law = adapter.classical[0]
    controller = tmp_path / "controller.json"
    output = tmp_path / "audit.json"
    controller.write_text(json.dumps({"structure": law.to_dict(), "gains": [0]*law.parameter_count}))
    monkeypatch.setattr(sys, "argv", ["verify", "--environment", "Reacher-v5",
                        "--controller", str(controller), "--episodes", "1", "--output", str(output)])
    main()
    payload = json.loads(output.read_text())
    assert payload["episodes"][0]["steps"] == 4
    assert not payload["certified_between_samples"]
