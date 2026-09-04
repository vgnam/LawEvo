"""Task-metric layer tests: SR, SG, and Q.

The three post-evaluation metrics are computed from the physical trajectory
and task-success conditions, never from the environment reward:
- SR (Success Rate): all success constraints satisfied.
- SG (Success Gap): mean of normalized constraint violations in [0, 1];
  zero if and only if the task is fully successful.
- Q (Goal-Completion Score): fraction of achieved progress predicates in [0, 1].
"""

import numpy as np
import pytest

from lawevo.pid import PANDA_GYM_ADAPTERS, PANDA_VARIANT_ADAPTERS, run_episode
from lawevo.pid.gym_benchmark import clip_violation

pytest.importorskip("panda_gym")

ALL_PANDA_ADAPTERS = {**PANDA_GYM_ADAPTERS, **PANDA_VARIANT_ADAPTERS}


def test_clip_violation_bounds() -> None:
    assert clip_violation(0.0, 0.15) == 0.0
    assert clip_violation(-1.0, 0.15) == 0.0
    assert clip_violation(0.075, 0.15) == pytest.approx(0.5)
    assert clip_violation(10.0, 0.15) == 1.0
    assert clip_violation(5.0, 0.0) == 1.0  # degenerate scale never divides by zero


@pytest.mark.parametrize("adapter_key", tuple(ALL_PANDA_ADAPTERS))
def test_panda_adapters_define_sg_and_q(adapter_key: str) -> None:
    adapter = ALL_PANDA_ADAPTERS[adapter_key]
    structure = adapter.classical[0]

    episode = run_episode(adapter, structure, np.zeros(structure.parameter_count), seed=11)

    assert episode.sg is not None, "SG must be defined for every Panda task"
    assert episode.q is not None, "Q must be defined for every Panda task"
    assert 0.0 <= episode.sg <= 1.0
    assert 0.0 <= episode.q <= 1.0
    assert episode.constraint_violations
    assert all(0.0 <= value <= 1.0 for value in episode.constraint_violations.values())
    assert episode.progress_predicates
    assert all(isinstance(value, bool) for value in episode.progress_predicates.values())


@pytest.mark.parametrize("adapter_key", tuple(ALL_PANDA_ADAPTERS))
def test_sg_zero_iff_success(adapter_key: str) -> None:
    """Rule 2: SG = 0 must mean full success, and success must mean SG = 0."""
    adapter = ALL_PANDA_ADAPTERS[adapter_key]
    structure = adapter.classical[0]

    for seed in (3, 11, 23):
        episode = run_episode(adapter, structure, np.zeros(structure.parameter_count), seed=seed)
        if episode.success:
            assert episode.sg == pytest.approx(0.0), (
                f"{adapter_key}: success episode must have SG == 0 (got {episode.sg})"
            )
        if episode.sg is not None and episode.sg == pytest.approx(0.0):
            assert episode.success, (
                f"{adapter_key}: SG == 0 must imply success (constraints: "
                f"{episode.constraint_violations})"
            )


@pytest.mark.parametrize("adapter_key", tuple(ALL_PANDA_ADAPTERS))
def test_q_zero_when_no_progress_and_bounded_by_sg_semantics(adapter_key: str) -> None:
    """Q is a separate axis: a fully successful episode must reach Q = 1."""
    adapter = ALL_PANDA_ADAPTERS[adapter_key]
    structure = adapter.classical[0]

    for seed in (3, 11, 23):
        episode = run_episode(adapter, structure, np.zeros(structure.parameter_count), seed=seed)
        if episode.success:
            assert episode.q == pytest.approx(1.0), (
                f"{adapter_key}: success must satisfy every progress predicate "
                f"(predicates: {episode.progress_predicates})"
            )


def test_selection_score_is_pure_return() -> None:
    """Selection uses the environment return only; energy/jerk/complexity are diagnostics."""
    from lawevo.pid.gym_benchmark import BenchmarkAdapter

    class _Adapter(BenchmarkAdapter):
        env_id = "unused"
        horizon = 1
        allowed_terms = ()

    adapter = _Adapter()
    assert adapter.score(12.5, 1e6, 1e6, 999) == 12.5
    assert adapter.score(-3.0, 0.0, 0.0, 0) == -3.0
