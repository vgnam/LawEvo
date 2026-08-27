import numpy as np

from lawevo.dsl import parse_barrier
from lawevo.evolve import Candidate, EvaluationScenario, EvolutionConfig, EvolutionRunner
from lawevo.filter import CBFSafetyFilter
from lawevo.robot import CircleObstacle, UnicycleRobot
from lawevo.sim import RolloutConfig, proportional_unicycle_policy, rollout
from lawevo.verify import BarrierVerifier, VerificationConfig


def setup_system():
    robot = UnicycleRobot(
        [CircleObstacle((0.0, 0.0), 0.5)],
        workspace=((-2, 2), (-2, 2)),
    )
    barrier = parse_barrier("min(dist_to_obstacle(0, 0.25))")
    verifier = BarrierVerifier(
        robot,
        VerificationConfig(safety_margin=0.5, max_grid_points=5000, bisection_iterations=16),
    )
    return robot, barrier, verifier


def test_verification_and_rollout_smoke() -> None:
    robot, barrier, verifier = setup_system()
    result = verifier.verify(barrier)
    assert result.accepted
    assert result.alpha is not None
    assert not result.certified_between_samples
    trajectory = rollout(
        robot,
        proportional_unicycle_policy,
        CBFSafetyFilter(robot, barrier, result.alpha),
        np.array([-1.5, -1.0, 0.0]),
        np.array([1.5, 1.0]),
        RolloutConfig(dt=0.05, steps=20),
    )
    assert len(trajectory.states) >= 2
    assert np.all(trajectory.cbf_residuals >= -1e-8)


def test_one_generation_evolution() -> None:
    robot, barrier, verifier = setup_system()
    candidate = Candidate("baseline", proportional_unicycle_policy, barrier)

    def no_children(survivors, belief, count, generation):
        assert count == 0
        return []

    runner = EvolutionRunner(
        robot,
        verifier,
        [EvaluationScenario(np.array([-1.5, -1.0, 0.0]), np.array([1.5, 1.0]))],
        no_children,
        EvolutionConfig(generations=1, population_size=1, survivor_count=1),
        RolloutConfig(steps=5),
    )
    best, reports = runner.run([candidate])
    assert best.name == "baseline"
    assert reports[0].pass_rate == 1.0
    assert runner.belief.primitive
