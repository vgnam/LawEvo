import numpy as np

from lawevo import BarrierVerifier, CBFSafetyFilter, CircleObstacle, UnicycleRobot, parse_barrier
from lawevo.sim import RolloutConfig, proportional_unicycle_policy, rollout
from lawevo.verify import VerificationConfig

robot = UnicycleRobot(
    obstacles=[CircleObstacle((0.0, 0.0), 0.6)],
    workspace=((-3.0, 3.0), (-3.0, 3.0)),
)
barrier = parse_barrier("min(dist_to_obstacle(0, 0.3))")
verifier = BarrierVerifier(robot, VerificationConfig(safety_margin=0.35, max_grid_points=25_000))
verification = verifier.verify(barrier)
if not verification.accepted:
    raise SystemExit(f"Barrier rejected: {verification.reason}")

trajectory = rollout(
    robot,
    proportional_unicycle_policy,
    # Every k >= the verified minimum remains feasible on h >= 0. A nonzero
    # operational value is less conservative away from the boundary.
    CBFSafetyFilter(robot, barrier, max(verification.alpha or 0.0, 2.0)),
    initial_state=np.array([-2.0, -1.0, 0.0]),
    goal=np.array([2.0, -1.0]),
    config=RolloutConfig(steps=250),
)
print(
    {
        "alpha": verification.alpha,
        "operational_alpha": max(verification.alpha or 0.0, 2.0),
        "sampled": verification.sampled_points,
        "certified_between_samples": verification.certified_between_samples,
        "reached_goal": trajectory.reached_goal,
        "safety_violation": trajectory.safety_violation,
        "fitness": trajectory.fitness(),
    }
)
