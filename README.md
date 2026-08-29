# LawEvo

LawEvo is an executable research prototype for knowledge-augmented evolution of
interpretable robot controllers and control-barrier functions (CBFs). It combines an LLM
outer loop that proposes compact symbolic controller structures with a numerical inner
loop that optimizes every gain under an equal evaluation budget.

## Research idea

The central idea is to separate **structure discovery** from **parameter optimization**:

1. A task-specific prompt tells the LLM the environment dynamics, observation/action
   semantics, success conditions, failure modes, available feedback signals, and the
   desired return/energy/jerk trade-off.
2. The LLM selects signal structures only; it never chooses numeric gains.
3. Cross-Entropy Method (CEM) optimizes every gain `K` for both evolved structures and
   classical baselines using the same simulation budget.
4. EoH-inspired `E1`/`E2` exploration and crossover plus `M1`/`M2`/`M3` mutation prompts
   generate structurally diverse, failure-directed, and generalization-oriented offspring.
5. An archive prevents duplicate structures, while retry, deterministic local mutation,
   and per-generation checkpoints make long runs resumable when the model endpoint fails.
6. Controllers are compared on held-out initial states and physical-parameter variations,
   reporting task return, success, control energy, command jerk, and complexity.

This makes it possible to test a precise question: can task-aware symbolic evolution find
controller structures that outperform tuned P/PI/PD/PID, LQR, posture-feedback, or CPG
baselines without hiding the cost in actuator effort or nonsmooth commands?

The variation prompts adapt the five strategies from [Evolution of Heuristics
(EoH)](https://arxiv.org/abs/2401.02051): `E1` explores forms unlike multiple parents,
`E2` recombines their common backbone, `M1` changes structural terms, `M2` targets an
observed metric failure, and `M3` simplifies for generalization. LawEvo deliberately does
not use EoH-style parameter mutation for gains: all numeric `K` values remain under the
equal-budget CEM optimizer.

## Included

- A strict barrier DSL with EBNF expressions and JSON trees, `min` / `wsum`
  composition, robot-specific primitive validation, analytic gradients, and compositional
  Lipschitz bounds.
- A `RobotInterface` and physically consistent `UnicycleRobot` implementation.
- Sampled CBF verification over a bounded state domain and bisection for the minimum
  feasible linear class-K coefficient `alpha(h) = k h`.
- A dependency-free, exact low-dimensional CBF-QP filter for a control box and one
  half-space constraint.
- Unicycle rollout, task/energy/jerk metrics, a knowledge-augmented evolutionary loop,
  bounded belief space, no-belief ablation switch, and separate policy/barrier prompts.
- Unit and end-to-end smoke tests.

## Quick start

```powershell
py -m pip install -e ".[dev,benchmarks]"
py -m pytest
py -m examples.unicycle_mvp
```

The package itself depends only on NumPy.

## NVIDIA NIM configuration

Create a local `.env` file from the tracked template:

```powershell
Copy-Item .env.example .env
```

Then edit only the key if the default model and endpoint are suitable:

```dotenv
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_MODEL=openai/gpt-oss-120b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1/chat/completions
```

`.env` is ignored by Git. Environment variables and CLI flags can still override these
settings. Never commit a real API key.

## Running the experiments

### Benchmark protocol

The standard benchmark evolves structures for **20 generations**, tunes every evolved and
classical gain with the same CEM budget using **6 training episodes**, and reports each
controller over exactly **30 held-out test episodes**. The CLI defaults use this 20/6/30
protocol.

### Commands for all 22 environments

The 22 copy-paste commands below each run exactly one environment. They explicitly use
the standard 20-generation, 6-training-episode, 30-test-episode protocol:

```powershell
# 1. Pendulum-v1
py -m experiments.gymnasium_classical_benchmarks --environment Pendulum-v1 --generations 20 --train-episodes 6 --test-episodes 30

# 2. InvertedPendulum-v5
py -m experiments.gymnasium_classical_benchmarks --environment InvertedPendulum-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 3. InvertedDoublePendulum-v5
py -m experiments.gymnasium_classical_benchmarks --environment InvertedDoublePendulum-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 4. Reacher-v5
py -m experiments.gymnasium_classical_benchmarks --environment Reacher-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 5. Pusher-v5
py -m experiments.gymnasium_classical_benchmarks --environment Pusher-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 6. Hopper-v5
py -m experiments.gymnasium_classical_benchmarks --environment Hopper-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 7. Walker2d-v5
py -m experiments.gymnasium_classical_benchmarks --environment Walker2d-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 8. HalfCheetah-v5
py -m experiments.gymnasium_classical_benchmarks --environment HalfCheetah-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 9. Swimmer-v5
py -m experiments.gymnasium_classical_benchmarks --environment Swimmer-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 10. Ant-v5
py -m experiments.gymnasium_classical_benchmarks --environment Ant-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 11. Humanoid-v5
py -m experiments.gymnasium_classical_benchmarks --environment Humanoid-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 12. HumanoidStandup-v5
py -m experiments.gymnasium_classical_benchmarks --environment HumanoidStandup-v5 --generations 20 --train-episodes 6 --test-episodes 30

# 13. BipedalWalker-v3
py -m experiments.gymnasium_classical_benchmarks --environment BipedalWalker-v3 --generations 20 --train-episodes 6 --test-episodes 30

# 14. Panda-Gym Reach (dense reward)
py -m experiments.gymnasium_classical_benchmarks --environment PandaReachDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# 15. Panda-Gym Push (dense reward)
py -m experiments.gymnasium_classical_benchmarks --environment PandaPushDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# 16. Panda-Gym Slide (dense reward)
py -m experiments.gymnasium_classical_benchmarks --environment PandaSlideDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# 17. Panda-Gym Pick and Place (dense reward)
py -m experiments.gymnasium_classical_benchmarks --environment PandaPickAndPlaceDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# 18. Panda-Gym Stack (dense reward)
py -m experiments.gymnasium_classical_benchmarks --environment PandaStackDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# 19. Robosuite Lift with Panda OSC
py -m experiments.gymnasium_classical_benchmarks --environment RobosuiteLift-v0 --generations 20 --train-episodes 6 --test-episodes 30

# 20. Robosuite Stack with Panda OSC
py -m experiments.gymnasium_classical_benchmarks --environment RobosuiteStack-v0 --generations 20 --train-episodes 6 --test-episodes 30

# 21. Robosuite square-nut assembly with Panda OSC
py -m experiments.gymnasium_classical_benchmarks --environment RobosuiteNutAssemblySquare-v0 --generations 20 --train-episodes 6 --test-episodes 30

# 22. Robosuite Door with Panda OSC
py -m experiments.gymnasium_classical_benchmarks --environment RobosuiteDoor-v0 --generations 20 --train-episodes 6 --test-episodes 30
```

The reported return, success rate, energy, and jerk are means over the 30 held-out test
episodes. No output path is required. Every invocation creates a timestamped run with one
folder per environment:

```text
results/<YYYYMMDD_HHMMSS>/<environment>/
  classical/
    controllers.json
  lawevo/
    best_controller.json
    generation_plans.json
    nim_responses.json
    generations/
      generation_000.json
      generation_001.json
      ...
  plot/
    <environment>_comparison.png
    <environment>_comparison.pdf
  summary/
    metrics_summary.csv
    rollout_metrics.csv
    results.json
```

Every generation JSON stores all individuals evaluated in that generation, their optimized
gains and metrics, the full ranking so far, and the best-so-far individual. The timestamp
root also contains `run_manifest.json` and a `state/` directory for checkpoints.

The same metrics are printed to the console when the run finishes. To resume a timestamped
run, provide its directory name:

```powershell
py -m experiments.gymnasium_classical_benchmarks `
  --environment Ant-v5 `
  --resume-run 20260827_231500 `
  --generations 20 `
  --proposals 6 `
  --cem-iterations 10 `
  --cem-population 32 `
  --train-episodes 6 `
  --test-episodes 30
```

### Supported benchmark environments

| Environment | Tuned classical baselines |
|---|---|
| `Pendulum-v1` | P, PI, PD, PID |
| `InvertedPendulum-v5` | P, PI, PD, PID, LQR |
| `InvertedDoublePendulum-v5` | P, PI, PD, PID |
| `Reacher-v5` | task-space P, PI, PD, PID |
| `Pusher-v5` | task-space P, PI, PD, PID |
| `Hopper-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `Walker2d-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `HalfCheetah-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `Swimmer-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `Ant-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `Humanoid-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `HumanoidStandup-v5` | Stand Posture P/PD, Height+Posture PD, Full Balance PD |
| `BipedalWalker-v3` | Posture P, Posture PD, CPG, CPG+PD |
| `PandaReachDense-v3` | Task P, PI, PD, PID |
| `PandaPushDense-v3` | Reach P/PD, Object Goal P, Contact+Goal PD |
| `PandaSlideDense-v3` | Reach P/PD, Object Goal P, Contact+Goal PD |
| `PandaPickAndPlaceDense-v3` | Reach P/PD, Pick+Place, Pick+Place PD |
| `PandaStackDense-v3` | Reach P/PD, Pick+Stack, Pick+Stack PD |
| `RobosuiteLift-v0` | Reach P/PD, Pick+Lift, Pick+Lift PD |
| `RobosuiteStack-v0` | Reach P/PD, Pick+Stack, Pick+Stack PD |
| `RobosuiteNutAssemblySquare-v0` | Reach P/PD, Pick+Insert, Pick+Insert PD |
| `RobosuiteDoor-v0` | Reach P/PD, Door P/PD |

These are the environments currently implemented by LawEvo adapters. Other Gymnasium
environments can be added, but require an adapter defining observation-to-signal mapping,
action semantics, classical structures, success criteria, and task-specific prompt goals.

The Panda-Gym tasks use PyBullet and dense-reward `v3` environments with normalized
end-effector displacement control. Install all benchmark dependencies, including
`panda-gym==3.0.7`, with `py -m pip install -e ".[benchmarks]"`.

### Panda-Gym commands

Install the benchmark dependencies once:

```powershell
py -m pip install -e ".[benchmarks]"
```

Run each supported Panda-Gym task with the standard 20/6/30 protocol:

```powershell
# Reach
py -m experiments.gymnasium_classical_benchmarks --environment PandaReachDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# Push
py -m experiments.gymnasium_classical_benchmarks --environment PandaPushDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# Slide
py -m experiments.gymnasium_classical_benchmarks --environment PandaSlideDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# Pick and Place
py -m experiments.gymnasium_classical_benchmarks --environment PandaPickAndPlaceDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

# Stack
py -m experiments.gymnasium_classical_benchmarks --environment PandaStackDense-v3 --generations 20 --train-episodes 6 --test-episodes 30
```

The robosuite tasks use a Panda robot with the `BASIC` operational-space controller. Their
seven-dimensional action is normalized `[delta_xyz, delta_axis_angle, gripper]`; therefore,
reported energy and jerk are OSC command-space metrics, not raw joint-torque metrics.
Robosuite 1.5.2 is pinned with MuJoCo 3.3.x because later MuJoCo releases remove an API that
this robosuite release still uses.

Runs checkpoint every evaluated structure and save complete generation plans. Use the
printed run ID with `--resume-run` to resume that exact timestamped run.

The original unicycle PID-structure experiment is available with:

```powershell
py -m experiments.evolve_pid_structure_nim `
  --output results/pid_structure_nim
```

## Barrier syntax

Expression form:

```text
min(dist_to_obstacle(0, 0.3), boundary_margin(x, 5, 0.2))
wsum(0.7*dist_to_obstacle(0, 0.3), 0.3*boundary_margin(x, 5, 0.2))
```

Equivalent JSON-tree form:

```json
{
  "op": "min",
  "terms": [
    {"primitive": "dist_to_obstacle", "args": [0, 0.3]}
  ]
}
```

The barrier must be validated against a robot before use. The kinematic unicycle exposes
`dist_to_obstacle` and `boundary_margin`. It intentionally rejects `speed_margin`: speed
is a control input in the state `(px, py, theta)`, not a state variable. A dynamic robot
model can expose that primitive correctly.

## Core API

```python
robot = UnicycleRobot(
    [CircleObstacle((0.0, 0.0), 0.5)],
    workspace=((-3, 3), (-3, 3)),
)
barrier = parse_barrier("min(dist_to_obstacle(0, 0.3))")
result = BarrierVerifier(robot).verify(barrier)
if result.accepted:
    safety_filter = CBFSafetyFilter(robot, barrier, result.alpha)
```

`EvolutionRunner` takes an injected `offspring_generator`. This boundary is deliberate:
it lets an experiment enforce an exact API-call budget and provider, parser-check barrier
JSON before verification, and compile policy code in a separate sandbox. LawEvo never
executes arbitrary LLM output in the host process.

## Verification semantics

The verifier checks

```text
max_{u_min <= u <= u_max} L_f h(x) + L_g h(x) u + k h(x) >= 0
```

at sampled states in `h(x) >= 0`. `VerificationResult.alpha` is the minimum sampled-feasible
coefficient found by bisection. Because increasing `k` relaxes the constraint inside the
safe set, experiments may use an operational coefficient `k_operational >= alpha`; this
choice must be fixed or reported as an ablation. In many driftless systems the minimum is
zero (stopping is feasible), and using zero can be unnecessarily conservative for task
performance.

By default, a pass is explicitly labeled sampled rather than continuously certified
(`certified_between_samples=False`). A Lipschitz bound on `h` alone does **not** bound the
full CBF residual, since `f`, `g`, and the gradient of `h` also vary. Supplying a valid
global `residual_lipschitz` makes the verifier subtract `L_residual * grid_radius` and mark
the result certified between samples. This avoids overstating the guarantee described by
the original methodology.

## Modeling cautions

- `min(h1, h2, ...) >= 0` encodes conjunction: every primitive must be safe.
- A positive weighted sum does not encode conjunction; one large positive primitive can
  hide another negative primitive. Use `wsum` only when that aggregate safe set is intended.
- `dist_to_obstacle` is a relative-degree-one barrier for translational unicycle control,
  but heading configurations tangent to the obstacle make its instantaneous control
  influence zero. More demanding dynamics may require higher-order CBFs.
- `boundary_margin(axis, bound, margin)` uses a signed bound: nonnegative `bound` means
  `x_axis <= bound`; negative `bound` means `x_axis >= bound`. Use `min` of the two sides
  to describe a bounded interval.

## Layout

```text
lawevo/
  dsl/       parser, AST, gradients, Lipschitz composition
  robot/     robot abstraction and unicycle adapter
  verify/    state grid, feasibility, alpha bisection
  filter/    CBF-QP projection
  sim/       dynamics rollout and metrics
  evolve/    population loop, belief space, prompt templates
examples/    runnable unicycle MVP
tests/       unit and integration tests
```
