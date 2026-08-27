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
4. An archive prevents duplicate structures, while retry, deterministic local mutation,
   and per-generation checkpoints make long runs resumable when the model endpoint fails.
5. Controllers are compared on held-out initial states and physical-parameter variations,
   reporting task return, success, control energy, command jerk, and complexity.

This makes it possible to test a precise question: can task-aware symbolic evolution find
controller structures that outperform tuned P/PI/PD/PID, LQR, posture-feedback, or CPG
baselines without hiding the cost in actuator effort or nonsmooth commands?

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

### Suite: `classical`

Runs task-specific classical-control and MuJoCo manipulation environments: `Pendulum-v1`,
`InvertedPendulum-v5`, `Reacher-v5`, and `Pusher-v5`.

```powershell
py -m experiments.gymnasium_classical_benchmarks `
  --suite classical `
  --generations 20 `
  --proposals 6 `
  --proposal-attempts 3 `
  --cem-iterations 10 `
  --cem-population 32 `
  --train-episodes 6 `
  --test-episodes 30
```

### Suite: `locomotion`

Runs the MuJoCo locomotion environments `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`,
`Ant-v5`, and `Humanoid-v5` against tuned Posture P, Posture PD, CPG, and CPG+PD baselines.

```powershell
py -m experiments.gymnasium_classical_benchmarks `
  --suite locomotion `
  --generations 20 `
  --proposals 6 `
  --proposal-attempts 3 `
  --cem-iterations 10 `
  --cem-population 32 `
  --train-episodes 6 `
  --test-episodes 30
```

### Suite: `all`

Runs both suites in one timestamped benchmark:

```powershell
py -m experiments.gymnasium_classical_benchmarks `
  --suite all `
  --generations 20 `
  --proposals 6 `
  --proposal-attempts 3 `
  --cem-iterations 10 `
  --cem-population 32 `
  --train-episodes 6 `
  --test-episodes 30
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
  --suite all `
  --resume-run 20260827_231500 `
  --generations 20 `
  --proposals 6 `
  --cem-iterations 10 `
  --cem-population 32 `
  --train-episodes 6 `
  --test-episodes 30
```

### Supported Gymnasium environments

| Suite | Environment | Tuned classical baselines |
|---|---|---|
| `classical` | `Pendulum-v1` | P, PI, PD, PID |
| `classical` | `InvertedPendulum-v5` | P, PI, PD, PID, LQR |
| `classical` | `Reacher-v5` | task-space P, PI, PD, PID |
| `classical` | `Pusher-v5` | task-space P, PI, PD, PID |
| `locomotion` | `Hopper-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `locomotion` | `Walker2d-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `locomotion` | `HalfCheetah-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `locomotion` | `Ant-v5` | Posture P, Posture PD, CPG, CPG+PD |
| `locomotion` | `Humanoid-v5` | Posture P, Posture PD, CPG, CPG+PD |

These are the environments currently implemented by LawEvo adapters. Other Gymnasium
environments can be added, but require an adapter defining observation-to-signal mapping,
action semantics, classical structures, success criteria, and task-specific prompt goals.

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
