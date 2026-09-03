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
2. The LLM proposes free-form mathematical expressions over the available signals; it never
   chooses numeric gains. A law is a symbolic expression tree (sums, products, unary
   functions `tanh/sin/cos/sqrt/square/abs/exp`, pairwise `min`/`max`, and signed numeric
   constants) with tunable `K` gain slots. Reusing a `K` name ties two places to one shared
   scalar. Laws are capped at 16 structural nodes, depth 5, and 12 distinct `K` slots.
3. Cross-Entropy Method (CEM) optimizes every gain `K` for both evolved expressions and
   classical baselines using the same simulation budget.
4. EoH-inspired `E1`/`E2` exploration and crossover plus `M1`/`M2`/`M3` mutation prompts
   generate structurally diverse, failure-directed, and generalization-oriented offspring.
5. An archive prevents duplicate expressions, while retry, deterministic local mutation,
   and per-generation checkpoints make long runs resumable when the model endpoint fails.
6. Controllers are compared on held-out initial states and physical-parameter variations,
   reporting task return, success, control energy, command jerk, and complexity.

This makes it possible to test a precise question: can task-aware symbolic evolution find
controller structures that outperform tuned P/PI/PD/PID, LQR, posture-feedback, or CPG
baselines without hiding the cost in actuator effort or nonsmooth commands?

The variation prompts adapt the five strategies from [Evolution of Heuristics
(EoH)](https://arxiv.org/abs/2401.02051): `E1` explores forms unlike multiple parents,
`E2` recombines their common backbone, `M1` changes structural signs or operators, `M2` targets an
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

## Symbolic expression representation

A controller law (or "structure") is a free-form symbolic expression over the task's
signals, not a fixed weighted-sum term list. It is written as a compact string, for example:

```
K1*tanh(K2*angle) + K3*angle*angular_velocity + min(K4*integral_angle, K5*angular_velocity)
```

Grammar: `+`, `-`, `*`, parentheses, unary functions `tanh sin cos sqrt square abs exp`,
pairwise `min(a, b)` / `max(a, b)`, signed numeric constants, and `K` gain tokens. Every
`K` token names one scalar parameter slot that CEM tunes; the LLM never proposes numeric
gains. Reusing a `K` name ties two places to one shared scalar (`K1*x + K1*y`), which is
one of the novel free-form degrees of freedom. A law is capped at 16 structural nodes,
depth 5, and 12 distinct `K` slots.

Signals are arrays broadcast over the action dimension (one component per actuator), so the
same law applies to any morphology. Evaluation guards against division by zero, overflow,
and non-finite values (clamping `exp`/`square` inputs and writing `NaN`/`inf` as zero).

`SymbolicExpression` (in `lawevo/pid/expression.py`) replaces the old flat `GymStructure`
term-list genome. It provides `to_expression_string()`, a canonical order-insensitive
`key()` for archive dedup, `parameter_count`, `complexity` (non-parameter node count), a
NumPy `evaluate(signals, gains)` and a batched Torch `evaluate_torch(signals, gains)` for
GPU rollouts, plus both a string parser and a JSON-tree parser. It owns an evaluator on the
standard pipeline of classical baselines, CEM tuning, and the LLM archive.

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
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openlux.ai/v1
OPENAI_MODEL=gpt-5.4-nano
```

The experiments read the first non-empty value among `OPENAI_*` and the legacy
`NVIDIA_*` variables (`OPENAI_*` takes priority). A base URL may be either an API
root (`.../v1`) or a full chat-completions URL; `/chat/completions` is appended
automatically when missing.

`.env` is ignored by Git. Environment variables and CLI flags can still override these
settings. Never commit a real API key.

## Running the experiments

### Benchmark protocol

The standard benchmark evolves structures for **20 generations**, tunes every evolved and
classical gain with the same CEM budget using **6 training episodes**, and reports each
controller over exactly **30 held-out test episodes**. The CLI defaults use this 20/6/30
protocol.

### Commands for all 26 environments

The 26 copy-paste commands below each run exactly one environment. They explicitly use
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

# 23. ManiSkill PushCube with Panda delta-pose control
py -m experiments.gymnasium_classical_benchmarks --environment PushCube-v1 --generations 20 --train-episodes 6 --test-episodes 30

# 24. ManiSkill PickCube with Panda delta-pose control
py -m experiments.gymnasium_classical_benchmarks --environment PickCube-v1 --generations 20 --train-episodes 6 --test-episodes 30

# 25. Genesis World PushCube with batched CUDA simulation
py -m experiments.gymnasium_classical_benchmarks --environment GenesisPushCube-v0 --genesis-batch-size 32 --generations 20 --train-episodes 6 --test-episodes 30

# 26. Genesis World PickCube with batched CUDA simulation
py -m experiments.gymnasium_classical_benchmarks --environment GenesisPickCube-v0 --genesis-batch-size 32 --generations 20 --train-episodes 6 --test-episodes 30
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
| `PushCube-v1` | Reach P/PD, Object Goal P, Contact+Goal PD |
| `PickCube-v1` | Reach P/PD, Pick+Place, Pick+Place PD |
| `GenesisPushCube-v0` | Reach P/PD, Object Goal P, Contact+Goal PD |
| `GenesisPickCube-v0` | Reach P/PD, Pick+Place, Pick+Place PD |

These are the environments currently implemented by LawEvo adapters. Other Gymnasium
environments can be added, but require an adapter defining observation-to-signal mapping,
action semantics, classical structures, success criteria, and task-specific prompt goals.

The Panda-Gym tasks use PyBullet and dense-reward `v3` environments with normalized
end-effector displacement control. Install all benchmark dependencies, including
`panda-gym==3.0.7`, with `py -m pip install -e ".[benchmarks]"`.

### ManiSkill commands

ManiSkill uses its standard single-environment CPU simulation, `state_dict` observations,
dense rewards, and the Panda `pd_ee_delta_pose` controller. Its action is normalized
`[delta_xyz, delta_axis_angle, gripper]`; the LawEvo signals command translation and the
gripper while leaving axis-angle rotation neutral. Install it with the other benchmark
dependencies (ManiSkill 3.x is required):

```powershell
py -m pip install -e ".[benchmarks]"
```

Then run either supported task:

```powershell
py -m experiments.gymnasium_classical_benchmarks --environment PushCube-v1 --generations 20 --train-episodes 6 --test-episodes 30
py -m experiments.gymnasium_classical_benchmarks --environment PickCube-v1 --generations 20 --train-episodes 6 --test-episodes 30
```

On Windows, ManiSkill supports CPU simulation but not GPU simulation. Rendering also
requires a working Vulkan setup; these headless state-based benchmarks do not request a
render mode.

### Genesis World GPU commands

Genesis World runs the Franka, cube, IK controller, and multiple independent rollouts on
the CUDA GPU. LawEvo batches the CEM population and training seeds into one reusable scene;
this is the accelerated path on Windows. The default batch size is 32, selected to fit a
4 GB GPU. Lower it to 8 or 16 if other applications are consuming VRAM, or raise it on a
larger GPU.

```powershell
py -m pip install -e ".[benchmarks]"

py -m experiments.gymnasium_classical_benchmarks `
  --environment GenesisPushCube-v0 `
  --genesis-batch-size 32 `
  --generations 20 `
  --train-episodes 6 `
  --test-episodes 30

py -m experiments.gymnasium_classical_benchmarks `
  --environment GenesisPickCube-v0 `
  --genesis-batch-size 32 `
  --generations 20 `
  --train-episodes 6 `
  --test-episodes 30
```

The first Genesis launch JIT-compiles its GPU kernels and can take roughly one minute on
this machine. Later evaluations in the same benchmark process reuse the compiled scene.
For a quick validation before the full 20-generation protocol:

```powershell
py -m experiments.gymnasium_classical_benchmarks `
  --environment GenesisPushCube-v0 `
  --genesis-batch-size 8 `
  --generations 2 --proposals 2 `
  --cem-iterations 2 --cem-population 8 `
  --train-episodes 2 --test-episodes 5
```

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

## MorpLaw: morphology × law co-evolution

MorpLaw co-evolves the robot's MJCF morphology and its symbolic control law. Each
individual is a **(morphology, structure) pair**; CEM tunes every pair's gains with an
equal simulation budget, and the pair is scored with mass-normalized energy plus a
morphology-cost penalty so bigger bodies cannot buy fitness with size or actuator
strength.

### Bidirectional experience

MorpLaw treats directed design knowledge as a first-class search object. Every proposal
contains an executable body or law plus a falsifiable hypothesis with an applicable
condition, recommendation, predicted metric effects, and mechanistic rationale. Evaluation
produces an immutable parent-to-offspring evidence record and updates one of two channels:

- `morph_to_law`: body mechanics and body-conditioned results guide controller motifs.
- `law_to_morph`: controller behavior and failure modes guide physical body changes.

Each channel has separate positive-insight and negative-pitfall banks. Retrieval uses soft
similarity over task, numeric body parameters, law terms, and metrics rather than requiring
an exact body/law JSON match. Retrieved knowledge receives downstream utility credit;
hypotheses progress through proposed, tested, supported, or refuted states.

A state-aware Navigator monitors stagnation, morphology/law diversity, operator validity,
operator improvement rates, and observed interactions. It issues explore, exploit, balance,
or joint-confirm directives while preserving the same proposal and evaluation protocol in
every ablation.

After the primary one-sided probes, MorpLaw asks for responsive laws specialized to the best
new body and responsive bodies specialized to the best new law. Counterfactual evaluations
complete the factorial quartet `(M,L)`, `(M',L)`, `(M,L')`, `(M',L')`, yielding the explicit
interaction term `I = F11 - F10 - F01 + F00`. Positive interaction indicates co-adaptation;
negative interaction exposes an incompatibility hidden by one-sided rankings.

Only four knowledge ablations are supported. All four retain the same Navigator,
counterfactuals, CEM budget, and LLM-call protocol:

- `no_knowledge`: record evidence but retrieve or accumulate neither channel.
- `m_to_l`: enable only morphology-to-law knowledge.
- `l_to_m`: enable only law-to-morphology knowledge.
- `full`: enable both directed channels and both positive/pitfall banks.

The on-disk evaluation cache is shared only to avoid recomputing an identical pair. Each
variant has an isolated search archive, elite set, knowledge base, and Navigator state, so a
later ablation cannot see candidates or guidance discovered by an earlier one.

### Morphology representation

Morphology fields are substituted into vendored, parameterized MJCF assets
(`lawevo/morplaw/assets/`) with coupled geometry rules (a longer thigh moves the leg
body). `MorphologyTemplate.compile` is the MuJoCo validity gate, and the rendered XML is
cached under the system temp directory. Three template families ship:

- **Parametric** (topology fixed; observation/action sizes never change): `walker2d`
  (8 fields), `reacher` (7), `reacher_payload` (9), `reacher_gravity` (7),
  `reacher_precision` (7), `pusher` (6), `hopper` (8), `half_cheetah` (8), `swimmer`
  (6), and `ant` (6).
- **Topology-changing** (count fields change the joint/actuator count and therefore the
  observation/action dimensions; the law space is unchanged because laws are expression
  trees over vector-valued signals): `swimmer_topology` (`n_links` 3..6) and
  `ant_topology` (`n_legs` 4..6). The locomotion adapters derive their per-actuator
  patterns from the live action dimension, and `morph_cost` penalizes count fields per
  added unit.
- **Grammar-native**: `robomorph_flat`, `robomorph_ridged`, `robomorph_frozen_lake`, and
  `robomorph_beams` evolve a complete module graph rather than fields on an Ant template. A
  graph contains 1..4 serial body modules connected by rigid/roll/twist joints. Any body
  module may carry a compiler-mirrored bilateral limb with 1..3 links,
  rigid/roll/knee/elbow joints, and a foot or passive-wheel terminal. The LLM may make
  non-local graph mutations or elite crossovers; bounds, symmetry, the 2..16 actuator limit,
  MJCF compilation, and one forward dynamics step form deterministic validity gates.

For grammar search, MorpLaw starts from three reproducible randomly sampled valid body graphs
and gives the morphology generator the highest-scoring **unique** body graphs as best-shot
examples. The topology-agnostic locomotion adapter reads MuJoCo's live actuator-to-joint map,
so symbolic terms and CEM work when the graph changes joint count, joint order, or contains
unactuated wheels. This adopts RoboMorph's grammar-generation and best-shot ideas while
retaining MorpLaw's interpretable law co-evolution, directed knowledge, one-sided
counterfactuals, and factorial interaction measurements. The four environments reproduce
the geometry and friction parameters of RoboMorph's
[official terrain suite](https://github.com/kevinxqiu/robomorph/tree/main/train/envs): flat
ground, 15 ground-level cylindrical ridges, a friction-0.05 frozen lake, and 15 cylindrical
beams centered 0.5 m above the floor. Obstacles are mirrored onto the positive x-axis because
Gymnasium Ant rewards positive-x travel, while RoboMorph's environment rewards negative-x
travel. This remains a MorpLaw symbolic-controller/CEM benchmark, not a reproduction of
RoboMorph's SAC/Brax training pipeline.

The PID-friendly arm suite separates four control regimes. `reacher_payload` adds an
evolvable concentrated endpoint load; `reacher_gravity` rotates gravity into the arm's
motion plane; `reacher_precision` tightens success to 0.02 m and doubles the settling
horizon to 100 steps; and `pusher` co-designs a seven-joint arm for contact-rich object
pushing. Reacher task features are computed from MuJoCo's live actuator-to-joint map and
body Jacobian, while Pusher exposes tip-to-object, object-to-goal, combined push, damping,
integral, and posture terms.

### Running

```powershell
py -m experiments.evolve_morplaw --environment reacher
py -m experiments.evolve_morplaw --environment reacher_payload
py -m experiments.evolve_morplaw --environment reacher_gravity
py -m experiments.evolve_morplaw --environment reacher_precision
py -m experiments.evolve_morplaw --environment pusher
py -m experiments.evolve_morplaw --environment walker2d
py -m experiments.evolve_morplaw --environment swimmer_topology
py -m experiments.evolve_morplaw --environment ant_topology
py -m experiments.evolve_morplaw --environment robomorph_flat
py -m experiments.evolve_morplaw --environment robomorph_ridged
py -m experiments.evolve_morplaw --environment robomorph_frozen_lake
py -m experiments.evolve_morplaw --environment robomorph_beams
```

Defaults: 5 generations, 4 primary proposals per side, 1 responsive proposal per side,
2 independently ranked joint probes, CEM 5 iterations × 24 population, 6 training episodes,
30 held-out episodes, 24 items per knowledge bank, and top-3 retrieval from each polarity.
Grammar templates additionally use 3 initial best-shot body graphs (`--grammar-seeds`).
One invocation runs `no_knowledge`, `m_to_l`, `l_to_m`, and `full`, and writes
`results/morplaw_<environment>/` with the comparison plot,
`results.json`, `nim_responses.json`, and a resumable `records.jsonl` pair cache.
`results.json` includes the evidence ledger, four knowledge banks, hypothesis lifecycle,
factorial interactions, Navigator decisions, operator statistics, actual episodes computed,
and the cache-independent requested episode budget:

```powershell
py -m experiments.evolve_morplaw --environment walker2d --resume
py -m experiments.evolve_morplaw --environment reacher --variants no_knowledge full
```

MorpLaw tests require the benchmarks extras; they are skipped automatically on a
NumPy-only install (`pytest.importorskip`).

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
