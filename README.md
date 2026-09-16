# LawEvo

LawEvo is an executable research prototype for knowledge-augmented evolution of
interpretable robot controllers and control-barrier functions (CBFs). It combines an LLM
outer loop that proposes compact symbolic controller structures with a numerical inner
loop that optimizes every gain under an equal evaluation budget.

## Robosuite Panda: Lift, DoorUnlocked, Wipe

Three additional LawEvo adapter IDs are available with the existing benchmark CLI
(robosuite 1.5.2 is included in the `benchmarks` extra):

```powershell
py -m experiments.gymnasium_classical_benchmarks --environment RobosuiteLiftNominal-v0
py -m experiments.gymnasium_classical_benchmarks --environment RobosuiteDoorUnlocked-v0
py -m experiments.gymnasium_classical_benchmarks --environment RobosuiteWipe-v0
```

These IDs belong to the LawEvo CLI, not `gymnasium.make`. Existing
`RobosuiteLift-v0`, Stack, NutAssembly and latched Door adapters remain available.
The new Lift uses stock mass / inertia without the legacy adapter's added mass
jitter. Door uses the native `use_latch=False` option. Wipe retains stock dirt,
friction, reward and early termination rules. Reset seeds are applied to the
generator shared by scene samplers, without replacing its object reference.

| Adapter | Horizon at 20 Hz | Baselines | Final success / SG |
| --- | --- | --- | --- |
| LiftNominal | 300 | Staged OSC P, PD, PID | Native cube height > table + 0.04 m; normalized height deficit / 0.04 m |
| DoorUnlocked | 300 | Staged OSC P, PD, PID | Native positive hinge angle > 0.3 rad; angle deficit / 0.3 rad |
| Wipe | 500 | Waypoint Wipe P, PD, PID | Native all markers wiped; fraction of markers still dirty |

SG is clipped to [0,1]; strict height / angle boundary failures retain a minimum
gap of 1e-12. Episode success is exactly SG=0. Q averages three milestones:
Lift (near cube once, native grasp once, final success), Door (near handle once,
half-open once, final success), Wipe (some cleaned, half cleaned, all cleaned).
Return, SR, SG, Q, integrated squared action and action slew are exported through
the existing reports. Selection remains SR first, then SG, then secondary utility;
Q does not select controllers. Secondary reference scales are horizon for return,
10 for command effort and 100 for action slew.

All three use fixed OSC_POSE, kp=150 and damping ratio=1. Actions are normalized
world-frame delta pose commands (0.05 m translation / 0.5 rad rotation), not
torques. Lift / Door have seven channels including gripper; Wipe has six and a
non-actuated wiping tool. Signals are dimensionless after explicit normalization,
with complete formula, shape, sign, zero-condition and memory contracts in prompts.

Baseline and evolved laws share hand-written helpers: Lift approach/close/lift,
Door approach/close/hinge-tangent tracking, and Wipe a nearest-unwiped-marker
waypoint scheduler. Wipe uses simulator marker positions and wipe flags, holds
the tool face horizontal, and presses through a 3 mm downward position offset;
it does not implement force feedback. Helper complexity is not counted in AST
size. This suite evaluates feedback around those helpers, not discovery of the
entire manipulation strategy.

Initial untuned P smoke rollouts achieved Lift and Wipe success on seed 31.
Door's untuned baseline did not succeed on that seed. These are integration checks,
not a measured success-rate claim; use the normal CEM and held-out evaluation to
assess baseline strength, especially Door.

## Controller barrier audit for the ten-task suite

Test evaluation automatically audits all baseline and evolved controllers for
InvertedPendulum/Pulse, Reacher/Payload, PandaReach/MovingSlow,
PandaPush/LowFriction and PandaSlide/FrictionShift. Reports are embedded in test
results and written to `summary/barrier_verification.json`. The selected
controller's report is also carried in its `test` record. Existing training
selection stays SR -> SG -> secondary utility; the audit does not rerank,
filter, modify actions, or add rollouts. It checks existing test trajectories.

`ControllerBarrierVerifier` is a **sampled executed-controller audit**, separate
from the existing model-based `BarrierVerifier` (which tests CBF feasibility).
Neither name means the ten-task controller has a global stability certificate.

| Pair | Barrier envelope |
| --- | --- |
| InvertedPendulum / Pulse | h=1-(theta/0.2 rad)^2, native upright episode boundary |
| Reacher / Payload | Per-joint h=1-(qdot/20 rad/s)^2; 20 rad/s is a declared engineering choice, not a native joint limit |
| PandaReach / MovingSlow | Seven arm joint position ranges read from the robot model |
| PandaPush / LowFriction | Arm joint ranges plus object AABB inside table AABB in xy, and object top above table top |
| PandaSlide / FrictionShift | Same checks, using Slide's actual larger table geometry |

Joint-position barriers use h=1-((q-midpoint)/half_range)^2. Table edge margins
are divided by table width; the fall margin is divided by 0.1 m. These are
restricted audit envelopes, not full robot safety specifications. In particular,
they do not establish collision freedom, acceptable force, task success or
robustness to disturbances beyond the simulated trajectory.

At each control-step endpoint the audit checks h>=-1e-6 and the discrete residual
`h_next - exp(-5*dt)*h >= -1e-6` for constraints whose previous state was inside
the envelope. The decay rate 5/s is fixed, not tuned after observing failures.
A negative residual can occur **without leaving the envelope**; the JSON reports
those separately. It also checks initial margins, raw action NaN/Inf before
replacement, finite signals and action shape (including legal scalar broadcast).
Action clipping is counted separately and is not by itself a failure.

Reports include seed, step, first failure, minimum margin/residual and
`certified_between_samples=false`. No substep or unseen-state guarantee is claimed.
The normal evaluator's numerical fallback remains in place; the audit exposes
its use instead of silently treating it as verified behavior.

Audit an already saved controller without rerunning CEM or calling an LLM:

```powershell
py -m experiments.verify_controller --environment Reacher-v5 --controller "results/<run>/Reacher-v5/summary/selected_controller.json" --episodes 30 --output "results/reacher_barrier_audit.json"
```

Choose the matching environment explicitly (saved law files do not always carry
the environment ID). Default audit seeds start at 90000; `--seed` changes that.
If you use audit outcomes to redesign or select laws, treat those seeds as
validation data and reserve new seeds for the final test.

## Genetic Programming baseline

The [baseline folder](baseline/README.md) provides tree GP with tournament selection,
subtree crossover/mutation, and the same CEM tuner and evaluation protocol as LawEvo.
Run `py -m baseline.genetic_programming --environment Reacher-v5 --gp-seed 1`.
Results are under `results/<env_id>/<timestamp>/data/gp/`; no LLM API is required.

## ManiSkill drawing tasks

```powershell
py -m experiments.gymnasium_classical_benchmarks --environment DrawTriangle-v1
py -m experiments.gymnasium_classical_benchmarks --environment DrawSVG-v1
```

Registry keys are `maniskill_draw_triangle` and `maniskill_draw_svg`.
Both use PandaStick with six delta-pose action channels (no gripper), state
observations, native sparse rewards, and P/PD/PID path-following baselines.
For single-robot CPU drawing, the PyTorch IK fallback uses a cached NumPy
serial-chain Jacobian. The IK solver, simulation timestep, rewards, and CEM
budget are unchanged; batched or autograd Jacobian calls retain the upstream implementation.
Triangle runs for 300 steps; SVG runs for 500 steps using ManiSkill's default
continuous outline. Install `pip install -e ".[benchmarks]"` to include
`svgpathtools`, required by DrawSVG.

A shared fixed scheduler approaches above the first vertex, lowers the tip,
then traces interpolated segments. This helper is not evolved. Native success
checks coverage and off-outline dots; SVG's native 0.1 m tolerance is loose,
so success is not a precision-drawing guarantee. SG/Q and audited signal units
are not defined for these adapters, consistent with the existing ManiSkill tasks.

## Three lightweight custom Panda contact tasks

These LawEvo CLI adapters use panda-gym's Panda robot and PyBullet DIRECT with
state observations, simple collision geometry and no image rendering. They
reuse the existing `benchmarks` dependencies; no new simulator is required.

```powershell
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaDrawer-v0
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaPlanarWipe-v0
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaPegInsertionEasy-v0
```

These are custom tasks, not stock Panda-Gym environments or `gymnasium.make`
registrations. All run for 150 control steps at 25 Hz (20 physics substeps per
action), with seeded task placement and fixed physics. Success does not end an
episode early. Each has P, PD and PID structures tuned by the existing equal-budget
CEM, complete typed signal contracts and SR/SG/Q/return/effort/action-slew output.
Selection stays SR -> SG -> secondary utility. No full CEM/LLM training was run
as part of integration.

| Task | Physical scene and task success | Fixed baseline support |
| --- | --- | --- |
| Drawer | Passive prismatic drawer, travel 0–0.18 m, real gripper contact; final opening >=0.12 m | Approach above handle, descend, close for 0.4 s, pull along -x; no synthetic grasp attachment |
| PlanarWipe | 50 mm square mounted pad and nine logical dirt markers; all markers must have contacted its footprint while table contact force >0.1 N | Nearest uncleared marker scheduler, approach then 2 mm downward position offset; no force feedback |
| PegInsertionEasy | Mounted 24 mm diameter peg and four collision walls defining a 36 mm square hole; final radial tip error <=4 mm, tip z in [5,40] mm | Fixed downward orientation; align above hole, lower in 5 mm target increments, retry alignment if radial error exceeds 5 mm |

The wipe markers are task-state points rather than particle bodies; clearing
requires physical pad/table contact and marker inclusion in the pad's actual
frame. Wipe contact is checked at every physics substep until all markers are
clean. Drawer checks every substep until first contact is recorded. Afterwards,
and for Peg throughout, only the final substep's contact force is queried.
All physics substeps still run. Observations, rewards, progress tracking and
controller features share one refreshed state snapshot per control step.
Peg walls have real collision
geometry; the peg starts mounted, so this task excludes grasp acquisition.
Pad and peg are attached with physical fixed constraints, not teleported during
steps. The Panda's internal IK holds a downward orientation and maps normalized
xyz commands to 0.05 m displacement. Drawer additionally has one gripper channel
(positive opens, negative closes). Baseline and evolved laws share the same
privileged state and helper memory; helper complexity is excluded from AST size.

SG is normalized opening deficit for Drawer, fraction of uncleared markers for
Wipe, and mean normalized lateral/depth/floor deficits for Peg. SG=0 exactly
matches task success. Q averages two intermediate milestones achieved at any
point and final success. The exact formulas and normalizers are recorded in the
run protocol and prompt. Secondary reference scales are 150 for return, 5 for
command effort, 100 for action slew. Contact force is a diagnostic, not a force
limit or a safety certificate. The ten-task barrier audit has not been extended
to these custom scenes.

Real-physics reference rollouts with unit P gains succeeded on seed 31 for all
three tasks. This confirms feasibility for that seed, not a held-out success-rate
claim. Tests also check seeded reset, actual wipe contact, peg-wall collision and
CLI report export. The Drawer URDF is included as package data.

For a short pipeline run, append:

```powershell
--generations 2 --proposals 2 --cem-iterations 2 --cem-population 8 --train-episodes 2 --test-episodes 5
```

## MovingSlow v1: 1 cm tracking radius

Use `LawevoPandaReachMovingSlow-v1` for the stricter tracking benchmark:

```powershell
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaReachMovingSlow-v1
```

Version 1 changes the tracking and final-position radius from 0.05 m to 0.01 m.
SR, SG and Q use the same new radius. Orbit amplitudes (0.025 m x, 0.0125 m y),
period (5 s), horizon (150 steps), warmup (10 steps), required tracking rate
(60%), dense reward, physics, signals and baseline structures remain unchanged.
The native task's success-distance threshold is also 0.01 m. The sampled joint
barrier audit supports v1 with the same envelope as v0.

The entire orbit fits inside a 0.025 m ball, so staying at its center can pass
v0's 0.05 m criterion. This stationary-center trajectory fails v1. This fixes
that metric loophole; it does not guarantee that well-tuned classical feedback
will find v1 difficult. `LawevoPandaReachMovingSlow-v0` retains its old definition
for existing runs. Use a new run with the v1 ID rather than resuming v0 as v1.

Radius validation with frozen gains from `20260906_095324` (no retuning), on
30 seeds starting at 90000: Task P/PD 27/30, PID 24/30, PI/Saturated PD/
Feedforward P 30/30. Tracking PD was 0/30 with its previously selected all-zero
gains, so that row does not establish the capability of a properly tuned tracking
controller. Full metrics and gains: `results/analysis/moving_slow_v1_baseline_check.json`.
This is a validation check of the new criterion, not a new training run.

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

## Success-first selection and typed signal contracts

All ten controlled-suite tasks also report Q, the mean fraction of achieved
progress milestones per episode, averaged over evaluation seeds. Q is diagnostic
only and does not enter the selection key. The four MuJoCo tasks use three equally
weighted milestones each:

| Task pair | Q milestones |
| --- | --- |
| Reacher / Payload | Ever within 0.10 m of the goal; ever within the 0.05 m success radius; finally within the success radius. All comparisons are strict `<`, on post-step fingertip-target xy distances. |
| InvertedPendulum / Pulse | Survive the first `ceil(H/2)` steps without termination; survive exactly H steps without termination; finish with absolute pole angle strictly below 0.1 rad. |

For these four tasks, each episode has Q in `{0, 1/3, 2/3, 1}`, and Q=1 iff
the episode succeeds. Reacher's near radius is twice its success tolerance so
subclasses with a different tolerance stay consistent. Reaching briefly and then
drifting away earns 2/3; falling after the halfway milestone does not erase that
milestone. No samples means no milestones achieved. Existing Panda Q definitions
are retained. The new milestones are declared in `progress_spec` and included in
prompts, manifests and result protocols; resume rejects a missing or changed
definition for the four affected tasks. An already-running process must be
restarted in a new run to load the new Q definitions.

All ten tasks in the controlled suite define Success Gap (SG). For each episode,
SG is the mean of its normalized constraint violations in [0,1]; the reported
aggregate SG is the mean over evaluation seeds. Episode SG is exactly zero iff
that episode succeeds. Aggregate SG is zero iff all evaluated episodes succeed.

| Task pair | Per-episode violations averaged into SG |
| --- | --- |
| InvertedPendulum / Pulse | Survival deficit `max((H-steps)/H, 1/H if terminated else 0)` and final-angle excess above 0.1 rad divided by 0.1 rad; each clipped to [0,1] |
| Reacher / Payload | Final fingertip-target xy distance excess above 0.05 m divided by 0.20 m, clipped to [0,1] |
| PandaReach | Final hand-goal distance excess above the task tolerance divided by 0.15 m, clipped to [0,1] |
| PandaReach MovingSlow | Post-warmup tracking-rate shortfall below 0.6 divided by 0.6, and final distance excess above 0.05 m divided by 0.15 m; each clipped to [0,1] |
| PandaPush / LowFriction | Final object-goal distance excess above the task tolerance divided by 0.20 m, clipped to [0,1] |
| PandaSlide / FrictionShift | Same position-gap normalization as Push (0.20 m), preserving the existing Panda metric |

The MuJoCo tasks keep their original strict final bounds (`angle < 0.1`,
`distance < 0.05`). At an exact boundary their gap has a positive floor of 1e-12,
so failure cannot be mistaken for zero gap. Termination on the last cart-pole
step also has a positive survival violation. Panda bounds remain inclusive as
in the existing benchmark. Reacher's criterion remains final position only;
no new velocity, survival or settling requirement is introduced. The SG formulas
and constants are included in the LLM prompt, run manifest and result protocol.
Runs created before protocol `success-first-typed-sg-v2` must be started afresh
because their cached rankings may omit SG.

The Gymnasium/Panda benchmark (`experiments.gymnasium_classical_benchmarks.py`)
uses the exact lexicographic key `(success_rate, -sg, secondary_score)`, descending.
A lower measured SR can never win through return, effort, smoothness or simplicity.
SG is compared only when SR ties exactly; if a task has no SG, that coordinate is
neutral. Q remains a diagnostic. There is no SR threshold, tolerance bucket or
weighted success penalty. This is a guarantee about selection on the evaluated
training seeds, not a guarantee of unchanged success on unseen seeds.

The same key is used by both CEM paths, best-gain retention, outer elite selection,
and final controller selection. `score` in JSON/CSV is now **secondary utility**,
not return and not the overall ordering. The secondary utility is:

```text
r = 0.5 + atan(return / return_scale) / pi
cost(x, scale) = x / (x + scale)
secondary_score = 0.60*r - 0.20*cost(effort, effort_scale)
                 - 0.15*cost(slew, slew_scale) - 0.05*cost(nodes, 16)
```

Scales are fixed per task family in `lawevo/pid/objective.py`; a base and its
controlled variant share the same scales. Defaults are initial engineering
settings, not calibrated optimal weights. They never depend on the current
population or test outcomes. Both scales and weights are saved in each manifest.
The legacy `energy` column means integrated squared action (command effort), and
`jerk` means integrated squared action slew; neither denotes measured joules or
mechanical jerk for Panda displacement commands.

`summary/selected_controller.json` is the usable controller selected on training
metrics, including classical incumbents and nominal LQR where applicable.
`lawevo/best_controller.json` remains the best **evolved-only** comparison; it may
be worse than the selected baseline. Test metrics are reported only and never used
to select either controller.

The five base tasks and five controlled variants have complete signal contracts
in `lawevo/pid/signal_schema.py`. Each records semantic category, scalar/vector
shape, SI units, formula/sign, coordinate frame, action mapping, zero conditions,
bounds (or explicitly no adapter clipping), and hand-designed memory/phase logic.
Runtime feature shapes are checked at the first step. The expression checker
infers metre/second exponents for all gains jointly; angles retain their semantic
category but radians are dimensionless for SI checks. It supports scalar broadcast,
requires compatible units for sums/min/max, and requires dimensionless inputs to
`tanh/sin/cos/exp`. The output is a normalized actuator command, not a measured force.
For example, `K1*goal_error + K2*eef_damping` is valid with different gain units,
whereas `K1*(goal_error + eef_damping)` is invalid. A literal zero is unit-polymorphic.
Where unit equations are underdetermined, the checker reports one consistent
assignment and the number of free unit variables; it does not invent unique units.
Legacy tasks outside this audited suite explicitly report an incomplete contract
and retain signal-name checks rather than fabricated physical metadata.

Prompts include the contracts, fixed objective configuration, tuned elite gains,
and inferred gain units. Type/parser rejection reasons are returned on retries.
The existing phase helpers remain hand-designed, not discovered by the LLM.

Law JSON format 2 treats the AST as authoritative and persists gain-slot identity.
Text rendering preserves grouping and numeric constants, and archive keys preserve
parameter sharing while ignoring gain names and commutative operand order.
Legacy expression-only files remain readable; historical lossy strings cannot be
assumed to reconstruct the original controller. Resume rejects old protocol files,
changed signal/objective contracts, and changed training/CEM evaluation budgets
before overwriting the manifest. Start a new run for those changes.

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
episodes. No output path is required. Every invocation creates a timestamped run grouped by
environment ID (`problem` means `env_id`):

```text
results/<env_id>/<YYYYMMDD_HHMMSS>/data/
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
`<env_id>` is the environment ID, for example `DrawSVG-v1`, `DrawTriangle-v1`,
or `InvertedPendulum-v5`. There is no repeated environment folder under the timestamp.

The same metrics are printed to the console when the run finishes. To resume a timestamped
run, provide its directory name (the bare timestamp also works when the run lives directly
under a legacy `results/<registry_key>/<timestamp>/` or `results/<timestamp>/`
location; those existing runs retain their original output layout):

```powershell
py -m experiments.gymnasium_classical_benchmarks `
  --environment Ant-v5 `
  --resume-run Ant-v5/20260827_231500 `
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
| `LawevoPandaReachMoving-v0` | Task P/PD, Feedforward P, Tracking PD |
| `LawevoPandaPushIce-v0` | Reach PD, Contact+Goal PD, Obstacle-aware PD |
| `LawevoPandaSlideGate-v0` | Reach PD, Object Goal P, Through-gate P/PD |
| `LawevoPandaPickDistractor-v0` | Reach P/PD, Pick+Place, Selective Pick PD |
| `LawevoPandaStackNarrow-v0` | Reach P/PD, Pick+Stack, Settle PD |
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

### Harder Panda-Gym arm variants

Beyond the five stock tasks above, LawEvo ships five difficulty-graded variants
(`lawevo/pid/panda_gym_variants.py`). Each variant subclasses a Panda-Gym
task/environment with one exposed physical parameter, so tuned classical
baselines stay honest while evolved laws gain structure to exploit:

| Variant | Environment id | Exposed parameter | What it changes |
|---|---|---|---|
| Reach-MovingGoal | `LawevoPandaReachMoving-v0` | `goal_speed` | The goal orbits a resampled center; success tracks the moving goal. Feedforward (`goal_velocity`, phase oscillators) beats reactive PD. |
| Push-IceObstacle | `LawevoPandaPushIce-v0` | `table_friction` (default 0.1) | Near-frictionless table plus a static obstacle between the cube's start zone and the goal zone. `obstacle_repel` bends push paths. |
| Slide-Gate | `LawevoPandaSlideGate-v0` | `gate_width` (default 0.09 m) | Two walls gate the straight path; the puck must pass through before the goal. `through_gate` stages the shot. |
| Pick-HeavyDistractor | `LawevoPandaPickDistractor-v0` | `cube_mass` (default 1.5 kg) | A heavier cube sags under transport while a resampled clutter box sits near the goal. `distractor_error` steers around it. |
| Stack-NarrowSettle | `LawevoPandaStackNarrow-v0` | `distance_threshold` (0.025 m) + `settle_speed` (0.08 m/s) | Tight tolerance plus an at-rest requirement: a dropped-in-place cube that is still moving fails. `settle_velocity` gates release. |

Each variant's adapter adds its new signals to the stock adapter's term list and
ships its own tuned classical baselines (for example a feedforward Tracking PD
for Reach-MovingGoal and a velocity-gated Settle PD for Stack-NarrowSettle), so
the same 20/6/30 protocol compares against classical laws that already use the
new physics. The variant env id and parameter defaults live in the adapter
constructors; pass custom values through `gym.make` kwargs if needed.

Run each variant with the standard protocol:

```powershell
# Reach-MovingGoal: track an orbiting goal
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaReachMoving-v0 --generations 20 --train-episodes 6 --test-episodes 30

# Push-IceObstacle: low-friction table plus an obstacle
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaPushIce-v0 --generations 20 --train-episodes 6 --test-episodes 30

# Slide-Gate: strike the puck through a gated wall pair
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaSlideGate-v0 --generations 20 --train-episodes 6 --test-episodes 30

# Pick-HeavyDistractor: heavier cube plus clutter near the goal
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaPickDistractor-v0 --generations 20 --train-episodes 6 --test-episodes 30

# Stack-NarrowSettle: tight tolerance plus an at-rest requirement
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaStackNarrow-v0 --generations 20 --train-episodes 6 --test-episodes 30
```

Quick validation of one variant before the full protocol:

```powershell
py -m experiments.gymnasium_classical_benchmarks `
  --environment LawevoPandaReachMoving-v0 `
  --generations 2 --proposals 2 `
  --cem-iterations 2 --cem-population 8 `
  --train-episodes 2 --test-episodes 5
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

### PyBullet variant templates (harder Panda-Gym arms, evolvable robot bodies)

MorpLaw also co-evolves the five harder Panda-Gym variants described above
(`lawevo/pid/panda_gym_variants.py`). These are **PyBullet URDF templates**
(`lawevo/morplaw/panda.py`): the morphology is the **Panda arm itself** —
rendered per individual from a parametric URDF — plus, for the variant tasks,
one physical environment parameter:

| Template | Robot-body fields | Environment field(s) |
|---|---|---|
| `panda_reach_moving` | arm shape + masses + `motor_force` | `goal_speed` 0.02–0.15 m/s |
| `panda_push_ice` | arm shape + masses + `motor_force` | `table_friction` 0.02–0.5 |
| `panda_slide_gate` | arm shape + masses + `motor_force` | `gate_width` 0.06–0.20 m |
| `panda_pick_distractor` | arm shape + masses + `motor_force` | `cube_mass` 1.0–3.0 kg |
| `panda_stack_narrow` | arm shape + masses + `motor_force` | `distance_threshold` 0.015–0.05 m, `settle_speed` 0.03–0.2 m/s |

The arm-shape fields are `base_height` (0.28–0.40 m), `upper_arm_len`
(0.25–0.42 m), `forearm_len` (0.30–0.48 m), `wrist_len` (0.06–0.13 m),
`shoulder_offset` (−0.12 to −0.05 m), per-link masses `mass_link1..7`, and
`motor_force` (0.6–1.6×) which scales every joint's position-control force —
PyBullet's gear. `MorphablePanda` loads the rendered URDF; the default spec
reproduces the stock Panda exactly (total mass 17.96 kg), and longer arms
measurably change forward kinematics, so body evolution is physically real.

`make_morph_env` detects URDF templates and forwards `urdf_path`,
`motor_force`, and the environment parameters as `gym.make` kwargs, so the
same MorpLaw engine — law/morphology generators, directed knowledge channels,
Navigator, CEM tuning, factorial counterfactuals, and the four knowledge
ablations — runs unchanged. Each variant carries task/morphology/
term-semantics prompt context in `lawevo/morplaw/tasks.py`, so the LLM
reasons about co-design trade-offs like a longer arm needing stronger motors
to track a faster goal.

Run the PyBullet co-design tasks like any MorpLaw environment:

```powershell
# ===== The five harder variants (arm shape + environment parameter + law) =====
py -m experiments.evolve_morplaw --environment panda_reach_moving
py -m experiments.evolve_morplaw --environment panda_push_ice
py -m experiments.evolve_morplaw --environment panda_slide_gate
py -m experiments.evolve_morplaw --environment panda_pick_distractor
py -m experiments.evolve_morplaw --environment panda_stack_narrow

# ===== Morphable-robot versions of the five stock tasks (arm shape + law) =====
py -m experiments.evolve_morplaw --environment panda_reach_morph
py -m experiments.evolve_morplaw --environment panda_push_morph
py -m experiments.evolve_morplaw --environment panda_slide_morph
py -m experiments.evolve_morplaw --environment panda_pick_and_place_morph
py -m experiments.evolve_morplaw --environment panda_stack_morph
```

The `*_morph` environments replay the stock Reach/Push/Slide/PickAndPlace/Stack
tasks unchanged but load the arm from the same parametric URDF, so all ten
Panda tasks co-evolve the robot body alongside the control law.

Quick validation of one variant before the full protocol:

```powershell
py -m experiments.evolve_morplaw --environment panda_reach_moving --generations 1 --proposals-per-side 1 --responsive-per-side 0 --cem-iterations 1 --cem-population 4 --train-episodes 2 --test-episodes 3 --variants no_knowledge
```

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

## Controlled classical benchmark: five base tasks and five variants

The benchmark CLI includes the following single-factor pairs. Defaults below
are starting settings for a pilot, not measured claims about task difficulty.

| Base environment | Variant environment | Only variant change | Classical comparisons |
| --- | --- | --- | --- |
| `InvertedPendulum-v5` | `LawevoInvertedPendulumPulse-v0` | +5 N horizontal cart force at step 250, for one control interval | State-feedback PD (`PD`), nominal analytic LQR |
| `Reacher-v5` | `LawevoReacherPayload-v0` | Centered fingertip payload: 15% of distal link mass, with matching spherical inertia | Jacobian-transpose PD, saturated PD |
| `PandaReachDense-v3` | `LawevoPandaReachMovingSlow-v0` | Slow planar goal motion, 5 s period, x/y amplitudes 2.5/1.25 cm | Cartesian PD/PID, saturated PD; Feedforward P and Tracking PD on moving goal |
| `PandaPushDense-v3` | `LawevoPandaPushLowFriction-v0` | Table lateral friction multiplied by 0.75 | Waypoint Push P/PD |
| `PandaSlideDense-v3` | `LawevoPandaSlideFrictionShift-v0` | Table lateral friction multiplied by 1.25 | Align-Strike-Retract, with optional PD damping |

Existing simple baselines remain available. The variants preserve stock geometry,
object friction, rewards, action limits and reset distributions. There are no
extra obstacles or gates in the new friction variants. Reacher and InvertedPendulum
now keep nominal masses; the existing InvertedPendulum initial-state perturbations
remain shared by the base and Pulse. The payload changes both mass and rotational
inertia at construction, never cumulatively at reset. Its mass is relative to
the distal arm link, not to the small fingertip body.

PandaReach and MovingSlow share a 150-step maximum horizon (6 s). Static Reach
still ends on native success. MovingSlow runs to the horizon even after reaching
the moving goal: after a 10-step warmup, at least 60% of samples must be within
5 cm, and the final sample must also be within 5 cm. The older
`LawevoPandaReachMoving-v0` also no longer terminates on a transient goal contact.
These protocol and baseline changes require fresh runs; do not resume old cached
runs or directly pool their scores with this suite.

The manipulation baselines deliberately expose their hand-designed logic:

- `waypoint_push` aligns the hand 7 cm behind the cube along the goal direction,
  pushes through the rear face, and realigns after lateral contact geometry is lost.
- `slide_align`, `slide_strike` and `slide_retract` share an episode-local state
  machine. It aligns 6 cm behind the puck, strikes for 0.24 s along a latched goal
  direction with distance-scaled amplitude, then retracts upward. `slide_damping`
  is active during alignment and retraction only.
- CEM tunes the baseline expression gains. Waypoint offsets, phase thresholds and
  strike duration are fixed helper constants, not searched parameters. Evolved
  expressions receive these same helper signals. Their internal state-machine
  complexity is not included in symbolic tree node counts.
- LQR uses the existing nominal MuJoCo linearization and is evaluated on both
  InvertedPendulum environments. It is an analytic reference, not a CEM-tuned law.
  All other listed baselines use the common CEM tuning path.

Run one new environment through the usual pipeline (uses the configured LLM endpoint):

```powershell
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaSlideFrictionShift-v0
```

Run all ten in PowerShell:

```powershell
$taskEnvironments = @(
    'InvertedPendulum-v5', 'LawevoInvertedPendulumPulse-v0',
    'Reacher-v5', 'LawevoReacherPayload-v0',
    'PandaReachDense-v3', 'LawevoPandaReachMovingSlow-v0',
    'PandaPushDense-v3', 'LawevoPandaPushLowFriction-v0',
    'PandaSlideDense-v3', 'LawevoPandaSlideFrictionShift-v0'
)
foreach ($taskEnvironment in $taskEnvironments) {
    py -m experiments.gymnasium_classical_benchmarks --environment $taskEnvironment
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $taskEnvironment" }
}
```

For programmatic use, adapters are exported as `CONTROLLED_VARIANT_ADAPTERS`
from `lawevo.pid`. Calling an adapter's `make_env()` registers the custom Gymnasium
IDs lazily. The implementation is in `lawevo/pid/controlled_variants.py`.

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
