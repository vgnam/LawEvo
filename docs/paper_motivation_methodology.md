# LawEvo — Motivation & Methodology (paper draft)

## 3. Motivation

Classical robot control rests on a small set of canonical structures — proportional,
integral, and derivative feedback, LQR, posture feedback, and central-pattern generators —
each of which commits a priori to a fixed functional form and tunes only its gains. These
controllers are interpretable, cheap, and robust, but their structural bias limits them on
tasks whose optimal policy is not obviously linear or single-loop: a pick-and-place law may
need to *gate* a transport term on a grasp signal, *switch* between reaching and lifting, or
*saturate* a correction that must vanish near the goal. Reinforcement learning can recover
such behaviors numerically, but at the cost of opaque, high-dimensional policies whose
decisions cannot be inspected, certified, or reused across morphologies.

Evolutionary program synthesis offers a middle path: search directly over symbolic control
laws so that the *structure* of the controller — which signals it reads, how it composes
them, and which nonlinearities it uses — becomes the object of optimization rather than a
hand-picked assumption. Two obstacles have kept this line of work from being adopted as a
control-design tool. First, structure and parameters are confounded: an expression is only
as good as its tuned gains, and comparing an evolved form against a hand-tuned baseline
without matching the optimization budget makes any performance claim ambiguous. Second,
generic grammar-based mutation ignores the task: it can invent expressions that no designer
would write, and it wastes evaluations on forms that are structurally redundant with the
current population.

LawEvo addresses both obstacles. It separates **structure discovery** from **parameter
optimization** through a two-layer search: a large language model (LLM), conditioned on a
task-specific description of the environment dynamics, observation and action semantics,
success conditions, observed failure modes, and the desired return–energy–jerk trade-off,
proposes compact symbolic controller structures in a free-form expression grammar; a
Cross-Entropy Method (CEM) inner loop then tunes every numeric gain — for evolved candidates
*and* for all classical baselines — under an identical simulation budget. The LLM never
chooses numbers, and the optimizer never chooses form, so measured differences are
attributable to structure alone.

This design makes a precise, falsifiable question testable:

> *Can task-aware symbolic evolution find interpretable controller structures that
> outperform equally budget-tuned P/PI/PD/PID, LQR, posture-feedback, and CPG baselines on
> return and task success — without hiding the cost in actuator effort, nonsmooth commands,
> or structural complexity?*

Beyond the head-to-head comparison, LawEvo is motivated by three properties that control
engineers — not only benchmark scores — care about. **(i) Interpretability:** every evolved
controller is a closed-form expression readable at a glance, and its size is reported as an
explicit complexity cost. **(ii) Honest cost accounting:** task return and success are
always accompanied by control energy and command jerk, so a "better" controller cannot win
by thrashing the actuators. **(iii) Process credit:** we report the success rate (SR), the
success gap (SG), and a subgoal-quality score (Q) alongside the score, distinguishing
controllers that complete the task from controllers that merely drift near it. The same
representation extends naturally to safety: the grammar also expresses control-barrier
functions that a CBF-QP filter certifies before deployment, keeping the
interpretable-symbolic philosophy from the proposal stage through verification.

## 4. Methodology

### 4.1 Problem setting

We consider discrete-time MDPs supplied by standard simulation backends (MuJoCo/Gymnasium,
PyBullet/Panda-Gym, Robosuite, ManiSkill, Genesis) with dense task rewards. At every step
$t$ the task adapter exposes a signal vector $\mathbf{s}_t \in \mathbb{R}^{d}$ (goal errors,
object poses, contact state, velocities, integral memory, and similar task features), and
the controller maps it to an action $\mathbf{u}_t \in \mathbb{R}^{m}$ that is clipped to the
environment action space. A controller is searched over *laws*: symbolic expressions over
the task signals, not over raw actions.

### 4.2 Law representation

**Definition (law).** A law is a finite expression tree $\ell$ built from the leaf set
{signal, constant, gain slot} and the operator set

$$
\texttt{+},\ \texttt{-},\ \texttt{×},\quad
f \in \{\texttt{tanh}, \texttt{sin}, \texttt{cos}, \texttt{sqrt}, \texttt{square},
\texttt{abs}, \texttt{exp}, \texttt{neg}\},\quad
g \in \{\texttt{min}, \texttt{max}\},
$$

where pairwise `min`/`max` nodes have exactly two children. A **gain slot** is written as a
token $K_i$ and denotes one scalar parameter that is *not* fixed by the genome: it indexes
position $i$ into a gain vector $\boldsymbol{\theta} \in \mathbb{R}^{p}$ optimized
numerically downstream. Signals are arrays broadcast over the action dimension (one
component per actuator), so a single law applies to every actuator of a morphology and
transfers unchanged across morphologies. Evaluation is elementwise with numerical guards
(`exp`/`square` inputs clamped, `sqrt` of negative arguments clamped, non-finite
intermediate values written as zero).

**Dual syntax.** Each law is stored in two equivalent, mutually parseable forms:
(i) a compact expression string, e.g.

```text
tanh(K1*normalized_reach_object)
  + min(K2*object_goal_error, K3*normalized_object_goal_error)
  + K4*abs(contact_then_goal)
```

and (ii) a JSON tree whose nodes are typed `{"op": signal | const | scale | sum | product |
unary | binary, ...}`. The string form is what the LLM reads and writes; the tree form is
what the archive stores and the evaluator walks. A `scale` node multiplies its child by the
scalar $\theta_{k}$ at gain slot $k$.

**Weight tying as a structural degree of freedom.** Reusing the same gain token at several
sites (`K1*x + K1*y`) binds those sites to a *single* tuned scalar. Structure that would
require a parameter-count penalty in a term-list genome is therefore expressible for free,
and the sharing pattern itself is evolvable.

**Size constraints.** Genomes are bounded to at most 16 structural (non-parameter) nodes,
depth 5, and 12 distinct gain slots. The reported **complexity** of a law is its structural
node count. A canonical, order-insensitive serialization of each law keys a persistent
archive, so structurally duplicate proposals are rejected before spending simulation
budget.

### 4.3 Two-layer optimization

**Outer layer — structural evolution.** Each generation, the LLM receives the task
description (dynamics, observation/action semantics, success conditions, available signals,
return–energy–jerk trade-off), the elite parents with their measured metrics, the latest
failure mode, and quantitative efficiency targets, and returns a fixed number of proposed
laws (6 by default). Proposals are parser-checked against the grammar, deduplicated against
the archive, and retried — with a deterministic local mutation fallback — when the endpoint
fails. The best-tuned laws become the parents of the next generation; per-generation
populations, gains, metrics, and rankings are checkpointed for resumability.

**Inner layer — equal-budget gain tuning.** Every candidate law, evolved or classical, is
tuned by CEM over its gain slots with an identical budget: $I = 5$ iterations, population
$P = 24$, scored by mean return over 6 fixed training episodes. With mean $\boldsymbol{\mu}$,
std $\boldsymbol{\sigma}$, and elite set $E_t$ (the top 20% of samples, at least 2), each
iteration samples
$\boldsymbol{\theta}_i \sim \mathrm{clip}(\mathcal{N}(\boldsymbol{\mu},
\mathrm{diag}(\boldsymbol{\sigma}^2)), -20, 20)$
and updates

$$
\boldsymbol{\mu}_{t+1} = 0.25\,\boldsymbol{\mu}_t + 0.75\,\mathrm{mean}(E_t),
\qquad
\boldsymbol{\sigma}_{t+1} = \max\!\big(0.05,\; 0.25\,\boldsymbol{\sigma}_t +
0.75\,\mathrm{std}(E_t)\big),
$$

keeping the best-scoring sample seen so far. Because every baseline receives the identical
treatment, the comparison isolates the value of structure.

**Selection objective.** The score used both by CEM and by outer-loop selection is the pure
mean episode return on the training episodes; energy, jerk, and complexity are reported but
deliberately excluded from the selection objective. The LLM is nonetheless steered toward
efficiency through *soft* targets: the best energy and jerk attained by the highest-success
cohort are injected into the prompt as Pareto improvement goals that must never be met at
the cost of success or return.

### 4.4 LLM-guided variation operators

Following the Evolution-of-Heuristics (EoH) paradigm, proposal slots are assigned to five
task-aware operators rather than blind grammar mutation:

- **E1 (exploration crossover)** — inspect two or more structurally different elites and
  produce a clearly different expression form.
- **E2 (backbone crossover)** — preserve the control mechanism shared by multiple elites
  and recombine it with complementary signals addressing a measured weakness.
- **M1 (structural mutation)** — add, remove, replace, or wrap one to three signals or
  operators of one elite parent.
- **M2 (goal-directed mutation)** — minimally edit one elite to target its most important
  observed failure (return, success, energy, or jerk); numeric gains are never mutated.
- **M3 (generalization mutation)** — prune redundant or over-specialized terms while
  retaining the mechanism needed for robustness under randomized initial states and
  physical parameters.

A **belief space** accumulates cross-run regularities (e.g., which signal idioms have
historically succeeded) and conditions subsequent prompts; a no-belief ablation switch
measures its contribution. When the remote model returns nothing usable, deterministic
local mutations of the elites (add a signal, drop a signal, fresh signal pairs) keep the
loop robust to API outages.

### 4.5 Evaluation metrics

After evolution, the best evolved law and every tuned classical baseline are evaluated on
30 held-out test episodes with randomized initial states and physical-parameter
variations. Per episode $i$ of length $T_i$, with actions
$\mathbf{u}^{(i)}_0, \ldots, \mathbf{u}^{(i)}_{T_i-1}$ and environment rewards
$r^{(i)}_t$, the engine records:

$$
R_i = \sum_{t} r^{(i)}_t, \qquad
\mathcal{E}_i = \sum_{t} \Delta t\,\lVert \mathbf{u}^{(i)}_t \rVert^2, \qquad
\mathcal{J}_i = \sum_{t} \Delta t \,\Big\lVert \tfrac{\mathbf{u}^{(i)}_t -
\mathbf{u}^{(i)}_{t-1}}{\Delta t} \Big\rVert^2 .
$$

**Success specification.** Each task adapter converts the recorded physical state into a
set of $M$ named thresholded conditions $C = \{c_1, \ldots, c_M\}$ (e.g., *object within
goal tolerance*, *final speed below a settle limit*, *grasp held through transport*).
Condition $j$ contributes a normalized violation

$$
v_j \;=\; \mathrm{clip}\!\left(\frac{\text{excess}_j}{\text{scale}_j},\, 0,\, 1\right)
\in [0, 1],
$$

where $\text{excess}_j$ is the amount by which the condition's threshold is exceeded and
$\text{scale}_j$ is a task-specific normalizer. An episode is a *success* exactly when
every violation is zero, so SR and SG share one full-success specification by construction:

$$
\boxed{\;
\mathrm{SR} \;=\; \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\Big[\textstyle\sum_{j=1}^{M}
v^{(i)}_j = 0\Big],
\qquad
\mathrm{SG} \;=\; \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M} \sum_{j=1}^{M} v^{(i)}_j .
\;}
$$

SG is the mean normalized violation of the success specification at episode end — a graded
"distance to success" that separates controllers which complete the task (SR $=1$, SG
$=0$) from controllers that merely approach it (SR $=0$, SG small).

**Process credit (Q).** Each adapter additionally defines $B$ named binary subgoal
predicates $p^{(i)}_b \in \{0,1\}$ evaluated over the episode trajectory (e.g., *cube
contacted*, *cube lifted above 2 cm*, *goal reached at any step*, *goal reached at the
final step*). The subgoal-quality score aggregates them:

$$
\boxed{\;
\mathrm{Q} \;=\; \frac{1}{N} \sum_{i=1}^{N} \frac{1}{B} \sum_{b=1}^{B} p^{(i)}_b .
\;}
$$

Q credits mid-episode progress even when the terminal specification fails, exposing
controllers that reach but do not settle, or lift but do not transport.

**Reported summary.** For every controller we report the test-set means of $R$ (Return),
SR, SG, Q, $\mathcal{E}$ (Energy), $\mathcal{J}$ (Jerk), and the law's complexity — with
Return and SR as the primary performance axes and Energy/Jerk/Complexity as explicit costs.

### 4.6 Benchmark suite and baselines

The standard benchmark runs 20 generations under the 6-train / 30-test protocol over 28
task configurations spanning four difficulty families: (i) classical low-dimensional
control (Pendulum, inverted pendula, Reacher, Pusher); (ii) locomotion (Hopper, Walker2d,
HalfCheetah, Swimmer, Ant, Humanoid, BipedalWalker); (iii) tabletop manipulation with a
Franka Panda (Panda-Gym Reach/Push/Slide/PickAndPlace/Stack with dense rewards), including
five *stress variants* that isolate distinct failure modes — a moving target
(ReachMoving), low friction (PushIce), a narrow gate (SlideGate), visual distractors
(PickDistractor), and a narrow-settle stack (StackNarrow); and (iv) cross-simulator
transfer (Robosuite, ManiSkill, and a GPU-batched Genesis path). Per environment, LawEvo
compares against 3–5 tuned classical controllers, e.g., task-space P/PI/PD/PID and LQR for
stabilization, Reach-P/PD + Object-Goal-P + Contact-Goal-PD for manipulation, Posture-P/PD
+ CPG (+PD) for locomotion, and task-specific hand-designed pipelines (Selective Pick PD,
Through-gate PD, Settle PD, Obstacle-aware PD, Tracking PD) for the stress variants. All
baselines are expressed in the same signal language and tuned by the same CEM budget as
evolved candidates.

### 4.7 Implementation

LawEvo is a dependency-light NumPy prototype; structure proposals are produced by an LLM
(`gpt-5-nano`) through an OpenAI-compatible chat-completions endpoint. Law evaluation has
both a NumPy path and a batched Torch path for GPU-batched simulators (Genesis). A strict
barrier DSL with analytic gradients and compositional Lipschitz bounds, sampled CBF
verification over a bounded state domain, and an exact low-dimensional CBF-QP filter
provide the safety extension; controllers and barriers are evolved by the same outer loop
under separate prompts.
