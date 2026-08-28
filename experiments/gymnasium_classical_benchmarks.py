from __future__ import annotations

import argparse
import csv
import getpass
import itertools
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from lawevo.evolve.nvidia_nim import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    NVIDIAChatClient,
    load_env_file,
)
from lawevo.pid import (
    ADAPTERS,
    LOCOMOTION_ADAPTERS,
    GymMetrics,
    GymStructure,
    evaluate_gym_structure,
    inverted_pendulum_lqr,
    tune_gym_cem,
)

ENVIRONMENT_DESCRIPTIONS = {
    "pendulum": """Task: swing a torque-limited pendulum up and keep it upright for a
200-step episode. Observation is [cos(theta), sin(theta), theta_dot], but the controller
features use wrapped upright angle theta in [-pi, pi]. The scalar action is torque clipped
to [-2, 2]. Reward penalizes squared angle error most strongly, then angular velocity and
torque. A final state counts as success when |theta| < 0.2 rad and |theta_dot| < 0.75.
Training randomizes mass by +/-15% and length by +/-10%. Useful design considerations:
large-error nonlinear terms can generate swing-up torque, near-zero terms should stabilize
without chatter, velocity feedback provides damping, and integral action can remove bias
but may wind up (the implemented integral is clipped to [-4, 4]).""",
    "inverted_pendulum": """Task: balance an initially perturbed pole on a translating
cart for all 500 steps. Observation and controller state are [cart position, pole angle,
cart velocity, pole angular velocity]; the scalar action is horizontal cart force clipped
to the MuJoCo actuator limit. The episode terminates if the pole angle or cart position
leaves the environment's safe range. Success requires surviving the full horizon with
|pole_angle| < 0.1 rad. Initial cart/pole states are broadened and both moving-body masses
are randomized by +/-15%. Pole stabilization is primary, cart centering prevents eventual
termination, velocity terms add damping, saturated tanh terms can limit aggressive force,
and clipped integrals can correct persistent offsets but may reduce transient robustness.""",
    "reacher": """Task: drive a two-link planar arm's fingertip to a randomized target in
50 steps. The action is a vector of two joint torques, clipped per actuator. Native reward
penalizes fingertip-target distance and control magnitude; success means final distance
below 0.05. Link/body masses are randomized by +/-10%. Every allowed feature is a two-vector:
jt_error is J^T times Cartesian target error; joint_velocity supplies joint damping;
integral_jt_error is clipped to [-0.5, 0.5]; tanh_jt_error is tanh(10*jt_error) and gives
strong bounded near-target feedback; normalized_jt_error has unit magnitude and can chatter
near the target; task_damping is J^T J qdot and damps Cartesian motion. Prefer structures
that approach quickly but settle smoothly instead of spending torque oscillating.""",
    "hopper": """Task: make a planar one-legged Hopper move forward without falling for
300 steps. The three actions are thigh, leg, and foot hinge torques in [-1, 1]. State
features include three joint posture errors and velocities, torso height and pitch errors,
forward-speed error, and a fixed-frequency three-phase CPG. The native reward combines
forward velocity, a healthy-standing reward, and control cost. Falling occurs when torso
height drops below 0.7 m or torso pitch leaves roughly [-0.2, 0.2] rad. Body masses are
randomized by +/-10%. Periodic terms create hopping; posture, height, and pitch feedback
must stabilize the gait, while velocity feedback can reduce impacts and chatter.""",
    "walker2d": """Task: make a planar two-legged Walker move forward without falling for
300 steps. Six actions command the thigh, leg, and foot hinges on both legs. The signal
library provides six-dimensional joint posture/velocity feedback, torso height and pitch,
forward-speed error, and an anti-phase CPG for alternating legs. Native reward combines
forward velocity, healthy survival, and control cost. Body masses are randomized by
+/-10%. Useful structures coordinate left/right stepping while damping joint motion and
correcting torso drift; excessive symmetric feedback can prevent an alternating gait.""",
    "half_cheetah": """Task: make a planar six-actuator HalfCheetah run forward for 300
steps. The environment does not normally terminate on a fall; native reward is forward
velocity minus control cost. Signals include rear/front joint posture and velocity,
body pitch and height, target forward-speed error, and an anti-phase periodic CPG. Body
masses are randomized by +/-10%. A useful structure must create a propulsive cyclic gait,
coordinate front and rear limbs, and limit high-frequency torque without merely holding a
static posture.""",
    "ant": """Task: make an eight-actuator quadruped Ant move forward for 300 steps while
remaining healthy. Actions control four hip/ankle pairs. Features are read from MuJoCo's
free-root state and include eight joint posture/velocity signals, diagonal-leg CPG phases,
torso roll/pitch and height feedback, and forward-speed error. Native reward combines
forward velocity, healthy survival, control cost, and contact cost. Body masses are
randomized by +/-10%. Useful structures coordinate diagonal legs while correcting body
attitude; large discontinuous torques can flip the torso or exploit impacts.""",
    "humanoid": """Task: make a 17-actuator Humanoid move forward for 300 steps without
falling. Actions control abdomen, hips, knees, shoulders, and elbows. Signals use the
MuJoCo free-root state, reordered into actuator order, with joint posture/velocity,
alternating leg-and-arm CPG phases, torso roll/pitch, height, and forward-speed feedback.
Native reward emphasizes forward velocity and healthy survival while penalizing control
and impact. Body masses are randomized by +/-10%. Stable whole-body coordination and
balance are more important than a fast but fragile initial lunge.""",
    "pusher": """Task: use a seven-joint arm to contact an object and push it to a target
within 100 steps. Features are seven-dimensional Jacobian-transpose signals for fingertip
to object error, object to goal error, and their combined push direction, plus joint
velocity, integral, saturation, normalization, task-space damping, and posture feedback.
Native reward penalizes object-goal distance, fingertip-object distance, and control cost.
Arm/object body masses are randomized by +/-10%. A useful structure must first establish
contact, then maintain a controlled push instead of merely reaching the object.""",
}

CONTROL_GOALS = {
    "pendulum": """Primary goal: maximize cumulative return by swinging the pendulum to
the upright equilibrium theta=0 and keeping both theta and theta_dot near zero for as much
of the episode as possible. Operational success target: finish with |theta| < 0.2 rad and
|theta_dot| < 0.75 across randomized mass, length, and initial states. Secondary goals:
avoid unnecessarily large torque, rapid torque changes, steady-state bias, and needless
terms. A useful structure must support both high-authority swing-up far from upright and
smooth, well-damped stabilization near upright.""",
    "inverted_pendulum": """Primary goal: prevent termination and balance the pole for all
500 steps, with pole_angle driven toward zero. Simultaneously keep cart_position near zero
so the cart does not drift outside its track. Operational success target: survive the full
horizon and finish with |pole_angle| < 0.1 rad across randomized initial conditions and
body masses. Among equally reliable stabilizers, prefer lower force energy, smoother force
changes, faster damping of velocities, and fewer terms. Never trade survival for a small
energy or complexity improvement.""",
    "reacher": """Primary goal: move the fingertip to the target quickly and keep it there,
maximizing cumulative return over 50 steps. Operational success target: final Cartesian
fingertip-target distance < 0.05 across randomized targets, initial states, and link masses.
Once near the target, suppress overshoot, oscillation, and torque chatter. Among structures
with comparable target accuracy and return, prefer lower torque energy, smoother torque
changes, robust damping, and fewer terms. Do not obtain a brief fast approach at the cost
of failing to settle at the target.""",
    "hopper": """Primary goal: travel forward quickly while remaining healthy for the full
300-step horizon. Operational success requires surviving without a fall and ending above
0.75 m/s forward velocity. Preserve torso height near 1.25 m and pitch near zero while
coordinating a repeatable hopping cycle. Among equally successful gaits, reduce squared
torque energy and torque-rate jerk; do not gain speed through violent impacts that make the
controller fragile to the +/-10% mass variations.""",
    "walker2d": """Primary goal: walk forward quickly and survive all 300 steps. Operational
success requires no unhealthy termination and final forward velocity above 0.75 m/s.
Maintain torso height near 1.25 m and pitch near zero while producing an alternating-leg
gait. Prefer structures that achieve speed with lower torque energy and smoother commands,
and reject low-energy solutions that simply stand still or fall early.""",
    "half_cheetah": """Primary goal: maximize forward-running return over 300 steps and end
above 1.5 m/s. Develop a stable cyclic gait rather than a static posture or a single initial
kick. Secondary goals are lower squared torque energy, lower torque-rate jerk, and fewer
signals, but they must not materially reduce forward return. Robustness to +/-10% body-mass
variation is part of the goal.""",
    "ant": """Primary goal: move forward quickly, stay healthy for all 300 steps, and end
above 0.75 m/s. Maintain torso height near 0.65 m and control roll/pitch using coordinated
diagonal-leg cycles. Among equally successful gaits, minimize squared torque energy and
torque-rate jerk. Do not trade robustness for impact-heavy hopping that fails under the
+/-10% body-mass variations.""",
    "humanoid": """Primary goal: remain upright for all 300 steps and move forward, ending
above 0.5 m/s. Keep torso height near 1.4 m, limit roll/pitch, and coordinate legs, arms,
and abdomen into a repeatable gait. Survival and forward return dominate; among comparable
gaits prefer lower actuator energy, smoother commands, and fewer signals. Reject solutions
that obtain return from one unstable launch and then fall.""",
    "pusher": """Primary goal: bring the object to within 0.1 m of the goal by the end of
100 steps. Approach the object quickly, establish contact, and continue applying force in
the goal direction without losing contact. Among controllers with comparable final
object-goal accuracy and return, minimize squared torque energy and torque-rate jerk and
prefer fewer signals. Low-energy behavior that never contacts or moves the object is not
successful.""",
}


EOH_OPERATOR_GUIDANCE = (
    (
        "E1",
        (
            "Exploration crossover: inspect at least two structurally different elite "
            "parents, then create a controller with a clearly different form rather than "
            "copying their union or making a cosmetic edit."
        ),
    ),
    (
        "E2",
        (
            "Backbone crossover: identify the common control mechanism shared by at least "
            "two elite parents, preserve that useful backbone, and recombine it with "
            "complementary signals that address a measured weakness."
        ),
    ),
    (
        "M1",
        (
            "Structural mutation: choose one elite parent and make a meaningful small "
            "mutation by adding, removing, or replacing one to three terms."
        ),
    ),
    (
        "M2",
        (
            "Goal-directed mutation: choose one elite parent, identify its most important "
            "return, success, energy, or jerk failure, and minimally change its terms to "
            "target that failure. Never mutate numeric gains because CEM tunes them "
            "separately."
        ),
    ),
    (
        "M3",
        (
            "Generalization mutation: choose one elite parent and prune redundant, fragile, "
            "or over-specialized terms while retaining the mechanism needed for robustness "
            "under randomized initial states and physical parameters."
        ),
    ),
)


def eoh_operator_plan(count: int) -> list[dict[str, object]]:
    """Assign EoH-inspired exploration and modification operators to proposal slots."""
    return [
        {
            "slot": index + 1,
            "operator": EOH_OPERATOR_GUIDANCE[index % len(EOH_OPERATOR_GUIDANCE)][0],
            "instruction": EOH_OPERATOR_GUIDANCE[index % len(EOH_OPERATOR_GUIDANCE)][1],
        }
        for index in range(count)
    ]


def efficiency_goal(elites: list[dict]) -> str:
    """Create quantitative energy/jerk targets from the strongest-success cohort."""
    if not elites:
        return "No measured reference is available yet; minimize both energy and jerk."
    max_success = max(float(item["metrics"]["success_rate"]) for item in elites)
    reliable = [
        item
        for item in elites
        if float(item["metrics"]["success_rate"]) >= max_success - 1e-12
    ]
    energy_target = min(float(item["metrics"]["energy"]) for item in reliable)
    jerk_target = min(float(item["metrics"]["jerk"]) for item in reliable)
    return f"""Energy goal: minimize E = sum(dt * ||u_t||^2); lower is better. Current
reference among the highest-success structures is E <= {energy_target:.6g}. Jerk goal:
minimize J = sum(dt * ||(u_t-u_(t-1))/dt||^2); lower means smoother commands and is better.
Current reference among the highest-success structures is J <= {jerk_target:.6g}. Treat
these as improvement targets, not hard constraints: never reduce success or materially
damage return merely to meet them. Seek Pareto improvements; if one structure cannot
improve everything, propose distinct performance-, energy-, jerk-, and balanced variants."""


def fallback_structures(
    allowed: tuple[str, ...],
    elites: list[dict],
    excluded: set[tuple[str, ...]],
    count: int,
) -> list[GymStructure]:
    """Deterministic local mutations when the remote generator returns no content."""
    elite_keys = [tuple(item["structure"]["terms"]) for item in elites]
    candidates = [
        terms
        for size in range(1, min(8, len(allowed)) + 1)
        for terms in itertools.combinations(allowed, size)
        if terms not in excluded
    ]

    def rank(terms: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
        distance = min(
            len(set(terms).symmetric_difference(elite)) for elite in elite_keys
        )
        return distance, len(terms), terms

    candidates.sort(key=rank)
    return [
        GymStructure(f"fallback_mutation_{index + 1}", terms)
        for index, terms in enumerate(candidates[:count])
    ]


def extract_structures(response: str, allowed: tuple[str, ...]) -> list[GymStructure]:
    text = re.sub(r"```(?:json)?|```", "", response, flags=re.IGNORECASE).strip()
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return []
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("structures", [payload])
    output: list[GymStructure] = []
    if not isinstance(payload, list):
        return output
    for index, item in enumerate(payload):
        try:
            structure = GymStructure(
                str(item.get("name", f"proposal_{index}"))[:50], tuple(item["terms"])
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if set(structure.terms) <= set(allowed) and structure.key() not in {
            existing.key() for existing in output
        }:
            output.append(structure)
    return output


def prompt(
    env_name: str,
    allowed: tuple[str, ...],
    elites: list[dict],
    archive: list[dict],
    count: int,
    generation: int,
    energy_weight: float,
    jerk_weight: float,
) -> str:
    return f"""Generation {generation}: evolve {count} compact feedback-controller structures
specifically for {env_name}. Treat the environment description below as authoritative.

Environment and task:
{ENVIRONMENT_DESCRIPTIONS[env_name]}

Control goal (what the evolved controller must accomplish):
{CONTROL_GOALS[env_name]}

Control-effort and smoothness goals:
{efficiency_goal(elites)}

The structure is a weighted sum of selected basis signals. Do NOT propose gains: every K is
tuned later by equal-budget Cross-Entropy Method. One gain multiplies each selected term;
vector-valued terms are combined componentwise.

Allowed terms: {json.dumps(allowed)}
Best current classical and evolved structures after CEM tuning:
{json.dumps(elites, indent=2)}

EoH-inspired variation plan:
{json.dumps(eoh_operator_plan(count), indent=2)}

Apply the operator assigned to each output slot. E1 and E2 must use multiple named elite
parents when at least two are available; M1, M2, and M3 must each use one named elite
parent. The evolvable genome is only the selected terms list, never the numeric gains.
Prefix every candidate name with its operator code (for example E2_backbone_damping) so
the variation provenance remains visible in generation logs.

Exact scalar fitness is environment return - {energy_weight}*energy
- {jerk_weight}*jerk - 0.02*term_count. Success is reported as a diagnostic, while the
environment return is the dominant optimized quantity. Infer failure modes and trade-offs
from the tuned metrics. Mutate/crossover strong structures, but also test task-motivated
alternatives when the elite has poor success, excessive energy, or excessive jerk.

Previously evaluated structures (never repeat an identical terms list):
{json.dumps(archive, indent=2)}

Return {count} genuinely novel and diverse structures. Use 1-8 unique allowed terms and
avoid redundant signals unless their different saturation/scaling has a clear purpose.
Before choosing each structure, reason internally about how its terms serve the control
goal and address an observed elite failure mode. When count >= 4, include at least one
performance-focused, energy-focused, jerk-focused, and balanced Pareto proposal; reflect
that role in each proposal's name. Do not include the internal reasoning in output.
Return ONLY a JSON array of exactly {count} objects with keys name and terms.
"""


def evaluate_test(adapter, records: list[dict], seeds: list[int]) -> dict[str, dict]:
    output = {}
    for record in records:
        metrics, episodes = evaluate_gym_structure(
            adapter, record["structure"], np.asarray(record["gains"]), seeds
        )
        output[record["label"]] = {
            "metrics": metrics.to_dict(),
            "episode_returns": [episode.episode_return for episode in episodes],
            "success": [float(episode.success) for episode in episodes],
            "energy": [episode.energy for episode in episodes],
            "jerk": [episode.jerk for episode in episodes],
        }
    return output


def write_metric_logs(all_results: dict[str, object], output: Path) -> None:
    """Write aggregate and per-rollout comparison logs in analysis-friendly CSV files."""
    summary_fields = (
        "environment",
        "controller",
        "controller_kind",
        "return",
        "success_rate",
        "energy",
        "jerk",
        "score",
        "complexity",
    )
    rollout_fields = (
        "environment",
        "controller",
        "controller_kind",
        "rollout",
        "seed",
        "return",
        "success",
        "energy",
        "jerk",
    )
    with (output / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in all_results.values():
            for controller, values in result["test"].items():
                metrics = values["metrics"]
                writer.writerow(
                    {
                        "environment": result["environment"],
                        "controller": controller,
                        "controller_kind": (
                            "evolved" if controller == "Evolved Structure" else "classical"
                        ),
                        "return": metrics["episode_return"],
                        "success_rate": metrics["success_rate"],
                        "energy": metrics["energy"],
                        "jerk": metrics["jerk"],
                        "score": metrics["score"],
                        "complexity": metrics["complexity"],
                    }
                )
    with (output / "rollout_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rollout_fields)
        writer.writeheader()
        for result in all_results.values():
            for controller, values in result["test"].items():
                for index, seed in enumerate(result["test_seeds"]):
                    writer.writerow(
                        {
                            "environment": result["environment"],
                            "controller": controller,
                            "controller_kind": (
                                "evolved"
                                if controller == "Evolved Structure"
                                else "classical"
                            ),
                            "rollout": index + 1,
                            "seed": seed,
                            "return": values["episode_returns"][index],
                            "success": values["success"][index],
                            "energy": values["energy"][index],
                            "jerk": values["jerk"][index],
                        }
                    )


def plot_environment(env_name: str, test_results: dict[str, dict], output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    panels = (
        ("episode_returns", "Episode Return ↑", "(a)"),
        ("success", "Success Rate (%) ↑", "(b)"),
        ("energy", "Control Energy ↓", "(c)"),
        ("jerk", "Control Jerk ↓", "(d)"),
    )
    labels = list(test_results)
    colors = ["#BDBDBD", "#999999", "#777777", "#555555", "#009E73"][: len(labels)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for axis, (metric, title, panel) in zip(axes.ravel(), panels):
        arrays = [np.asarray(test_results[label][metric]) for label in labels]
        if metric == "success":
            arrays = [100 * values for values in arrays]
        means = [float(np.mean(values)) for values in arrays]
        sems = [float(np.std(values, ddof=1) / np.sqrt(len(values))) for values in arrays]
        x = np.arange(len(labels))
        axis.bar(x, means, yerr=sems, color=colors, capsize=3, edgecolor="white")
        axis.set_title(f"{panel} {title}", loc="left", fontweight="bold")
        axis.set_xticks(x, labels, rotation=17, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle(
        f"{env_name}: Classical Controllers vs. Evolved Structure",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(0.5, 0.01, "Mean ± SEM over held-out initializations.", ha="center")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.15, hspace=0.35, wspace=0.22)
    stem = env_name.lower().replace("-", "_")
    fig.savefig(output / f"{stem}_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    load_env_file()
    available_adapters = {
        adapter.env_id: (name, adapter)
        for name, adapter in {**ADAPTERS, **LOCOMOTION_ADAPTERS}.items()
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        choices=tuple(sorted(available_adapters)),
        required=True,
        help="run one supported Gymnasium environment",
    )
    parser.add_argument("--model", default=os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL))
    parser.add_argument(
        "--base-url", default=os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)
    )
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--proposals", type=int, default=6)
    parser.add_argument("--proposal-attempts", type=int, default=3)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--cem-population", type=int, default=24)
    parser.add_argument("--train-episodes", type=int, default=6)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument(
        "--resume-run",
        help="resume a timestamped run directory under results, for example 20260827_231500",
    )
    args = parser.parse_args()
    started_at = datetime.now().astimezone()
    run_id = args.resume_run or started_at.strftime("%Y%m%d_%H%M%S")
    run_root = Path("results") / run_id
    state_dir = run_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    manifest = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "status": "running",
        "requested_environment": args.environment,
    }
    if args.resume_run and manifest_path.exists():
        manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest["status"] = "running"
        manifest["resumed_at"] = started_at.isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    api_key = os.environ.get("NVIDIA_API_KEY") or getpass.getpass("NVIDIA API key: ")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is required")
    client = NVIDIAChatClient(api_key, model=args.model, endpoint=args.base_url)
    all_results: dict[str, object] = {}
    responses_path = state_dir / "nim_responses.json"
    cache_path = state_dir / "evaluation_cache.json"
    plans_path = state_dir / "generation_plans.json"
    raw_responses: list[dict] = (
        json.loads(responses_path.read_text(encoding="utf-8"))
        if responses_path.exists()
        else []
    )
    evaluation_cache: dict[str, list[dict]] = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    generation_plans: dict[str, dict[str, list[dict]]] = (
        json.loads(plans_path.read_text(encoding="utf-8"))
        if plans_path.exists()
        else {}
    )

    env_name, adapter = available_adapters[args.environment]
    adapters = {env_name: adapter}
    manifest["environment_folders"] = [adapter.env_id for adapter in adapters.values()]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for env_index, (env_name, adapter) in enumerate(adapters.items()):
        environment_output = run_root / adapter.env_id
        for child in ("classical", "lawevo", "plot", "summary"):
            (environment_output / child).mkdir(parents=True, exist_ok=True)
        train_seeds = [
            10_000 * (env_index + 1) + index for index in range(args.train_episodes)
        ]
        test_seeds = [
            90_000 + 1_000 * env_index + index for index in range(args.test_episodes)
        ]
        evaluated: dict[tuple[str, ...], dict] = {}
        cached_by_generation: dict[int, list[dict]] = {}
        for record in evaluation_cache.get(env_name, []):
            cached_by_generation.setdefault(int(record["generation"]), []).append(record)
        env_plans = generation_plans.setdefault(env_name, {})
        for record in raw_responses:
            if record.get("environment") != env_name or not record.get("response"):
                continue
            generation_key = str(record["generation"])
            recovered = env_plans.setdefault(generation_key, [])
            recovered_keys = {tuple(item["terms"]) for item in recovered}
            for structure in extract_structures(record["response"], adapter.allowed_terms):
                if structure.key() not in recovered_keys and len(recovered) < args.proposals:
                    recovered.append(structure.to_dict())
                    recovered_keys.add(structure.key())
        current = list(adapter.classical)
        for generation in range(args.generations + 1):
            if generation == 0:
                current = list(adapter.classical)
            elif str(generation) in env_plans:
                current = [
                    GymStructure(record["name"], tuple(record["terms"]))
                    for record in env_plans[str(generation)]
                ]
            elif generation in cached_by_generation:
                current = [
                    GymStructure(record["structure"]["name"], tuple(record["structure"]["terms"]))
                    for record in cached_by_generation[generation]
                ]
            for structure in current:
                if structure.key() in evaluated:
                    continue
                cached = next(
                    (
                        record
                        for record in cached_by_generation.get(generation, [])
                        if tuple(record["structure"]["terms"]) == structure.key()
                    ),
                    None,
                )
                if cached:
                    gains = np.asarray(cached["gains"], dtype=float)
                    metrics = GymMetrics(**cached["metrics"])
                else:
                    gains, metrics = tune_gym_cem(
                        adapter,
                        structure,
                        train_seeds,
                        iterations=args.cem_iterations,
                        population_size=args.cem_population,
                    )
                evaluated[structure.key()] = {
                    "structure": structure,
                    "gains": gains,
                    "metrics": metrics,
                    "generation": generation,
                }
                if not cached:
                    evaluation_cache.setdefault(env_name, []).append(
                        {
                            "structure": structure.to_dict(),
                            "gains": gains.tolist(),
                            "metrics": metrics.to_dict(),
                            "generation": generation,
                        }
                    )
                    cache_path.write_text(
                        json.dumps(evaluation_cache, indent=2), encoding="utf-8"
                    )
                print(
                    f"env={env_name} gen={generation} structure={structure.name!r} "
                    f"score={metrics.score:.4f}",
                    flush=True,
                )
            ranked = sorted(
                evaluated.values(), key=lambda item: item["metrics"].score, reverse=True
            )
            generation_output = environment_output / "lawevo" / "generations"
            generation_output.mkdir(parents=True, exist_ok=True)
            (generation_output / f"generation_{generation:03d}.json").write_text(
                json.dumps(
                    {
                        "environment": adapter.env_id,
                        "generation": generation,
                        "best_so_far": {
                            "structure": ranked[0]["structure"].to_dict(),
                            "gains": ranked[0]["gains"].tolist(),
                            "metrics": ranked[0]["metrics"].to_dict(),
                        },
                        "evaluated_this_generation": [
                            {
                                "structure": item["structure"].to_dict(),
                                "gains": item["gains"].tolist(),
                                "metrics": item["metrics"].to_dict(),
                            }
                            for item in ranked
                            if item["generation"] == generation
                        ],
                        "ranking_so_far": [
                            {
                                "structure": item["structure"].to_dict(),
                                "gains": item["gains"].tolist(),
                                "metrics": item["metrics"].to_dict(),
                                "generation": item["generation"],
                            }
                            for item in ranked
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if generation == args.generations:
                break
            if str(generation + 1) in env_plans:
                current = [
                    GymStructure(record["name"], tuple(record["terms"]))
                    for record in env_plans[str(generation + 1)]
                ]
                continue
            if generation + 1 in cached_by_generation:
                current = [
                    GymStructure(record["structure"]["name"], tuple(record["structure"]["terms"]))
                    for record in cached_by_generation[generation + 1]
                ]
                continue
            elites = [
                {"structure": item["structure"].to_dict(), "metrics": item["metrics"].to_dict()}
                for item in ranked[:6]
            ]
            proposals: list[GymStructure] = []
            for attempt in range(1, args.proposal_attempts + 1):
                archive = [item["structure"].to_dict() for item in ranked] + [
                    proposal.to_dict() for proposal in proposals
                ]
                response = client.complete(
                    "You are a control researcher evolving interpretable feedback structures "
                    "for one explicitly described environment. Return strict JSON only.",
                    prompt(
                        env_name,
                        adapter.allowed_terms,
                        elites,
                        archive,
                        args.proposals - len(proposals),
                        generation + 1,
                        adapter.energy_weight,
                        adapter.jerk_weight,
                    ),
                    temperature=0.8,
                    reasoning_effort="high",
                )
                known_keys = set(evaluated) | {proposal.key() for proposal in proposals}
                fresh = [
                    proposal
                    for proposal in extract_structures(response, adapter.allowed_terms)
                    if proposal.key() not in known_keys
                ]
                proposals.extend(fresh[: args.proposals - len(proposals)])
                raw_responses.append(
                    {
                        "environment": env_name,
                        "generation": generation + 1,
                        "attempt": attempt,
                        "valid_new": len(fresh),
                        "response": response,
                    }
                )
                responses_path.write_text(
                    json.dumps(raw_responses, indent=2), encoding="utf-8"
                )
                if len(proposals) == args.proposals:
                    break
                time.sleep(5 * attempt)
            if not proposals:
                proposals = fallback_structures(
                    adapter.allowed_terms,
                    elites,
                    set(evaluated),
                    args.proposals,
                )
                raw_responses.append(
                    {
                        "environment": env_name,
                        "generation": generation + 1,
                        "fallback": True,
                        "valid_new": len(proposals),
                    }
                )
                responses_path.write_text(
                    json.dumps(raw_responses, indent=2), encoding="utf-8"
                )
            if not proposals:
                raise RuntimeError(f"structure space exhausted for {env_name}")
            current = proposals[: args.proposals]
            env_plans[str(generation + 1)] = [
                proposal.to_dict() for proposal in current
            ]
            plans_path.write_text(
                json.dumps(generation_plans, indent=2), encoding="utf-8"
            )

        ranked = sorted(evaluated.values(), key=lambda item: item["metrics"].score, reverse=True)
        classical_keys = {structure.key() for structure in adapter.classical}
        classical = [item for item in ranked if item["structure"].key() in classical_keys]
        evolved = [item for item in ranked if item["structure"].key() not in classical_keys]
        best_evolved = evolved[0]
        comparison_records = [
            {
                "label": item["structure"].name,
                "structure": item["structure"],
                "gains": item["gains"],
            }
            for item in classical
        ]
        if env_name == "inverted_pendulum":
            lqr_structure, lqr_gains = inverted_pendulum_lqr()
            comparison_records.append(
                {"label": "LQR", "structure": lqr_structure, "gains": lqr_gains}
            )
        comparison_records.append(
            {
                "label": "Evolved Structure",
                "structure": best_evolved["structure"],
                "gains": best_evolved["gains"],
            }
        )
        test_results = evaluate_test(adapter, comparison_records, test_seeds)
        plot_environment(adapter.env_id, test_results, environment_output / "plot")
        all_results[env_name] = {
            "environment": adapter.env_id,
            "train_seeds": train_seeds,
            "test_seeds": test_seeds,
            "best_evolved": {
                "structure": best_evolved["structure"].to_dict(),
                "gains": best_evolved["gains"].tolist(),
                "train_metrics": best_evolved["metrics"].to_dict(),
                "test": test_results["Evolved Structure"],
            },
            "classical_controllers": [
                {
                    "label": record["label"],
                    "structure": record["structure"].to_dict(),
                    "gains": np.asarray(record["gains"]).tolist(),
                    "test": test_results[record["label"]],
                }
                for record in comparison_records
                if record["label"] != "Evolved Structure"
            ],
            "test": test_results,
            "all_structures": [
                {
                    "structure": item["structure"].to_dict(),
                    "gains": item["gains"].tolist(),
                    "train_metrics": item["metrics"].to_dict(),
                    "generation": item["generation"],
                }
                for item in ranked
            ],
        }
    protocol = {
        "generations": args.generations,
        "proposals": args.proposals,
        "proposal_attempts": args.proposal_attempts,
        "cem_iterations": args.cem_iterations,
        "cem_population": args.cem_population,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "requested_environment": args.environment,
        "prompt_context": "task-specific environment description and control goal",
    }
    for env_name, result in all_results.items():
        environment_output = run_root / result["environment"]
        (environment_output / "classical" / "controllers.json").write_text(
            json.dumps(result["classical_controllers"], indent=2), encoding="utf-8"
        )
        (environment_output / "lawevo" / "best_controller.json").write_text(
            json.dumps(result["best_evolved"], indent=2), encoding="utf-8"
        )
        (environment_output / "lawevo" / "generation_plans.json").write_text(
            json.dumps(generation_plans.get(env_name, {}), indent=2), encoding="utf-8"
        )
        (environment_output / "lawevo" / "nim_responses.json").write_text(
            json.dumps(
                [item for item in raw_responses if item.get("environment") == env_name],
                indent=2,
            ),
            encoding="utf-8",
        )
        (environment_output / "summary" / "results.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "protocol": protocol,
                    "result": result,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_metric_logs({env_name: result}, environment_output / "summary")
    responses_path.write_text(json.dumps(raw_responses, indent=2), encoding="utf-8")
    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nHeld-out controller comparison:", flush=True)
    for result in all_results.values():
        for controller, values in result["test"].items():
            metrics = values["metrics"]
            print(
                f"env={result['environment']} controller={controller!r} "
                f"return={metrics['episode_return']:.6g} "
                f"success_rate={metrics['success_rate']:.3f} "
                f"energy={metrics['energy']:.6g} jerk={metrics['jerk']:.6g}",
                flush=True,
            )
    print(
        json.dumps({name: result["best_evolved"] for name, result in all_results.items()}, indent=2)
    )
    print(f"Results saved under {run_root}", flush=True)


if __name__ == "__main__":
    main()
