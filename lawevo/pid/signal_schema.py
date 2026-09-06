"""Signal contracts and dimensional inference for the controlled ten-task suite.

Radians are dimensionless in SI; angle remains a distinct semantic category.
Gain units are inferred (not assumed dimensionless). Legacy tasks explicitly
report incomplete contracts and retain name-only validation.
"""

from dataclasses import asdict, dataclass

import numpy as np

from lawevo.pid.expression import BinaryOp, Const, ProductOp, Scale, Signal, SumOp, UnaryOp


@dataclass(frozen=True)
class SignalSpec:
    semantic_type: str
    unit: str
    dimensions: tuple[float, float]  # exponents of metre and second
    shape: tuple[int, ...]
    formula: str
    sign: str
    zero_when: str
    value_range: str
    memory: str = "none"
    frame: str = "world"
    action_mapping: str = "componentwise to action channels"

    def to_dict(self):
        return {**asdict(self), "value_type": "scalar" if not self.shape else "vector"}


@dataclass(frozen=True)
class SignalContract:
    signals: dict[str, SignalSpec]
    action_shape: tuple[int, ...]
    action_description: str
    complete: bool = True

    def to_dict(self):
        return {
            "schema_version": 1, "complete": self.complete,
            "action_shape": self.action_shape, "action": self.action_description,
            "gain_units": "inferred jointly; a repeated gain must have one consistent unit",
            "signals": {name: spec.to_dict() for name, spec in self.signals.items()},
        }

    def validate_values(self, values):
        for name, spec in self.signals.items():
            value = np.asarray(values[name])
            if value.shape != spec.shape or not np.isfinite(value).all():
                raise ValueError(f"signal {name}: expected finite shape {spec.shape}, got {value.shape}")


def contract_for(env_id, allowed):
    specs = {}
    if env_id in ("InvertedPendulum-v5", "LawevoInvertedPendulumPulse-v0"):
        shape, mapping = (1,), "single cart actuator command"
        rows = {
            "cart_position": ("position", "m", (1, 0), "x", "cart at origin"),
            "pole_angle": ("angle", "rad", (0, 0), "theta", "pole upright"),
            "cart_velocity": ("velocity", "m/s", (1, -1), "dx/dt", "cart stationary"),
            "pole_velocity": ("angular_velocity", "rad/s", (0, -1), "dtheta/dt", "pole stationary"),
            "integral_cart": ("integral_position", "m*s", (1, 1), "clip(integral(x dt), -2, 2)", "integral is zero"),
            "integral_angle": ("integral_angle", "rad*s", (0, 1), "clip(integral(theta dt), -1, 1)", "integral is zero"),
        }
        for name, (semantic, unit, dims, formula, zero) in rows.items():
            specs[name] = SignalSpec(semantic, unit, dims, shape, formula,
                                    "raw MuJoCo coordinate sign; not negated error", zero,
                                    "clipped as formula" if name.startswith("integral") else "no adapter clipping",
                                    "integral reset each episode" if name.startswith("integral") else "none",
                                    "MuJoCo generalized coordinates", mapping)
        for name, formula in {
            "tanh_cart": "tanh(x / 1 m)", "tanh_angle": "tanh(5*theta / 1 rad)",
            "tanh_cart_velocity": "tanh(xdot / (1 m/s))",
            "tanh_pole_velocity": "tanh(thetadot / (1 rad/s))",
        }.items():
            specs[name] = SignalSpec("normalized_feedback", "1", (0, 0), shape, formula,
                                    "same sign as raw coordinate", "input is zero", "[-1,1]",
                                    action_mapping=mapping)
    elif env_id in ("Reacher-v5", "LawevoReacherPayload-v0"):
        shape, mapping = (2,), "actuated shoulder and elbow order; normalized motor commands"
        rows = {
            "jt_error": ("projected_position_error", "m^2/rad", (2, 0), "J_xy.T @ (target_xy - fingertip_xy)", "at goal or in Jacobian-transpose nullspace", "none", "no adapter clipping"),
            "joint_velocity": ("angular_velocity", "rad/s", (0, -1), "qdot[actuated_dofs]", "joints stationary", "none", "no adapter clipping"),
            "task_damping": ("projected_velocity", "m^2/(rad*s)", (2, -1), "J_xy.T @ (J_xy @ qdot)", "projected task velocity is zero", "none", "no adapter clipping"),
            "integral_jt_error": ("integral_projected_error", "m^2*s/rad", (2, 1), "clip(integral(jt_error dt), -0.5, 0.5)", "integral is zero", "integrator, cleared at reset", "[-0.5,0.5]"),
            "tanh_jt_error": ("normalized_error", "1", (0, 0), "tanh(jt_error / (0.1 m^2/rad))", "jt_error is zero", "none", "[-1,1]"),
            "tanh_velocity": ("normalized_velocity", "1", (0, 0), "tanh(qdot / (1 rad/s))", "joints stationary", "none", "[-1,1]"),
            "normalized_jt_error": ("direction", "1", (0, 0), "jt_error/max(norm(jt_error),1e-6 m^2/rad)", "jt_error is zero", "none", "norm <= 1"),
        }
        for name, (semantic, unit, dims, formula, zero, memory, bounds) in rows.items():
            specs[name] = SignalSpec(semantic, unit, dims, shape, formula,
                                    "error is target minus current; velocities are positive, not negated",
                                    zero, bounds, memory, "joint order, world xy Jacobian", mapping)
    elif env_id in (
        "PandaReachDense-v3", "LawevoPandaReachMovingSlow-v0", "LawevoPandaReachMovingSlow-v1", "LawevoPandaReachMoving-v0",
        "PandaPushDense-v3", "LawevoPandaPushLowFriction-v0",
        "PandaSlideDense-v3", "LawevoPandaSlideFrictionShift-v0",
    ):
        shape, mapping = (3,), "world xyz normalized end-effector displacement; blocked gripper"
        def add(name, semantic, unit, dims, formula, zero, bounds="no adapter clipping", memory="none"):
            specs[name] = SignalSpec(semantic, unit, dims, shape, formula,
                                    "positive components command positive world axes after positive gain",
                                    zero, bounds, memory, action_mapping=mapping)
        add("goal_error", "position_error", "m", (1, 0), "goal - eef", "eef at goal")
        add("integral_goal_error", "integral_error", "m*s", (1, 1),
            "clip(integral((goal-eef) dt), -0.25, 0.25)", "integral zero", "[-0.25,0.25]",
            "integrator reset each episode")
        add("eef_damping", "velocity", "m/s", (1, -1), "-eef_linear_velocity",
            "eef stationary")
        add("goal_velocity", "velocity", "m/s", (1, -1), "(goal - previous_goal)/dt",
            "first feature call or stationary goal", memory="previous goal, reset each episode")
        add("tanh_goal_error", "normalized_error", "1", (0, 0), "tanh((goal-eef)/(0.1 m))", "at goal", "[-1,1]")
        for name, raw in (("normalized_goal_error", "goal-eef"),
                          ("normalized_reach_object", "object-eef"),
                          ("normalized_object_goal_error", "goal-object")):
            add(name, "direction", "1", (0, 0), f"({raw})/max(norm({raw}),1e-6 m)", "numerator zero", "norm <= 1")
        for name, fn in (("phase_sin", "sin"), ("phase_cos", "cos")):
            add(name, "phase", "1", (0, 0), f"[{fn}(t/(1 s)),0,0]", f"{fn} phase zero; y,z always zero",
                "[-1,1]", "clock advanced once per feature call, reset each episode")
        add("reach_object", "position_error", "m", (1, 0), "object - eef", "eef at object")
        add("object_goal_error", "position_error", "m", (1, 0), "goal - object", "object at goal")
        add("contact_then_goal", "position_error", "m", (1, 0),
            "goal-object if norm(object-eef)<0.08 m else object-eef", "selected error zero",
            memory="none; proximity gate is hand-designed, not contact sensing")
        add("waypoint_push", "position_error", "m", (1, 0),
            "align: object-0.07*d+[0,0,0.01]-eef; push: (object+0.02*d-eef)*min(1,norm(goal_xy-object_xy)/0.05); d=normalized planar goal direction",
            "selected error zero; push vanishes at planar goal",
            memory="FSM align->push when within 0.025 m of waypoint; realign if lateral error>0.05 m or hand >0.02 m ahead of object; reset each episode")
        add("slide_align", "position_error", "m", (1, 0),
            "object-0.06*d+[0,0,0.01]-eef during align", "outside align or at waypoint",
            memory="shared Slide FSM: align->strike within 0.025 m; latch direction, distance, retract target")
        add("slide_strike", "action_direction", "1", (0, 0),
            "latched_planar_direction*min(1,latched_goal_distance/(0.25 m)) during strike",
            "outside strike or zero latched distance", "norm <= 1",
            "shared Slide FSM; strike lasts 0.24 s then retract; geometry latched at transition")
        add("slide_retract", "position_error", "m", (1, 0),
            "latched_retract_target-eef during retract; target=eef_at_strike_start+[0,0,0.10 m]",
            "outside retract or at target", memory="shared Slide FSM; retract is terminal phase until reset")
        add("slide_damping", "velocity", "m/s", (1, -1),
            "-eef_velocity during align/retract", "during strike or eef stationary",
            memory="shared Slide FSM")
    else:
        return SignalContract({}, (), "legacy task: signal units not yet audited", complete=False)
    missing = set(allowed) - set(specs)
    if missing:
        raise ValueError(f"missing signal specifications: {sorted(missing)}")
    return SignalContract({name: specs[name] for name in allowed}, shape, mapping)


def validate_expression(law, contract):
    """Check shapes and solve gain-unit equations; reject inconsistent sharing.

    Return one consistent gain assignment plus the number of free unit variables.
    Dimensions are SI metre/second exponents. Numeric zero is unit-polymorphic.
    """
    if not contract.complete:
        return {"checked": False, "reason": "legacy task has no audited unit contract"}
    count = law.parameter_count
    equations, targets = [], []

    def equal(a, b):
        equations.append(a[0] - b[0])
        targets.append(b[1] - a[1])

    def combine_shape(a, b):
        if a and b and a != b:
            raise ValueError(f"incompatible signal shapes {a} and {b}")
        return a or b

    def walk(node):
        # (unknown gain coefficients, known SI exponents, shape, literal-zero flag)
        if isinstance(node, Signal):
            spec = contract.signals[node.name]
            return np.zeros(count), np.array(spec.dimensions, dtype=float), spec.shape, False
        if isinstance(node, Const):
            return np.zeros(count), np.zeros(2), (), node.value == 0
        if isinstance(node, Scale):
            a, d, shape, zero = walk(node.child)
            a = a.copy()
            a[node.k] += 1
            return a, d, shape, zero
        if isinstance(node, UnaryOp):
            a, d, shape, zero = child = walk(node.child)
            if node.fn in ("tanh", "sin", "cos", "exp"):
                if not zero:
                    equal(child, (np.zeros(count), np.zeros(2)))
                return np.zeros(count), np.zeros(2), shape, False
            factor = 2 if node.fn == "square" else 0.5 if node.fn == "sqrt" else 1
            return factor*a, factor*d, shape, zero
        if isinstance(node, (SumOp, ProductOp)):
            children = node.children if isinstance(node, SumOp) else node.factors
        elif isinstance(node, BinaryOp):
            children = (node.left, node.right)
        else:
            raise TypeError(type(node))
        values = [walk(child) for child in children]
        shape = ()
        for value in values:
            shape = combine_shape(shape, value[2])
        if isinstance(node, ProductOp):
            return sum(v[0] for v in values), sum(v[1] for v in values), shape, any(v[3] for v in values)
        nonzero = [v for v in values if not v[3]]
        if not nonzero:
            return np.zeros(count), np.zeros(2), shape, True
        for value in nonzero[1:]:
            equal(nonzero[0], value)
        return nonzero[0][0], nonzero[0][1], shape, False

    result = walk(law.root)
    if result[2] not in ((), contract.action_shape):
        raise ValueError("law output shape does not match action channels")
    if not result[3]:
        equal(result, (np.zeros(count), np.zeros(2)))
    if not equations:
        units, rank = np.zeros((count, 2)), 0
    else:
        matrix, target = np.vstack(equations), np.vstack(targets)
        units, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        if not np.allclose(matrix @ units, target, atol=1e-8, rtol=1e-8):
            raise ValueError("inconsistent units: scale signals before addition/min/max/nonlinearity; check tied gains")
    return {"checked": True, "gain_unit_exponents_m_s": units.tolist(),
            "free_unit_variables": count - int(rank)}
