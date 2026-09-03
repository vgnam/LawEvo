"""Free-form symbolic controller expressions.

A law is no longer a flat weighted sum over a closed term library. It is a
mathematical expression tree over task-specific signals, written either as a
compact string (``K1*tanh(K2*jt_error) + K3*jt_error*joint_velocity``) or as a
JSON tree. Every ``K`` token marks one scalar parameter slot that the
Cross-Entropy Method tunes; the LLM only ever proposes structure. Repeating the
same ``K`` name (``K1*x + K1*y``) intentionally shares one tuned scalar.

Evaluation semantics mirror the previous weighted-sum controller so every
existing adapter keeps working: signals are numpy arrays broadcast over the
action dimension (one component per actuator), ``scale`` multiplies by one
tuned scalar, ``sum``/``product`` combine elementwise, and the final value is
clipped to the environment action space by the rollout code as before.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

MAX_NODES = 16
MAX_DEPTH = 5
MAX_PARAMS = 12

UNARY_FNS: tuple[str, ...] = ("tanh", "sin", "cos", "sqrt", "square", "abs", "exp", "neg")
BINARY_FNS: tuple[str, ...] = ("min", "max")

_SIGNAL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GAIN_NAME = re.compile(r"^[Kk][0-9]*$")

_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<num>[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
    r"|(?P<op>[()+*,\-]))"
)


@dataclass(frozen=True)
class Signal:
    name: str


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Scale:
    """Multiply the child by one CEM-tuned scalar (slot ``k`` into the gain vector)."""

    k: int
    child: Node


@dataclass(frozen=True)
class SumOp:
    children: tuple[Node, ...]


@dataclass(frozen=True)
class ProductOp:
    factors: tuple[Node, ...]


@dataclass(frozen=True)
class UnaryOp:
    fn: str
    child: Node


@dataclass(frozen=True)
class BinaryOp:
    fn: str
    left: Node
    right: Node


Node = Signal | Const | Scale | SumOp | ProductOp | UnaryOp | BinaryOp


def _make_sum(children: Sequence[Node]) -> Node:
    flat: list[Node] = []
    for child in children:
        if isinstance(child, SumOp):
            flat.extend(child.children)
        else:
            flat.append(child)
    if not flat:
        raise ValueError("a sum requires at least one child")
    if len(flat) == 1:
        return flat[0]
    return SumOp(tuple(flat))


def _make_product(factors: Sequence[Node]) -> Node:
    flat: list[Node] = []
    for factor in factors:
        if isinstance(factor, ProductOp):
            flat.extend(factor.factors)
        else:
            flat.append(factor)
    if len(flat) < 2:
        raise ValueError("a product requires at least two factors")
    return ProductOp(tuple(flat))


def _signal_name(node: Signal) -> str:
    if not _SIGNAL_NAME.match(node.name):
        raise ValueError(f"invalid signal name {node.name!r}")
    return node.name


def _depth(node: Node) -> int:
    """Structural depth; parameter wrappers (Scale) do not count."""
    if isinstance(node, Signal):
        return 1
    if isinstance(node, Const):
        return 1
    if isinstance(node, Scale):
        return _depth(node.child)
    if isinstance(node, (SumOp, ProductOp)):
        children = node.children if isinstance(node, SumOp) else node.factors
        return 1 + max(_depth(child) for child in children)
    if isinstance(node, UnaryOp):
        return 1 + _depth(node.child)
    return 1 + max(_depth(node.left), _depth(node.right))


def _count_nodes(node: Node) -> int:
    """Structural node count; parameter wrappers (Scale) do not count."""
    if isinstance(node, (Signal, Const)):
        return 1
    if isinstance(node, Scale):
        return _count_nodes(node.child)
    if isinstance(node, (SumOp, ProductOp)):
        children = node.children if isinstance(node, SumOp) else node.factors
        return 1 + sum(_count_nodes(child) for child in children)
    if isinstance(node, UnaryOp):
        return 1 + _count_nodes(node.child)
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)


def _signals(node: Node, out: list[str]) -> None:
    if isinstance(node, Signal):
        if node.name not in out:
            out.append(node.name)
    elif isinstance(node, (Const,)):
        return
    elif isinstance(node, Scale):
        _signals(node.child, out)
    elif isinstance(node, SumOp):
        for child in node.children:
            _signals(child, out)
    elif isinstance(node, ProductOp):
        for factor in node.factors:
            _signals(factor, out)
    elif isinstance(node, UnaryOp):
        _signals(node.child, out)
    else:
        _signals(node.left, out)
        _signals(node.right, out)


def _scale_nodes(node: Node) -> list[Scale]:
    if isinstance(node, Signal):
        return []
    if isinstance(node, Const):
        return []
    if isinstance(node, Scale):
        return [node, *_scale_nodes(node.child)]
    if isinstance(node, SumOp):
        return [item for child in node.children for item in _scale_nodes(child)]
    if isinstance(node, ProductOp):
        return [item for factor in node.factors for item in _scale_nodes(factor)]
    if isinstance(node, UnaryOp):
        return _scale_nodes(node.child)
    return [*_scale_nodes(node.left), *_scale_nodes(node.right)]


def _canonical(node: Node) -> tuple:
    """Order-insensitive structural key: slot indices and sum ordering do not matter."""
    if isinstance(node, Signal):
        return ("sig", node.name)
    if isinstance(node, Const):
        return ("const", round(node.value, 9))
    if isinstance(node, Scale):
        return ("scale", _canonical(node.child))
    if isinstance(node, SumOp):
        return ("sum", tuple(sorted(_canonical(child) for child in node.children)))
    if isinstance(node, ProductOp):
        return ("prod", tuple(sorted(_canonical(factor) for factor in node.factors)))
    if isinstance(node, UnaryOp):
        return ("unary", node.fn, _canonical(node.child))
    if node.fn in BINARY_FNS:
        return ("binary", node.fn) + tuple(
            sorted((_canonical(node.left), _canonical(node.right)))
        )
    return ("binary", node.fn, _canonical(node.left), _canonical(node.right))


def _render(node: Node, gains: Sequence[float] | None, names: Mapping[int, str]) -> str:
    def token_for(k: int) -> str:
        if gains is not None:
            return f"({gains[k]:.4g})"
        return names.get(k, f"K{k + 1}")

    if isinstance(node, Signal):
        return node.name
    if isinstance(node, Const):
        return f"{node.value:.6g}"
    if isinstance(node, Scale):
        return f"{token_for(node.k)}*{_render(node.child, gains, names)}"
    if isinstance(node, SumOp):
        return " + ".join(_render(child, gains, names) for child in node.children)
    if isinstance(node, ProductOp):
        return "*".join(_render(factor, gains, names) for factor in node.factors)
    if isinstance(node, UnaryOp):
        return f"{node.fn}({_render(node.child, gains, names)})"
    return f"{node.fn}({_render(node.left, gains, names)}, {_render(node.right, gains, names)})"


def _to_tree(node: Node) -> dict[str, object]:
    if isinstance(node, Signal):
        return {"op": "signal", "name": node.name}
    if isinstance(node, Const):
        return {"op": "const", "value": node.value}
    if isinstance(node, Scale):
        return {"op": "scale", "child": _to_tree(node.child)}
    if isinstance(node, SumOp):
        return {"op": "sum", "children": [_to_tree(child) for child in node.children]}
    if isinstance(node, ProductOp):
        return {"op": "product", "factors": [_to_tree(f) for f in node.factors]}
    if isinstance(node, UnaryOp):
        return {"op": "unary", "fn": node.fn, "child": _to_tree(node.child)}
    return {
        "op": "binary",
        "fn": node.fn,
        "left": _to_tree(node.left),
        "right": _to_tree(node.right),
    }


def _as_mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be an object")
    return value


def _from_tree(payload: Mapping[str, object]) -> Node:
    op = str(payload.get("op", ""))
    if op == "signal":
        return Signal(str(payload["name"]))
    if op == "const":
        return Const(float(payload["value"]))  # type: ignore[arg-type]
    if op == "scale":
        return Scale(-1, _from_tree(_as_mapping(payload["child"], "scale child")))
    if op == "sum":
        children = payload.get("children")
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
            raise ValueError("sum requires a children list")
        return _make_sum([_from_tree(_as_mapping(child, "sum child")) for child in children])
    if op == "product":
        factors = payload.get("factors")
        if not isinstance(factors, Sequence) or isinstance(factors, (str, bytes)):
            raise ValueError("product requires a factors list")
        return _make_product(
            [_from_tree(_as_mapping(factor, "product factor")) for factor in factors]
        )
    if op == "unary":
        fn = str(payload["fn"])
        if fn not in UNARY_FNS:
            raise ValueError(f"unknown unary function {fn!r}")
        return UnaryOp(fn, _from_tree(_as_mapping(payload["child"], "unary child")))
    if op == "binary":
        fn = str(payload["fn"])
        if fn not in BINARY_FNS:
            raise ValueError(f"unknown binary function {fn!r}")
        return BinaryOp(
            fn,
            _from_tree(_as_mapping(payload["left"], "binary left")),
            _from_tree(_as_mapping(payload["right"], "binary right")),
        )
    raise ValueError(f"unknown expression op {op!r}")


def _finalize_slots(root: Node, gain_keys: Mapping[int, str] | None = None) -> Node:
    """Assign every Scale a concrete gain-vector slot in pre-order.

    String-parsed trees pass ``gain_keys`` mapping temporary node ids to shared
    K-token names so that repeated names reuse one slot (tied gains).
    """
    key_to_slot: dict[str, int] = {}
    temp_to_key = dict(gain_keys or {})
    next_slot = [0]

    def walk(node: Node) -> Node:
        if isinstance(node, (Signal, Const)):
            return node
        if isinstance(node, Scale):
            # Assign this wrapper's slot before walking children (true pre-order)
            # so textual K numbering matches evaluation order.
            key = temp_to_key.get(id(node))
            if key is not None and key in key_to_slot:
                slot = key_to_slot[key]
            else:
                slot = next_slot[0]
                next_slot[0] += 1
                if key is not None:
                    key_to_slot[key] = slot
            child = walk(node.child)
            return replace(node, k=slot, child=child)
        if isinstance(node, SumOp):
            return SumOp(tuple(walk(child) for child in node.children))
        if isinstance(node, ProductOp):
            return ProductOp(tuple(walk(factor) for factor in node.factors))
        if isinstance(node, UnaryOp):
            return UnaryOp(node.fn, walk(node.child))
        return BinaryOp(node.fn, walk(node.left), walk(node.right))

    return walk(root)


def _validate_tree(root: Node) -> None:
    depth = _depth(root)
    if depth > MAX_DEPTH:
        raise ValueError(f"expression depth {depth} exceeds the maximum of {MAX_DEPTH}")
    nodes = _count_nodes(root)
    if nodes > MAX_NODES:
        raise ValueError(f"expression node count {nodes} exceeds the maximum of {MAX_NODES}")
    scales = _scale_nodes(root)
    if not scales:
        raise ValueError("expression requires at least one K gain slot")
    names: list[str] = []
    _signals(root, names)
    if not names:
        raise ValueError("expression requires at least one signal")


def _safe_square(value: np.ndarray) -> np.ndarray:
    return np.square(np.clip(value, -1e4, 1e4))


def _safe_exp(value: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(value, -15.0, 15.0))


def _safe_sqrt(value: np.ndarray) -> np.ndarray:
    return np.sign(value) * np.sqrt(np.abs(value))


class _GainRef:
    def __init__(self, name: str) -> None:
        self.name = name


class _Parser:
    """Recursive-descent parser for the expression string grammar.

    expression := term (('+' | '-') term)*
    term       := factor ('*' factor)*
    factor     := '-' factor | atom
    atom       := '(' expression ')' | call | gain | number | signal
    call       := fname '(' expression [',' expression] ')'

    A gain token (``K``, ``K3``, ``k12``) multiplies the enclosing factor
    product and marks one scalar parameter slot; repeated names share a slot.
    """

    def __init__(self, text: str) -> None:
        self.tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if match is None:
                remainder = text[position:].strip()
                if not remainder:
                    break
                raise ValueError(f"unexpected character in expression: {remainder[:10]!r}")
            if match.group("name"):
                self.tokens.append(("name", match.group("name")))
            elif match.group("num"):
                self.tokens.append(("num", match.group("num")))
            else:
                self.tokens.append(("op", match.group("op")))
            position = match.end()
        self.index = 0
        self.gain_keys: dict[int, str] = {}

    def parse(self) -> Node:
        if not self.tokens:
            raise ValueError("empty expression")
        node = self._expression()
        if self.index != len(self.tokens):
            value = self.tokens[self.index][1]
            raise ValueError(f"unexpected trailing token {value!r} in expression")
        return node

    def _peek_op(self, *ops: str) -> str | None:
        if self.index < len(self.tokens):
            kind, value = self.tokens[self.index]
            if kind == "op" and value in ops:
                return value
        return None

    def _expression(self) -> Node:
        terms = [self._term()]
        while (op := self._peek_op("+", "-")) is not None:
            self.index += 1
            term = self._term()
            terms.append(UnaryOp("neg", term) if op == "-" else term)
        return _make_sum(terms)

    def _term(self) -> Node:
        factors: list[Node] = [self._factor()]
        while self._peek_op("*") is not None:
            self.index += 1
            factors.append(self._factor())
        gain_names = [item.name for item in factors if isinstance(item, _GainRef)]
        rest = [item for item in factors if not isinstance(item, _GainRef)]
        if not rest:
            raise ValueError("a gain token must multiply at least one signal")
        node = rest[0] if len(rest) == 1 else _make_product(rest)
        for name in reversed(gain_names):
            wrapper = Scale(-1, node)
            self.gain_keys[id(wrapper)] = name
            node = wrapper
        return node

    def _factor(self) -> Node:
        if self._peek_op("-") is not None:
            self.index += 1
            inner = self._factor()
            if isinstance(inner, _GainRef):
                raise ValueError("unary minus cannot precede a gain token; fold it into K")
            return UnaryOp("neg", inner)
        return self._atom()

    def _atom(self) -> Node:
        if self.index >= len(self.tokens):
            raise ValueError("expression ended unexpectedly")
        kind, value = self.tokens[self.index]
        if kind == "op" and value == "(":
            self.index += 1
            node = self._expression()
            if self._peek_op(")") is None:
                raise ValueError("missing closing parenthesis")
            self.index += 1
            return node
        if kind == "num":
            self.index += 1
            return Const(float(value))
        if kind != "name":
            raise ValueError(f"unexpected token {value!r} in expression")
        self.index += 1
        if self._peek_op("(") is not None:
            return self._call(value)
        if _GAIN_NAME.match(value):
            return _GainRef(value)
        return Signal(value)

    def _call(self, name: str) -> Node:
        self.index += 1  # consume "("
        lowered = name.lower()
        if lowered in UNARY_FNS:
            child = self._expression()
            if self._peek_op(")") is None:
                raise ValueError(f"{name}(...) requires exactly one argument")
            self.index += 1
            return UnaryOp(lowered, child)
        if lowered in BINARY_FNS:
            left = self._expression()
            if self._peek_op(",") is None:
                raise ValueError(f"{name}(...) requires two comma-separated arguments")
            self.index += 1
            right = self._expression()
            if self._peek_op(")") is None:
                raise ValueError(f"{name}(...) requires exactly two arguments")
            self.index += 1
            return BinaryOp(lowered, left, right)
        raise ValueError(f"unknown function {name!r}")


def parse_expression(text: str) -> tuple[Node, dict[int, str]]:
    """Parse an expression string into a tree plus K-token names keyed by node id."""
    parser = _Parser(text)
    root = parser.parse()
    return root, parser.gain_keys


class SymbolicExpression:
    """A named free-form symbolic controller law with CEM-tuned gain slots."""

    def __init__(self, name: str, expression: str | Mapping | Sequence[str] | Node) -> None:
        gain_keys: dict[int, str] | None = None
        if isinstance(expression, str):
            root, gain_keys = parse_expression(expression)
            root = _finalize_slots(root, gain_keys)
        elif isinstance(expression, Mapping):
            if "expression" in expression and isinstance(expression["expression"], str):
                root, gain_keys = parse_expression(str(expression["expression"]))
                root = _finalize_slots(root, gain_keys)
            elif "op" in expression:
                root = _finalize_slots(_from_tree(expression))
            else:
                raise ValueError("mapping expression requires 'expression' text or a JSON tree")
        elif isinstance(expression, Sequence) and not isinstance(expression, (str, bytes)):
            if not expression:
                raise ValueError("a legacy term list must not be empty")
            root = _finalize_slots(_make_sum([Scale(-1, Signal(str(t))) for t in expression]))
        elif isinstance(expression, (Signal, Const, Scale, SumOp, ProductOp, UnaryOp, BinaryOp)):
            root = _finalize_slots(expression)
        else:
            raise TypeError("expression must be a string, mapping, term list, or node")

        _validate_tree(root)
        names: list[str] = []
        _signals(root, names)
        for signal in names:
            _signal_name(Signal(signal))

        self.name = name
        self.root = root
        self._signals = tuple(names)
        scales = _scale_nodes(root)
        distinct = {node.k for node in scales}
        self._parameter_count = max(distinct) + 1
        self._gain_names = {
            node.k: gain_keys[id(node)]
            for node in scales
            if gain_keys is not None and id(node) in gain_keys
        }
        # Later duplicates win deliberately: identical names map to one slot.
        deduped: dict[int, str] = {}
        for node in scales:
            if node.k in self._gain_names:
                deduped[node.k] = self._gain_names[node.k]
        self._gain_names = deduped

    # -- basic properties -----------------------------------------------------

    @property
    def signals(self) -> tuple[str, ...]:
        """Distinct signal names used by the expression, in first-use order."""
        return self._signals

    @property
    def terms(self) -> tuple[str, ...]:
        """Backward-compatible view: the distinct signals, like the old term list."""
        return self._signals

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @property
    def complexity(self) -> int:
        """Structural complexity: every non-parameter node counts once."""
        return _count_nodes(self.root)

    @property
    def node_count(self) -> int:
        return _count_nodes(self.root)

    # -- identity -------------------------------------------------------------

    def key(self) -> tuple:
        return ("expr", _canonical(self.root), self.parameter_count)

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "expression": self.to_expression_string(),
            "tree": _to_tree(self.root),
            "signals": list(self._signals),
            "parameter_count": self._parameter_count,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SymbolicExpression:
        name = str(payload.get("name", "structure"))
        if "expression" in payload:
            value = payload["expression"]
            if isinstance(value, str):
                return cls(name, value)
            if isinstance(value, Mapping):
                return cls(name, value)
        if "tree" in payload:
            return cls(name, _as_mapping(payload["tree"], "tree"))
        if "terms" in payload:
            terms = payload["terms"]
            if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes)):
                return cls(name, [str(term) for term in terms])
            raise ValueError("legacy terms must be a list of signal names")
        raise ValueError("structure payload requires an expression, tree, or legacy terms")

    def to_expression_string(self) -> str:
        return _render(self.root, None, self._gain_names)

    def renamed(self, name: str) -> SymbolicExpression:
        return SymbolicExpression(name, self.root)

    @classmethod
    def linear(cls, name: str, signals: Sequence[str]) -> SymbolicExpression:
        if not signals:
            raise ValueError("a linear law requires at least one signal")
        return cls(name, " + ".join(f"K{i + 1}*{signal}" for i, signal in enumerate(signals)))

    # -- evaluation -----------------------------------------------------------

    def validate(self, allowed: Sequence[str]) -> None:
        unknown = [signal for signal in self._signals if signal not in set(allowed)]
        if unknown:
            raise ValueError(
                f"expression uses unavailable signals: {', '.join(unknown)}; "
                f"allowed: {', '.join(allowed)}"
            )

    def formula(self, gains: Sequence[float]) -> str:
        return _render(self.root, list(gains), self._gain_names)

    def evaluate(self, signal_values: Mapping[str, object], gains: Sequence[float]) -> np.ndarray:
        missing = [signal for signal in self._signals if signal not in signal_values]
        if missing:
            raise ValueError(f"missing signal values: {', '.join(missing)}")
        values = {
            signal: np.asarray(signal_values[signal], dtype=float) for signal in self._signals
        }
        vector = np.asarray(gains, dtype=float)
        if vector.shape != (self._parameter_count,):
            raise ValueError("gain vector does not match the expression parameter slots")
        result = self._eval_numpy(self.root, values, vector)
        return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    def _eval_numpy(
        self, node: Node, values: dict[str, np.ndarray], gains: np.ndarray
    ) -> np.ndarray:
        if isinstance(node, Signal):
            return values[node.name]
        if isinstance(node, Const):
            return np.asarray(node.value, dtype=float)
        if isinstance(node, Scale):
            return gains[node.k] * self._eval_numpy(node.child, values, gains)
        if isinstance(node, SumOp):
            total = np.zeros_like(self._eval_numpy(node.children[0], values, gains))
            for child in node.children:
                total = total + self._eval_numpy(child, values, gains)
            return total
        if isinstance(node, ProductOp):
            result = self._eval_numpy(node.factors[0], values, gains)
            for factor in node.factors[1:]:
                result = result * self._eval_numpy(factor, values, gains)
            return result
        if isinstance(node, UnaryOp):
            value = self._eval_numpy(node.child, values, gains)
            if node.fn == "tanh":
                return np.tanh(value)
            if node.fn == "sin":
                return np.sin(value)
            if node.fn == "cos":
                return np.cos(value)
            if node.fn == "sqrt":
                return _safe_sqrt(value)
            if node.fn == "square":
                return _safe_square(value)
            if node.fn == "abs":
                return np.abs(value)
            if node.fn == "exp":
                return _safe_exp(value)
            return -value
        left = self._eval_numpy(node.left, values, gains)
        right = self._eval_numpy(node.right, values, gains)
        return np.minimum(left, right) if node.fn == "min" else np.maximum(left, right)

    def evaluate_torch(self, signal_values: Mapping[str, object], gains: object) -> object:
        """Torch twin of :meth:`evaluate` for GPU-batched rollouts (Genesis).

        ``signal_values`` maps names to ``(batch, action_dim)`` tensors and
        ``gains`` is a ``(batch, parameter_count)`` tensor.
        """
        import torch

        missing = [signal for signal in self._signals if signal not in signal_values]
        if missing:
            raise ValueError(f"missing signal values: {', '.join(missing)}")
        return torch.nan_to_num(
            self._eval_torch(self.root, signal_values, gains), nan=0.0, posinf=1e6, neginf=-1e6
        )

    def _eval_torch(self, node: Node, values: Mapping[str, object], gains: object) -> object:
        import torch

        if isinstance(node, Signal):
            return values[node.name]
        if isinstance(node, Const):
            return torch.as_tensor(node.value, dtype=values[self._signals[0]].dtype)  # type: ignore[union-attr]
        if isinstance(node, Scale):
            return gains[:, node.k : node.k + 1] * self._eval_torch(node.child, values, gains)
        if isinstance(node, SumOp):
            total = None
            for child in node.children:
                value = self._eval_torch(child, values, gains)
                total = value if total is None else total + value
            return total
        if isinstance(node, ProductOp):
            result = None
            for factor in node.factors:
                value = self._eval_torch(factor, values, gains)
                result = value if result is None else result * value
            return result
        if isinstance(node, UnaryOp):
            value = self._eval_torch(node.child, values, gains)
            if node.fn == "tanh":
                return torch.tanh(value)
            if node.fn == "sin":
                return torch.sin(value)
            if node.fn == "cos":
                return torch.cos(value)
            if node.fn == "sqrt":
                return torch.sign(value) * torch.sqrt(torch.abs(value))
            if node.fn == "square":
                return torch.square(torch.clamp(value, -1e4, 1e4))
            if node.fn == "abs":
                return torch.abs(value)
            if node.fn == "exp":
                return torch.exp(torch.clamp(value, -15.0, 15.0))
            return -value
        left = self._eval_torch(node.left, values, gains)
        right = self._eval_torch(node.right, values, gains)
        return torch.minimum(left, right) if node.fn == "min" else torch.maximum(left, right)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SymbolicExpression({self.name!r}, {self.to_expression_string()!r})"
