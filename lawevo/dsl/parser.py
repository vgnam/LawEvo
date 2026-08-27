from __future__ import annotations

import json
import re
from collections.abc import Mapping

from lawevo.dsl.ast import Barrier, Combine, Primitive, WeightedTerm


class BarrierSyntaxError(ValueError):
    pass


_TOKEN = re.compile(
    r"\s*(?:(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|(?P<punct>[(),*]))"
)


class _Parser:
    def __init__(self, source: str) -> None:
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        while pos < len(source):
            match = _TOKEN.match(source, pos)
            if not match:
                raise BarrierSyntaxError(f"unexpected character at offset {pos}")
            kind = (
                "number" if match.group("number") else "ident" if match.group("ident") else "punct"
            )
            self.tokens.append((kind, match.group(kind)))
            pos = match.end()
        self.index = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, expected: str | None = None) -> tuple[str, str]:
        token = self.peek()
        if token is None:
            raise BarrierSyntaxError("unexpected end of expression")
        if expected is not None and token[1] != expected:
            raise BarrierSyntaxError(f"expected {expected!r}, got {token[1]!r}")
        self.index += 1
        return token

    def parse(self) -> Combine:
        op = self.take()[1]
        if op not in {"min", "wsum"}:
            raise BarrierSyntaxError("top-level expression must be min(...) or wsum(...)")
        self.take("(")
        terms: list[Primitive | WeightedTerm] = []
        while True:
            if op == "wsum":
                kind, value = self.take()
                if kind != "number":
                    raise BarrierSyntaxError("a weighted term must start with a numeric weight")
                weight = float(value)
                self.take("*")
                terms.append(WeightedTerm(weight, self.parse_primitive()))
            else:
                terms.append(self.parse_primitive())
            if self.peek() and self.peek()[1] == ",":
                self.take(",")
                continue
            break
        self.take(")")
        if self.peek() is not None:
            raise BarrierSyntaxError(f"unexpected trailing token {self.peek()[1]!r}")
        return Combine(op, tuple(terms))

    def parse_primitive(self) -> Primitive:
        kind, name = self.take()
        if kind != "ident" or name not in {
            "dist_to_obstacle",
            "speed_margin",
            "joint_margin",
            "boundary_margin",
        }:
            raise BarrierSyntaxError(f"unknown primitive {name!r}")
        self.take("(")
        args: list[object] = []
        while self.peek() and self.peek()[1] != ")":
            arg_kind, raw = self.take()
            if arg_kind == "number":
                number = float(raw)
                args.append(
                    int(number)
                    if number.is_integer() and "." not in raw and "e" not in raw.lower()
                    else number
                )
            elif arg_kind == "ident":
                args.append(raw)
            else:
                raise BarrierSyntaxError(f"invalid primitive argument {raw!r}")
            if self.peek() and self.peek()[1] == ",":
                self.take(",")
            else:
                break
        self.take(")")
        return Primitive(name, tuple(args))


def _primitive_from_dict(data: Mapping[str, object]) -> Primitive:
    try:
        name = str(data["primitive"])
        args = data.get("args", [])
    except KeyError as exc:
        raise BarrierSyntaxError("primitive object requires a 'primitive' field") from exc
    if not isinstance(args, list):
        raise BarrierSyntaxError("primitive 'args' must be a list")
    return Primitive(name, tuple(args))


def _from_dict(data: Mapping[str, object]) -> Combine:
    op = data.get("op")
    terms = data.get("terms")
    if op not in {"min", "wsum"} or not isinstance(terms, list):
        raise BarrierSyntaxError("barrier object requires op=min|wsum and a terms list")
    parsed: list[Primitive | WeightedTerm] = []
    for item in terms:
        if not isinstance(item, Mapping):
            raise BarrierSyntaxError("every term must be an object")
        if op == "min":
            parsed.append(_primitive_from_dict(item))
        else:
            if "weight" not in item or not isinstance(item.get("term"), Mapping):
                raise BarrierSyntaxError("wsum terms require 'weight' and 'term'")
            parsed.append(WeightedTerm(float(item["weight"]), _primitive_from_dict(item["term"])))
    return Combine(str(op), tuple(parsed))


def parse_barrier(source: str | Mapping[str, object]) -> Barrier:
    """Parse either the EBNF expression or its JSON tree representation."""
    if isinstance(source, Mapping):
        return _from_dict(source)
    text = source.strip()
    if text.startswith("{"):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BarrierSyntaxError(str(exc)) from exc
        if not isinstance(decoded, Mapping):
            raise BarrierSyntaxError("JSON barrier must be an object")
        return _from_dict(decoded)
    return _Parser(text).parse()
