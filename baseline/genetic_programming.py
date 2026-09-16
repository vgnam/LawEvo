"""Elitist tree GP: tournament selection, subtree crossover and mutation.

Only structure evolves here. The shared benchmark runner tunes gains with CEM.
"""

from dataclasses import replace

import numpy as np

from lawevo.pid.expression import (
    BINARY_FNS,
    MAX_PARAMS,
    UNARY_FNS,
    BinaryOp,
    ProductOp,
    Scale,
    Signal,
    SumOp,
    SymbolicExpression,
    UnaryOp,
)


def children(node):
    if isinstance(node, (Scale, UnaryOp)):
        return (node.child,)
    if isinstance(node, SumOp):
        return node.children
    if isinstance(node, ProductOp):
        return node.factors
    if isinstance(node, BinaryOp):
        return (node.left, node.right)
    return ()


def rebuild(node, parts):
    if isinstance(node, (Scale, UnaryOp)):
        return replace(node, child=parts[0])
    if isinstance(node, SumOp):
        return SumOp(tuple(parts))
    if isinstance(node, ProductOp):
        return ProductOp(tuple(parts))
    if isinstance(node, BinaryOp):
        return replace(node, left=parts[0], right=parts[1])
    return node


def paths(node):
    yield ()
    for index, child in enumerate(children(node)):
        for path in paths(child):
            yield (index, *path)


def subtree(node, path):
    for index in path:
        node = children(node)[index]
    return node


def graft(node, path, donor):
    if not path:
        return donor
    parts = list(children(node))
    parts[path[0]] = graft(parts[path[0]], path[1:], donor)
    return rebuild(node, parts)


def separate_gains(node):
    result = rebuild(node, [separate_gains(child) for child in children(node)])
    return replace(result, k=result.k + MAX_PARAMS) if isinstance(result, Scale) else result


def propose(adapter, ranked, excluded, count, *, seed, generation, population_size=24):
    """Generate exactly count valid, novel trees or explicitly fail on exhaustion.

    The retained population is the best population_size trees from the archive.
    A generation-local RNG makes checkpoint resumes reproduce proposal generation.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, generation]))
    population = ranked[:population_size]
    if not population or count < 1 or population_size < 1:
        raise ValueError("GP needs a nonempty population and positive budgets")

    def choose_path(node):
        candidates = list(paths(node))
        return candidates[int(rng.integers(len(candidates)))]

    def tournament():
        candidates = rng.integers(len(population), size=min(3, len(population)))
        return population[int(min(candidates))]["structure"].root

    def grow(depth=2):
        if depth == 0 or rng.random() < 0.4:
            return Scale(-1, Signal(str(rng.choice(adapter.allowed_terms))))
        operation = int(rng.integers(4))
        if operation == 0:
            return UnaryOp(str(rng.choice(UNARY_FNS)), grow(depth - 1))
        left, right = grow(depth - 1), grow(depth - 1)
        if operation == 1:
            return SumOp((left, right))
        if operation == 2:
            return ProductOp((left, right))
        return BinaryOp(str(rng.choice(BINARY_FNS)), left, right)

    seen, proposals = set(excluded), []
    for _ in range(max(1000, count * 500)):
        parent = tournament()
        if rng.random() < 0.5:
            other = tournament()
            donor = separate_gains(subtree(other, choose_path(other)))
            operator = "crossover"
        else:
            donor = grow()
            operator = "mutation"
        try:
            law = SymbolicExpression(
                f"GP_{generation}_{operator}_{len(proposals) + 1}",
                graft(parent, choose_path(parent), donor),
            )
            adapter.validate_structure(law)
        except ValueError:
            continue
        if law.key() in seen:
            continue
        seen.add(law.key())
        proposals.append(law)
        if len(proposals) == count:
            return proposals
    raise RuntimeError(f"GP produced only {len(proposals)}/{count} valid novel trees")


if __name__ == "__main__":
    import sys

    from experiments.gymnasium_classical_benchmarks import main

    sys.argv.extend(["--search-method", "gp"])
    main()
