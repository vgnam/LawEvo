from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from lawevo.dsl.ast import Barrier
from lawevo.evolve.belief import BeliefSpace, Experience
from lawevo.filter import CBFSafetyFilter
from lawevo.robot.base import Array, RobotInterface
from lawevo.sim.rollout import NominalPolicy, RolloutConfig, Trajectory, rollout
from lawevo.verify import BarrierVerifier, VerificationResult


@dataclass(frozen=True)
class EvaluationScenario:
    initial_state: Array
    goal: Array


@dataclass
class Candidate:
    name: str
    policy: NominalPolicy
    barrier: Barrier
    policy_source: str = ""
    parent: str | None = None
    fitness: float = -math.inf
    trajectories: list[Trajectory] = field(default_factory=list)
    verification: VerificationResult | None = None


class OffspringGenerator(Protocol):
    def __call__(
        self,
        survivors: Sequence[Candidate],
        belief: BeliefSpace,
        count: int,
        generation: int,
    ) -> Sequence[Candidate]: ...


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 30
    population_size: int = 20
    survivor_count: int = 5
    energy_weight: float = 0.02
    jerk_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.generations < 1 or self.population_size < 1:
            raise ValueError("generations and population_size must be positive")
        if not 1 <= self.survivor_count <= self.population_size:
            raise ValueError("survivor_count must be within population size")


@dataclass(frozen=True)
class GenerationReport:
    generation: int
    pass_rate: float
    best_fitness: float
    rejected: int
    population_size: int


class EvolutionRunner:
    """Knowledge-augmented evolutionary loop with injected LLM generation.

    Generation is injected so experiments can enforce budgets, select a provider,
    validate/compile policy code in a sandbox, and run the no-belief ablation without
    coupling the scientific core to an API.
    """

    def __init__(
        self,
        robot: RobotInterface,
        verifier: BarrierVerifier,
        scenarios: Sequence[EvaluationScenario],
        offspring_generator: OffspringGenerator,
        config: EvolutionConfig | None = None,
        rollout_config: RolloutConfig | None = None,
        use_belief_space: bool = True,
    ) -> None:
        if not scenarios:
            raise ValueError("at least one evaluation scenario is required")
        self.robot = robot
        self.verifier = verifier
        self.scenarios = tuple(scenarios)
        self.offspring_generator = offspring_generator
        self.config = config or EvolutionConfig()
        self.rollout_config = rollout_config or RolloutConfig()
        self.use_belief_space = use_belief_space
        self.belief = BeliefSpace()

    def _evaluate(self, candidate: Candidate) -> list[Experience]:
        candidate.verification = self.verifier.verify(candidate.barrier)
        candidate.trajectories.clear()
        if not candidate.verification.accepted or candidate.verification.alpha is None:
            candidate.fitness = -math.inf
            return [
                Experience(
                    "failure",
                    f"{candidate.barrier.to_dict()} rejected: {candidate.verification.reason}",
                )
            ]
        safety_filter = CBFSafetyFilter(self.robot, candidate.barrier, candidate.verification.alpha)
        for scenario in self.scenarios:
            candidate.trajectories.append(
                rollout(
                    self.robot,
                    candidate.policy,
                    safety_filter,
                    scenario.initial_state,
                    scenario.goal,
                    self.rollout_config,
                )
            )
        candidate.fitness = float(
            np.mean(
                [
                    item.fitness(self.config.energy_weight, self.config.jerk_weight)
                    for item in candidate.trajectories
                ]
            )
        )
        experiences = [
            Experience(
                "primitive",
                f"{candidate.barrier.to_dict()} verified with k={candidate.verification.alpha:.4g} "
                f"and fitness={candidate.fitness:.4g}",
                candidate.fitness,
            )
        ]
        if candidate.parent is not None:
            experiences.append(
                Experience(
                    "code_idiom",
                    f"policy {candidate.name} from {candidate.parent}: fitness={candidate.fitness:.4g}",
                    candidate.fitness,
                )
            )
        return experiences

    def run(
        self, initial_population: Sequence[Candidate]
    ) -> tuple[Candidate, list[GenerationReport]]:
        if not initial_population:
            raise ValueError("initial population must not be empty")
        population = list(initial_population)[: self.config.population_size]
        reports: list[GenerationReport] = []
        best: Candidate | None = None
        for generation in range(1, self.config.generations + 1):
            experiences: list[Experience] = []
            for candidate in population:
                experiences.extend(self._evaluate(candidate))
            ranked = sorted(population, key=lambda item: item.fitness, reverse=True)
            survivors = ranked[: self.config.survivor_count]
            if best is None or survivors[0].fitness > best.fitness:
                best = survivors[0]
            accepted = sum(
                bool(item.verification and item.verification.accepted) for item in population
            )
            reports.append(
                GenerationReport(
                    generation,
                    accepted / len(population),
                    survivors[0].fitness,
                    len(population) - accepted,
                    len(population),
                )
            )
            if self.use_belief_space:
                self.belief.update(experiences)
            if generation == self.config.generations:
                break
            count = self.config.population_size - len(survivors)
            children = list(
                self.offspring_generator(
                    survivors,
                    self.belief if self.use_belief_space else BeliefSpace(),
                    count,
                    generation,
                )
            )
            if len(children) != count:
                raise ValueError(
                    f"offspring generator returned {len(children)} candidates, expected {count}"
                )
            population = survivors + children
        assert best is not None
        return best, reports
