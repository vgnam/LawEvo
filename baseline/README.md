# Genetic Programming baseline

Run from the repository root. No LLM call or API key is needed for GP.

```powershell
cd D:\LawEvo
py -m baseline.genetic_programming --environment Reacher-v5 --gp-seed 1
```

Equivalent: `py -m experiments.gymnasium_classical_benchmarks --environment Reacher-v5 --search-method gp --gp-seed 1`.

GP evolves expression trees, with a 50/50 choice of subtree crossover and subtree
mutation. Size-3 tournaments select from the best `--gp-population` archived laws
(default 24); ranking uses training metrics only. Generation zero evaluates the
same classical structures as LawEvo. Elitism retains the best historical laws.
Mutation grows new trees over the shared signal and operator library. Donor gains
are separated from recipient gains to avoid accidental parameter sharing.
The same AST size/depth/parameter and unit checks reject invalid proposals.

The shared CEM tuner fits gains; GP does not evolve numeric gains. Default search
budget is 20 generations × 6 new valid unique trees, in addition to the classical
initialization. Each tree gets 5 CEM iterations × 24 candidates × 6 training
episodes, plus the initial-gain evaluation. `--gp-population` controls the retained
parent pool; `--proposals` controls new evaluations per generation. Compare with
LLM runs using identical settings and report actual evaluated tree counts, since
LLM failures can yield fewer proposals. This compares search quality at a matched
maximum simulation budget, not equal wall time.

Train and test seeds, helpers, success-first ranking, CEM seed policy, and held-out
evaluation are shared with LawEvo. `--gp-seed` changes GP structural search only.
Generation-local random streams plus saved plans support reproducible resume;
changing GP seed or population during resume is rejected.

Outputs: `results/<env_id>/<timestamp>/data/gp/` contains the best GP controller and
generation histories; `data/summary/` has CSV/JSON comparisons under the label
`Genetic Programming`. `selected_controller` may still select a classical baseline;
use the explicit GP row for GP-only comparisons. The `best_evolved` JSON field is
retained for compatibility and contains the best GP-only law in GP runs.

Three independent GP searches for all ten primary benchmarks:

```powershell
cd D:\LawEvo
$environments = @(
    "Reacher-v5", "LawevoReacherPayload-v0",
    "PandaPushDense-v3", "LawevoPandaPushLowFriction-v0",
    "PandaReachDense-v3", "LawevoPandaReachMovingSlow-v1",
    "PandaSlideDense-v3", "LawevoPandaSlideGate-v0",
    "InvertedPendulum-v5", "LawevoInvertedPendulumPulse-v0"
)
foreach ($environment in $environments) {
    foreach ($gpSeed in 1..3) {
        py -m baseline.genetic_programming --environment $environment --gp-seed $gpSeed `
            --generations 20 --proposals 6 --cem-iterations 5 --cem-population 24 `
            --train-episodes 6 --test-episodes 30
        if ($LASTEXITCODE -ne 0) { throw "GP failed: $environment seed=$gpSeed" }
    }
}
```

For a quick smoke run, use `--generations 1 --proposals 2 --cem-iterations 1
--cem-population 4 --train-episodes 1 --test-episodes 2`.
