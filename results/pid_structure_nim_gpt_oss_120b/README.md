# PID Structure Evolution + CEM Gain Tuning

## Design

- Outer loop: NVIDIA NIM `openai/gpt-oss-120b` evolves symbolic controller structures.
- Inner loop: Cross-Entropy Method tunes every coefficient `K` for each structure.
- CEM budget: 7 iterations × 32 candidates = 224 simulator evaluations per structure.
- Structure DSL: conventional P/I/D signals, bounded nonlinear feedback, nonlinear error,
  distance-heading coupling, and three velocity gates. No generated Python is executed.
- Objective includes explicit jerk and term-count penalties.
- Train: 10 deterministic scenarios; test: 30 disjoint held-out scenarios.

## Evolved formula

Let `e_d` be distance error and `e_h` be wrapped heading error:

```text
v = max(0, cos(e_h)) * (
      1.009 * sqrt(e_d)
    - 1.479 * derivative(e_d)
)

omega = 1.581 * sign(e_h)*sqrt(abs(e_h))
      - 0.5354 * tanh(derivative(e_h))
```

This structure uses four tuned coefficients instead of the six coefficients of the fixed
dual-PID controller.

## Held-out comparison

| Method | Score ↑ | Final distance ↓ | Settling ↓ | Energy ↓ | Jerk ↓ | Success |
|---|---:|---:|---:|---:|---:|---:|
| Grid PID | 113.029 | 0.0726 | 3.410 s | 9.603 | 17.556 | 100% |
| Gain-only NIM PID | 113.505 | 0.0288 | 3.263 s | 10.023 | 30.424 | 100% |
| Fixed PID + CEM | 113.618 | **0.0210** | **3.257 s** | 10.068 | 27.088 | 100% |
| **Evolved structure + CEM** | **113.883** | 0.0212 | 3.263 s | **9.734** | **16.661** | **100%** |

Against fixed PID tuned by the same CEM budget, the evolved structure improved score by
0.234%, reduced energy by 3.31%, and reduced jerk by 38.49%. Final distance and settling
time were statistically indistinguishable on this set. Paired two-sided Wilcoxon results:

- Score: `p=5.59e-9`
- Energy: `p=7.99e-6`
- Jerk: `p=0.00237`
- Final distance: `p=0.984`
- Settling time: `p=0.236`

## Interpretation

The structural result is stronger than gain-only NIM for this objective: it preserves the
accuracy and speed of CEM-tuned PID while substantially reducing jerk with fewer terms.
This remains one outer-loop evolution run and one train/test split. Repeat the complete
outer evolution across at least five LLM/random seeds before making a paper-level
generalization claim.
