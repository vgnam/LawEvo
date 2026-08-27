# Provenance-clean Structure Evolution

The initial population contained only classical P, PI, PD, and PID structures. Therefore,
every nonlinear structure was proposed by NVIDIA NIM `openai/gpt-oss-120b`. Each structure's
coefficients were tuned independently by CEM using the same budget: 7 iterations × 32
candidates = 224 simulator evaluations.

## Winning NIM-proposed structure

```text
v = cos(e_heading) * (
      3.076 * tanh(e_distance)
    - 1.178 * tanh(derivative(e_distance))
)

omega = 3.910 * tanh(e_heading)
      - 0.5722 * tanh(derivative(e_heading))
```

This controller has four coefficients versus six in the fixed dual-PID. The outer loop
selected the basis functions and velocity gate; CEM determined the four numeric values.

## Held-out results (30 scenarios)

| Method | Score ↑ | Final distance ↓ | Settling ↓ | Energy ↓ | Jerk ↓ | Success |
|---|---:|---:|---:|---:|---:|---:|
| Grid PID | 113.029 | 0.0726 | 3.410 s | **9.603** | **17.556** | 100% |
| Gain-only NIM PID | 113.505 | 0.0288 | 3.263 s | 10.023 | 30.424 | 100% |
| Fixed PID + CEM | 113.618 | 0.0210 | 3.257 s | 10.068 | 27.088 | 100% |
| **NIM structure + CEM** | **113.990** | **0.0209** | **3.160 s** | 9.932 | 18.336 | **100%** |

Relative to fixed PID tuned with the identical CEM budget, the NIM structure:

- improved objective score by 0.328% (`p=1.86e-9`),
- reduced settling time by 2.97% (`p=1.93e-4`),
- reduced energy by 1.35% (`p=0.00172`),
- reduced jerk by 32.31% (`p=0.00123`), and
- had statistically indistinguishable final distance (`p=0.670`).

P-values are paired two-sided Wilcoxon tests across held-out scenarios. This is still one
outer-loop evolution run; repeat across independent LLM/CEM seeds before making a general
paper-level claim.
