# Gymnasium and MuJoCo Classical-Control Benchmarks

## Protocol

- Environments: `Pendulum-v1`, `InvertedPendulum-v5`, and `Reacher-v5`.
- Classical initial population: P, PI, PD, PID; InvertedPendulum also includes a true LQR
  obtained by finite-difference MuJoCo linearization and a discrete Riccati solve.
- Proposed method: NVIDIA NIM GPT-OSS evolves signal structure; CEM tunes all coefficients.
- Equal inner budget: 5 CEM iterations × 24 candidates = 120 candidates per structure.
- Outer budget: 2 NIM generations × 4 requested structures per environment.
- Evaluation: 6 train and 30 held-out seeds per environment.
- Robustness: held-out physical parameter variations (±15% Pendulum/InvertedPendulum,
  ±10% Reacher) plus unseen initial states/targets.

## Results

### Pendulum-v1

Best classical test controller: PD.

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| PD+CEM | **-825.81** | **30.0%** | **29.29** | **1425.27** |
| NIM structure+CEM | -892.46 | 3.3% | 32.26 | 3950.71 |

NIM found a structure with better training return (`-627.49`) but it did not generalize.
The held-out return difference was not significant (`p=0.570`), while success was lower
(`p=0.0114`) and jerk was much higher (`p<1e-8`). This is a clear overfitting result.

NIM structure:

```text
1.945*angle + 0.537*integral_angle - 6.548*sin(angle) - 1.514*tanh(angle)
```

### InvertedPendulum-v5

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| PD+CEM | **500.0** | **100%** | **0.0107** | **7.39** |
| LQR | **500.0** | **100%** | 0.0823 | 127.71 |
| NIM structure+CEM | **500.0** | **100%** | 0.0533 | 10.69 |

All three stabilize every held-out episode. PD+CEM is the preferred controller because it
uses the least energy and has the lowest jerk. NIM does not beat the classical baseline.

### Reacher-v5

Best classical test controller by objective: task-space PI.

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| Task PI+CEM | -7.819 | 20.0% | **0.0052** | **0.566** |
| NIM structure+CEM | **-6.147** | **56.7%** | 0.0168 | 14.648 |

NIM improves return by 1.672 (`p<1e-8`) and success by 36.7 percentage points
(`p=0.00091`), but spends 3.24× the energy and produces 25.9× the jerk. Its evolved
structure is a saturated Jacobian-transpose PD:

```text
tau = 2.354*tanh(10*J^T*task_error) - 0.07365*joint_velocity
```

## Overall conclusion

Structure evolution is not uniformly superior: it wins task performance on Reacher,
ties but loses efficiency on InvertedPendulum, and overfits on Pendulum. These results
support reporting environment-specific trade-offs rather than claiming general dominance.
For a paper, repeat the full outer evolution over at least five independent NIM/CEM seeds.
