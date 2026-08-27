# Goal-aware long evolution with deeper gain tuning

## Protocol

- Outer evolution: 8 generations x 6 proposals for each of 3 environments.
- Gain tuning: 10 CEM iterations x 32 candidate gain vectors per structure, plus the
  initial zero-gain evaluation.
- Training: 6 randomized seeds; held-out evaluation: 30 randomized seeds.
- Prompt goals: task success and return, lower control energy, lower command jerk, and
  Pareto-diverse performance/energy/jerk/balanced proposals.
- Model: NVIDIA NIM `openai/gpt-oss-120b`, high reasoning effort.

NIM service reliability was poor during this run: 55 of 62 calls returned `None`, so 17
of 24 environment-generation steps used deterministic elite-local mutations after three
failed attempts. Results therefore represent the complete evolution pipeline, but not 24
fully LLM-generated generations.

## Held-out results

### Pendulum-v1

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| PD+CEM | **-825.850** | **30.0%** | 29.346 | 1424.934 |
| Evolved+CEM | -842.188 | 20.0% | **28.930** | **693.830** |

The evolved structure roughly halves jerk and slightly reduces energy, but gives up return
and success relative to PD. Its terms are `angular_velocity`, `sin_angle`, and
`tanh_velocity`.

### InvertedPendulum-v5

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| PD+CEM | **500.000** | **100.0%** | **0.0139** | **5.333** |
| Evolved+CEM | 485.300 | 83.3% | 0.1358 | 8.567 |

PD remains clearly preferable. The evolved `cart_position + tanh_angle` structure fit all
training seeds but did not generalize to the held-out physical variations.

### Reacher-v5

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| Task PD+CEM | -7.320 | 36.7% | **0.0035** | **0.862** |
| Evolved+CEM | **-5.377** | **90.0%** | 0.0163 | 23.130 |

The evolved structure substantially improves return and success, but remains much less
smooth and uses more energy than Task PD. Its terms are `joint_velocity`,
`tanh_jt_error`, `normalized_jt_error`, and `task_damping`.

## Comparison with the preceding 6-generation run

The deeper run improves Pendulum energy and jerk but worsens return, leaves
InvertedPendulum unsuccessful relative to PD, and modestly improves Reacher return,
success, and energy while increasing jerk. Increasing the CEM budget therefore improves
gain optimization but does not remove structural overfitting or the task/smoothness
trade-off.
