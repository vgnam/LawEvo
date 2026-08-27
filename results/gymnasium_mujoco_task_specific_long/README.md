# Long task-specific structure evolution

## Protocol

- Environments: `Pendulum-v1`, `InvertedPendulum-v5`, and `Reacher-v5`.
- Outer evolution: 6 generations x 6 requested structures per environment.
- Inner tuning: 5 CEM iterations x 24 candidates per structure on 6 training seeds.
- Evaluation: 30 held-out seeds per environment with the existing parameter randomization.
- Generator: NVIDIA NIM `openai/gpt-oss-120b`, high reasoning effort.
- Prompt context: environment dynamics, observation/action semantics, horizon, success and
  failure conditions, randomization, basis-signal semantics, exact objective weights,
  current elites, and the complete evaluated-structure archive.

The NIM endpoint returned empty content frequently during this run. Across 34 generation
calls, 22 responses were `None`; 7 of the 18 environment-generation steps therefore used
the deterministic elite-local mutation fallback after three failed attempts. Raw responses
and fallback events are preserved in `nim_responses.json`.

## Held-out results

### Pendulum-v1

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| PD+CEM | -825.812 | **30.0%** | **29.291** | 1425.269 |
| Evolved+CEM | **-757.510** | 13.3% | 32.448 | **1253.279** |

The evolved structure improves held-out return by 68.302 and reduces jerk, but its final
success rate is 16.7 percentage points below PD. Its selected terms are `angle`,
`integral_angle`, `sin_angle`, and `sqrt_angle`.

### InvertedPendulum-v5

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| PD+CEM | **500.000** | **100.0%** | **0.0107** | 7.386 |
| Evolved+CEM | 479.833 | 83.3% | 0.0967 | **5.661** |

The evolved `pole_angle + tanh_cart` structure was perfect on the six training seeds but
did not generalize. PD remains preferable on held-out return, success, and energy.

### Reacher-v5

| Controller | Return ↑ | Success ↑ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|
| Task PI+CEM | -7.819 | 20.0% | **0.0052** | **0.566** |
| Evolved+CEM | **-5.496** | **86.7%** | 0.0174 | 12.539 |

The evolved structure improves held-out return by 2.323 and success by 66.7 percentage
points, at 3.36x the energy and 22.2x the jerk of Task PI. Its terms are
`integral_jt_error`, `tanh_jt_error`, `normalized_jt_error`, and `task_damping`.

## Conclusion

Longer task-specific evolution materially improves Pendulum return and Reacher return and
success, but it does not uniformly dominate classical controllers. InvertedPendulum shows
clear train-to-test overfitting, while Reacher retains a substantial smoothness and energy
trade-off. Because one run and one outer random trajectory are insufficient for a general
claim, repeat the complete protocol over independent NIM/CEM seeds before paper-level use.
