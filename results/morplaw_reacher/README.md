# MorpLaw run

- environment: `Reacher-v5`
- model: `openai/gpt-oss-120b`
- config: {
  "generations": 20,
  "proposals_per_side": 4,
  "joint_top_k": 2,
  "cem_iterations": 5,
  "cem_population": 24,
  "train_episodes": 6,
  "test_episodes": 30,
  "variants": [
    "both"
  ]
}
- variants: both

## Best pair per variant (train-tuned; held-out means in `results.json`)

| variant | best pair | train score | held-out score |
|---|---|---:|---:|
| both | `BoundedError_TaskDamping@{density0=600, density1=600, gear=400, l0=0.06, l1=0.06, r0=0.012, r1=0.012}` | -4.098 | -3.783 |