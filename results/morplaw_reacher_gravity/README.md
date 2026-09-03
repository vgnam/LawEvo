# MorpLaw run

- environment: `Reacher-v5`
- model: `openai/gpt-oss-120b`
- config: {
  "generations": 20,
  "proposals_per_side": 4,
  "responsive_per_side": 1,
  "joint_top_k": 2,
  "cem_iterations": 6,
  "cem_population": 32,
  "knowledge_capacity": 32,
  "retrieve_per_polarity": 4,
  "train_episodes": 8,
  "test_episodes": 50,
  "grammar_seeds": 3,
  "variants": [
    "full"
  ]
}
- variants: full

## Best pair per variant (train-tuned; held-out means in `results.json`)

| variant | best pair | train score | held-out score |
|---|---|---:|---:|
| full | `Bounded PD without Integral (Ablation Test)@{density0=900, density1=900, gear=400, l0=0.14, l1=0.14, r0=0.011, r1=0.011}` | -6.036 | -5.798 |