# MorpLaw run

- environment: `Ant-v5`
- model: `openai/gpt-oss-120b`
- config: {
  "generations": 20,
  "proposals_per_side": 4,
  "responsive_per_side": 1,
  "joint_top_k": 2,
  "cem_iterations": 5,
  "cem_population": 24,
  "knowledge_capacity": 24,
  "retrieve_per_polarity": 3,
  "train_episodes": 6,
  "test_episodes": 30,
  "variants": [
    "full"
  ]
}
- variants: full

## Best pair per variant (train-tuned; held-out means in `results.json`)

| variant | best pair | train score | held-out score |
|---|---|---:|---:|
| full | `CPG@{ankle_len=0.35, density=6, gear=210, hip_len=0.18, leg_radius=0.07, torso_radius=0.25}` | 433.4 | 218.5 |