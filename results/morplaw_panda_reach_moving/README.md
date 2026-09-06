# MorpLaw run

- environment: `LawevoPandaReachMoving-v0`
- model: `gpt-5.4-nano`
- config: {
  "generations": 1,
  "proposals_per_side": 1,
  "responsive_per_side": 0,
  "joint_top_k": 2,
  "cem_iterations": 1,
  "cem_population": 4,
  "knowledge_capacity": 24,
  "retrieve_per_polarity": 3,
  "train_episodes": 2,
  "test_episodes": 3,
  "grammar_seeds": 3,
  "variants": [
    "no_knowledge"
  ]
}
- variants: no_knowledge

## Best pair per variant (train-tuned; held-out means in `results.json`)

| variant | best pair | train score | held-out score |
|---|---|---:|---:|
| no_knowledge | `Feedforward P@{base_height=0.333, forearm_len=0.384, goal_speed=0.05, mass_link1=2.7, mass_link2=2.73, mass_link3=2.04, mass_link4=2.08, mass_link5=3, mass_link6=1.3, mass_link7=0.2, motor_force=1, shoulder_offset=-0.0825, upper_arm_len=0.316, wrist_len=0.088}` | -1.105 | -2.014 |