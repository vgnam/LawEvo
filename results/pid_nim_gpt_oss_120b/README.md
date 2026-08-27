# NVIDIA NIM GPT-OSS PID Evolution

## Protocol

- Model: `openai/gpt-oss-120b` through NVIDIA NIM.
- Six LLM calls/generations, ten valid gain proposals per call.
- Genome: `(kp_distance, ki_distance, kd_distance, kp_heading, ki_heading, kd_heading)`.
- Training: 16 deterministic unicycle goal-tracking scenarios (seed `20260826`).
- Evaluation: 30 disjoint held-out scenarios (seed `20260917`).
- Baselines: P, PI, PD, manually tuned PID, and grid-tuned PID (324 configurations).
- Every candidate was scored by the same local simulator. The LLM did not produce or
  estimate fitness values.

Objective:

```text
120*success - 10*final_distance - 1.2*settling_time
- 0.08*energy - 0.0003*jerk - 2*mean_heading_error
```

## Selected NIM controller

```text
distance PID: Kp=1.95, Ki=0.39, Kd=0.21
heading PID:  Kp=4.80, Ki=0.13, Kd=0.66
```

## Held-out results

| Controller | Score ↑ | Success | Final distance ↓ | Settling time ↓ | Energy ↓ | Jerk ↓ |
|---|---:|---:|---:|---:|---:|---:|
| P | 112.639 | 100% | 0.1096 | 4.295 s | 8.186 | 10.583 |
| PI | 113.362 | 100% | 0.0907 | 3.768 s | 8.252 | 9.082 |
| PD | 112.544 | 100% | 0.1097 | 4.378 s | 8.150 | 11.328 |
| Manual PID | 113.178 | 100% | 0.0961 | 3.902 s | 8.186 | 9.913 |
| Grid PID | 114.027 | 100% | 0.0752 | 3.307 s | 8.956 | 14.119 |
| **NIM GPT-OSS PID** | **114.602** | **100%** | **0.0275** | **3.148 s** | 9.355 | 31.943 |

Relative to grid-tuned PID, NIM PID improved objective score by 0.51%, reduced final
distance by 63.40%, and reduced settling time by 4.79%. It increased energy by 4.45% and
jerk by 126.24%. Paired two-sided Wilcoxon tests over the 30 scenarios gave `p<0.001` for
score, final distance, settling time, and energy.

## Interpretation limits

This is one deterministic train/test split and one LLM evolution run. The paired scenario
tests quantify differences for this controller pair on this held-out set; they do not
measure variation across LLM evolution seeds. A paper-level claim should repeat the full
six-generation evolution with at least five independent LLM seeds and report mean ± SD
across runs. The current objective puts a small weight on jerk, so the selected controller
accepts much higher jerk in exchange for accuracy and speed. A jerk-constrained or
multi-objective follow-up is warranted.
