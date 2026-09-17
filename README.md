cd d:\lawevo

# Task 1 — Reach
py -m experiments.gymnasium_classical_benchmarks --environment PandaReachDense-v3 --generations 20 --train-episodes 6 --test-episodes 30


cd d:\lawevo

# Task 2 — Push
py -m experiments.gymnasium_classical_benchmarks --environment PandaPushDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

cd d:\lawevo

# Task 3 — Slide
py -m experiments.gymnasium_classical_benchmarks --environment PandaSlideDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

cd d:\lawevo

# Task 4 — Pick and Place
py -m experiments.gymnasium_classical_benchmarks --environment PandaPickAndPlaceDense-v3 --generations 20 --train-episodes 6 --test-episodes 30

cd d:\lawevo

# Task 5 — Stack
py -m experiments.gymnasium_classical_benchmarks --environment PandaStackDense-v3 --generations 20 --train-episodes 6 --test-episodes 30


cd d:\lawevo

# Task 6 — Reach-MovingGoal
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaReachMoving-v0 --generations 20 --train-episodes 6 --test-episodes 30

cd d:\lawevo

# Task 7 — Push-IceObstacle
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaPushIce-v0 --generations 20 --train-episodes 6 --test-episodes 30

cd d:\lawevo

# Task 8 — Slide-Gate
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaSlideGate-v0 --generations 20 --train-episodes 6 --test-episodes 30

cd d:\lawevo


# Task 9 — Pick-HeavyDistractor
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaPickDistractor-v0 --generations 20 --train-episodes 6 --test-episodes 30

cd d:\lawevo

# Task 10 — Stack-NarrowSettle
py -m experiments.gymnasium_classical_benchmarks --environment LawevoPandaStackNarrow-v0 --generations 20 --train-episodes 6 --test-episodes 30
