# MorphLaw

Knowledge-guided co-evolution of robot morphologies and symbolic feedback laws.
The search alternates body and controller proposals, tunes controller gains with
CEM, and transfers directed knowledge between the two search spaces.

## Development checkout

Develop on branch `morphlaw-dev` in `D:\MorphLaw`. This is a separate Git worktree,
created from commit `45122ea`. The original checkout `D:\LawEvo` stays on
`free-form-symbolic-evolution`, including its uncommitted work and active runs.
Uncommitted changes in that checkout were not imported into this branch.

Open `D:\MorphLaw` as a separate editor project. Do not switch branches, clean,
reset, or install packages into the original checkout while its jobs are running.
The worktrees share Git history and remotes, but have separate files and indexes.
CPU, RAM, GPU and API quotas remain shared: defer expensive experiments while
existing runs need those resources.

## Setup (PowerShell)

The current `D:\MorphLaw\.venv` is prepared independently of the original
environment. Its PyBullet 3.2.7 runtime was copied into this environment because
the configured package index did not supply a matching Windows wheel. No shared
site-packages link is used. A fresh Windows setup may need a compatible PyBullet
wheel or C++ build tools before installing the `benchmarks` extra.
NumPy is constrained below 2 for compatibility with the PyBullet binary ABI.

Create a dedicated environment; always install and run through its Python:

```powershell
cd D:\MorphLaw
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,benchmarks]"
Copy-Item .env.example .env
```

Fill in `.env` for the API provider you use. Credentials are ignored by Git and
were not copied from the original project. Simulation tests do not need API keys.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m experiments.evolve_morphlaw --help
.\.venv\Scripts\python.exe -m experiments.evolve_morphlaw --environment reacher --variants full
```

The installed `morphlaw` command exposes the same CLI. The `benchmarks` extra
supplies plotting and Panda-Gym support; MuJoCo is included in the base package.
Run commands from this checkout so `.env` and relative output paths are local.
New results default to `results/morphlaw_<environment>`; `--output` overrides it
and `--resume` reuses an existing run cache. Give each concurrent run a distinct
output directory. Rendered MuJoCo XML uses a separate `morphlaw_xml`
temporary cache; generated Panda URDFs stay in this package's asset directory.

## Scope

- `morphlaw/`: morphology templates, robot grammar, co-evolution engine, knowledge,
  proposals and prompts; import `MorphLawConfig` and `MorphLawRunner` from here.
- `morphlaw/control/`: shared symbolic expressions, CEM, metrics and simulation
  adapters required by the morphology experiments.
- `morphlaw/llm.py` and `morphlaw/belief.py`: API client and search memory.
- `experiments/evolve_morphlaw.py`: the morphology/controller experiment CLI.
- `tests/`: morphology, symbolic control, simulator and API-client checks.

Supported families include Reacher (payload, gravity, precision), Pusher,
Walker2d, Hopper, HalfCheetah, Swimmer, Ant, variable topology, RoboMorph terrain
grammars and parameterized Panda arms. Use `--help` for exact environment keys.

Standalone controller-only evolution, unicycle/CBF pipelines, unrelated
ManiSkill/Genesis/Robosuite tasks and their scripts, tests and results were removed.
The Python namespace is now `morphlaw`; custom environment IDs use `Morphlaw`.

Historical morphology results remain under `results/morplaw_*` exactly as recorded,
including their original identifiers and commands. They are reference artifacts;
use the new CLI and a new output directory for future runs. Old caches are not
promised to resume across the namespace and environment-ID migration.
