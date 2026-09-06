"""Audit a saved ten-task controller without CEM or LLM calls."""
import argparse
import json
from pathlib import Path

import numpy as np

from lawevo.pid import ADAPTERS, CONTROLLED_VARIANT_ADAPTERS, PANDA_GYM_ADAPTERS
from lawevo.pid.gym_benchmark import GymStructure, evaluate_gym_structure
from lawevo.verify.controller import SUPPORTED_ENVIRONMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=sorted(SUPPORTED_ENVIRONMENTS))
    parser.add_argument("--controller", required=True, type=Path,
                        help="selected_controller.json or best_controller.json")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    if args.controller.resolve() == args.output.resolve():
        parser.error("output must not overwrite the controller")
    adapters = {a.env_id: a for a in {**ADAPTERS, **CONTROLLED_VARIANT_ADAPTERS, **PANDA_GYM_ADAPTERS}.values()}
    record = json.loads(args.controller.read_text(encoding="utf-8"))
    law = GymStructure.from_dict(record["structure"])
    gains = np.asarray(record["gains"], dtype=float)
    seeds = list(range(args.seed, args.seed + args.episodes))
    metrics, episodes = evaluate_gym_structure(adapters[args.environment], law, gains, seeds,
                                               verify_barriers=True)
    reports = [e.barrier_verification for e in episodes]
    passed = sum(r["status"] == "passed_sampled_checks" for r in reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "environment": args.environment, "structure": law.to_dict(), "gains": gains.tolist(),
        "metrics": metrics.to_dict(), "sampled_pass_count": passed,
        "certified_between_samples": False, "episodes": reports,
    }, indent=2), encoding="utf-8")
    print(f"sampled_barrier_pass={passed}/{len(reports)} sr={metrics.success_rate:.4f} "
          f"sg={metrics.sg} certified=False report={args.output}")


if __name__ == "__main__":
    main()
