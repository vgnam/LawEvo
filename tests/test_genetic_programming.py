import json
import sys

import pytest

from baseline.genetic_programming import graft, propose, separate_gains
from lawevo.pid import (
    ADAPTERS,
    CONTROLLED_VARIANT_ADAPTERS,
    PANDA_GYM_ADAPTERS,
    PANDA_VARIANT_ADAPTERS,
)
from lawevo.pid.expression import SymbolicExpression


@pytest.mark.parametrize("key", [
    "reacher", "reacher_payload", "panda_push", "panda_push_low_friction",
    "panda_reach", "panda_reach_moving_slow_v1", "panda_slide", "panda_slide_gate",
    "inverted_pendulum", "inverted_pendulum_pulse",
])
def test_gp_proposals_are_valid_novel_and_reproducible(key):
    adapter = {**ADAPTERS, **CONTROLLED_VARIANT_ADAPTERS,
               **PANDA_GYM_ADAPTERS, **PANDA_VARIANT_ADAPTERS}[key]
    ranked = [{"structure": law} for law in adapter.classical]
    excluded = {law.key() for law in adapter.classical}
    first = propose(adapter, ranked, excluded, 6, seed=1, generation=1)
    second = propose(adapter, ranked, excluded, 6, seed=1, generation=1)
    different = propose(adapter, ranked, excluded, 6, seed=2, generation=1)
    assert [law.key() for law in first] == [law.key() for law in second]
    assert [law.key() for law in first] != [law.key() for law in different]
    assert len({law.key() for law in first} - excluded) == 6
    for law in first:
        adapter.validate_structure(law)


def test_crossover_does_not_accidentally_tie_parent_gains():
    parent = SymbolicExpression("parent", "K1*x + K1*y")
    donor = SymbolicExpression("donor", "K1*x")
    child = SymbolicExpression("child", graft(parent.root, (1,), separate_gains(donor.root)))
    assert parent.parameter_count == 1
    assert child.parameter_count == 2


def test_gp_cli_artifacts_and_resume_without_llm(tmp_path, monkeypatch):
    import experiments.gymnasium_classical_benchmarks as benchmark

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "NVIDIAChatClient", lambda *a, **k: pytest.fail("GP called LLM"))
    monkeypatch.setattr(benchmark.getpass, "getpass", lambda *a: pytest.fail("GP asked for key"))
    monkeypatch.setattr(benchmark, "plot_environment", lambda *a: None)
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--environment", "InvertedPendulum-v5", "--search-method", "gp",
        "--gp-seed", "7", "--generations", "1", "--proposals", "2",
        "--cem-iterations", "0", "--train-episodes", "1", "--test-episodes", "1",
    ])
    benchmark.main()
    root = next((tmp_path / "results" / "InvertedPendulum-v5").iterdir())
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["search_method"] == "gp"
    assert manifest["search_config"]["seed"] == 7
    assert (root / "data/gp/best_controller.json").is_file()
    assert not (root / "data/lawevo").exists()
    result = json.loads((root / "data/summary/results.json").read_text())
    assert "Genetic Programming" in result["result"]["test"]
    assert result["model"] is None
    generation = json.loads((root / "data/gp/generations/generation_001.json").read_text())
    assert len(generation["evaluated_this_generation"]) == 2
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume-run", root.name])
    benchmark.main()
    assert json.loads((root / "data/summary/results.json").read_text()) == result
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--gp-seed", "8"])
    with pytest.raises(ValueError, match="incompatible"):
        benchmark.main()
