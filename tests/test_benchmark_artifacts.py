import json
import re
import sys

from lawevo.evolve.nvidia_nim import NIMError, NVIDIAChatClient


def test_prompt_assigns_all_eoh_variation_operators() -> None:
    import experiments.gymnasium_classical_benchmarks as benchmark

    text = benchmark.prompt(
        "pendulum",
        ("angle", "angular_velocity", "tanh_angle"),
        [],
        [],
        count=5,
        generation=1,
    )

    for operator in ("E1", "E2", "M1", "M2", "M3"):
        assert f'"operator": "{operator}"' in text
    assert "never the numeric gains" in text
    assert "Prefix every candidate name with its operator code" in text
    assert "Exact scalar fitness is the environment return" in text


def test_timestamped_environment_artifact_layout(tmp_path, monkeypatch) -> None:
    import experiments.gymnasium_classical_benchmarks as benchmark

    def fake_complete(self, system, prompt, **kwargs):
        del self, system, kwargs
        allowed = json.loads(re.search(r"Allowed signals: (\[[^\n]+\])", prompt).group(1))
        return json.dumps([{"name": "smoke_evolved", "expression": f"K1*{allowed[-1]}"}])

    monkeypatch.setattr(NVIDIAChatClient, "complete", fake_complete)
    monkeypatch.setenv("NVIDIA_API_KEY", "smoke-only")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gymnasium_classical_benchmarks",
            "--environment",
            "InvertedPendulum-v5",
            "--generations",
            "1",
            "--proposals",
            "1",
            "--proposal-attempts",
            "1",
            "--cem-iterations",
            "0",
            "--cem-population",
            "2",
            "--train-episodes",
            "1",
            "--test-episodes",
            "2",
        ],
    )

    benchmark.main()

    run_roots = list((tmp_path / "results").glob("????????_??????"))
    assert len(run_roots) == 1
    run_root = run_roots[0]
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["requested_environment"] == "InvertedPendulum-v5"
    environment = run_root / "InvertedPendulum-v5"
    assert (environment / "classical" / "controllers.json").is_file()
    assert (environment / "lawevo" / "best_controller.json").is_file()
    assert (environment / "lawevo" / "generations" / "generation_000.json").is_file()
    assert (environment / "lawevo" / "generations" / "generation_001.json").is_file()
    assert list((environment / "plot").glob("*_comparison.png"))
    assert list((environment / "plot").glob("*_comparison.pdf"))
    assert (environment / "summary" / "metrics_summary.csv").is_file()
    assert (environment / "summary" / "rollout_metrics.csv").is_file()
    assert (environment / "summary" / "results.json").is_file()


def test_llm_failure_falls_back_without_stopping_run(tmp_path, monkeypatch) -> None:
    import experiments.gymnasium_classical_benchmarks as benchmark

    def fail_complete(self, system, prompt, **kwargs):
        del self, system, prompt, kwargs
        raise NIMError("temporary timeout")

    monkeypatch.setattr(NVIDIAChatClient, "complete", fail_complete)
    monkeypatch.setattr(benchmark.time, "sleep", lambda delay: None)
    monkeypatch.setattr(benchmark, "plot_environment", lambda *args: None)
    monkeypatch.setenv("NVIDIA_API_KEY", "smoke-only")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gymnasium_classical_benchmarks",
            "--environment",
            "InvertedPendulum-v5",
            "--generations",
            "1",
            "--proposals",
            "1",
            "--proposal-attempts",
            "1",
            "--cem-iterations",
            "0",
            "--cem-population",
            "2",
            "--train-episodes",
            "1",
            "--test-episodes",
            "2",
        ],
    )

    benchmark.main()

    run_root = next((tmp_path / "results").glob("????????_??????"))
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((run_root / "state" / "generation_plans.json").read_text())
    assert manifest["status"] == "complete"
    assert plan["inverted_pendulum"]["1"][0]["name"] == "fallback_mutation_1"
