import json
import re
import sys

from lawevo.evolve.nvidia_nim import NVIDIAChatClient
from lawevo.pid import ADAPTERS


def test_timestamped_environment_artifact_layout(tmp_path, monkeypatch) -> None:
    import experiments.gymnasium_classical_benchmarks as benchmark

    def fake_complete(self, system, prompt, **kwargs):
        del self, system, kwargs
        allowed = json.loads(re.search(r"Allowed terms: (\[[^\n]+\])", prompt).group(1))
        return json.dumps([{"name": "smoke_evolved", "terms": [allowed[-1]]}])

    monkeypatch.setattr(NVIDIAChatClient, "complete", fake_complete)
    monkeypatch.setenv("NVIDIA_API_KEY", "smoke-only")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gymnasium_classical_benchmarks",
            "--suite",
            "classical",
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
    for adapter in ADAPTERS.values():
        environment = run_root / adapter.env_id
        assert (environment / "classical" / "controllers.json").is_file()
        assert (environment / "lawevo" / "best_controller.json").is_file()
        assert (environment / "lawevo" / "generations" / "generation_000.json").is_file()
        assert (environment / "lawevo" / "generations" / "generation_001.json").is_file()
        assert list((environment / "plot").glob("*_comparison.png"))
        assert list((environment / "plot").glob("*_comparison.pdf"))
        assert (environment / "summary" / "metrics_summary.csv").is_file()
        assert (environment / "summary" / "rollout_metrics.csv").is_file()
        assert (environment / "summary" / "results.json").is_file()
