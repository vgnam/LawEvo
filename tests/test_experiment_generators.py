"""Exercise generator configuration without contacting an API provider."""

from types import SimpleNamespace

import pytest

from experiments import evolve_morphlaw as experiment


@pytest.mark.parametrize("side", ["law", "morphology"])
def test_generators_forward_reasoning_effort(monkeypatch, side):
    class RequestCaptured(Exception):
        pass

    def complete(system, prompt, **kwargs):
        assert kwargs["reasoning_effort"] == "high"
        raise RequestCaptured

    monkeypatch.setattr(experiment, f"{side}_knowledge_query", lambda *args: "query")
    monkeypatch.setattr(experiment, f"{side}_mutation_prompt", lambda *args, **kwargs: "prompt")
    factory = experiment.make_law_generator if side == "law" else experiment.make_morph_generator
    generator = factory(
        SimpleNamespace(complete=complete), "reacher", None, {}, [], "full",
        reasoning_effort="high",
    )
    knowledge = SimpleNamespace(retrieve=lambda *args, **kwargs: [])
    with pytest.raises(RequestCaptured):
        generator(None, knowledge, 1, 0, None, False)
