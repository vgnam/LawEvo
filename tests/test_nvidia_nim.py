import json

from lawevo.evolve.nvidia_nim import NVIDIAChatClient


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": "[]"}}]}
        ).encode("utf-8")


def test_complete_does_not_set_a_token_limit(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = NVIDIAChatClient("test-key")

    assert client.complete("system", "prompt", reasoning_effort="high") == "[]"
    assert "max_tokens" not in captured["body"]
    assert "max_completion_tokens" not in captured["body"]
    assert captured["body"]["reasoning_effort"] == "high"
