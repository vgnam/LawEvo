from lawevo.evolve import nvidia_nim
from lawevo.evolve.nvidia_nim import env_setting, load_env_file, resolve_endpoint


def test_env_setting_returns_first_non_empty(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    assert env_setting("OPENAI_API_KEY", "NVIDIA_API_KEY", default="fallback") == "fallback"

    monkeypatch.setenv("NVIDIA_API_KEY", "nim-key")
    assert env_setting("OPENAI_API_KEY", "NVIDIA_API_KEY", default="fallback") == "nim-key"

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert env_setting("OPENAI_API_KEY", "NVIDIA_API_KEY", default="fallback") == "openai-key"

    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert env_setting("OPENAI_API_KEY", "NVIDIA_API_KEY", default="fallback") == "nim-key"


def test_resolve_endpoint_accepts_root_and_full_url():
    assert resolve_endpoint("https://api.example.com/v1") == "https://api.example.com/v1/chat/completions"
    assert resolve_endpoint("https://api.example.com/v1/") == "https://api.example.com/v1/chat/completions"
    assert (
        resolve_endpoint("https://integrate.api.nvidia.com/v1/chat/completions")
        == "https://integrate.api.nvidia.com/v1/chat/completions"
    )


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nOPENAI_MODEL='from-file'\nOPENAI_API_KEY=\"file-key\"\nBAD_LINE\n\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_MODEL", "from-shell")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    load_env_file()

    assert nvidia_nim.os.environ["OPENAI_MODEL"] == "from-shell"
    assert nvidia_nim.os.environ["OPENAI_API_KEY"] == "file-key"
