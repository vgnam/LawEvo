from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NVIDIA_MODEL = "openai/gpt-oss-120b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE settings without adding a runtime dependency."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def env_setting(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among *names*."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value
    return default


def resolve_endpoint(base_url: str) -> str:
    """Accept both an API root (``.../v1``) and a full chat-completions URL."""
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class NIMError(RuntimeError):
    pass


@dataclass(frozen=True)
class NVIDIAChatClient:
    api_key: str
    model: str = DEFAULT_NVIDIA_MODEL
    endpoint: str = DEFAULT_NVIDIA_BASE_URL
    timeout: float = 600.0

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.7,
        reasoning_effort: str = "medium",
        retries: int = 3,
    ) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "LawEvo/0.1",
            },
        )
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return str(payload["choices"][0]["message"]["content"])
            except (
                TimeoutError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        raise NIMError(f"NVIDIA NIM request failed after {retries} attempts: {last_error}")
