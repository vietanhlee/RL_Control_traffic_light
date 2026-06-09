from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _candidate_env_files() -> Iterable[Path]:
    cwd_env = Path.cwd() / ".env"
    project_env = Path(__file__).resolve().parents[1] / ".env"
    seen: set[Path] = set()
    for candidate in (cwd_env, project_env):
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield candidate


def load_dotenv() -> None:
    """Load a local .env file without overriding existing environment variables."""
    for env_file in _candidate_env_files():
        if not env_file.exists():
            continue

        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value

