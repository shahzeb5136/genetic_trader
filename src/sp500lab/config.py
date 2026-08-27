"""Settings loader: config/settings.toml + .env overrides.

Precedence (highest first): environment variable, .env file, settings.toml, default.
No third-party dependency - tomllib is stdlib on 3.11+ and .env parsing is trivial.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .paths import CONFIG_DIR, PROJECT_ROOT


def _load_dotenv() -> None:
    """Load .env into os.environ without clobbering real environment variables."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


@dataclass(frozen=True)
class Settings:
    user_agent: str
    history_start: str
    price_start: str
    rate_limits: dict[str, float]
    max_retries: int
    backoff_base_seconds: float
    timeout_seconds: int
    compression: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def rate_limit_for(self, host: str) -> float:
        """Requests/second allowed for a host, falling back to the default."""
        return float(self.rate_limits.get(host, self.rate_limits.get("_default", 2.0)))

    def secret(self, name: str) -> str | None:
        val = os.environ.get(name, "").strip()
        return val or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    with open(CONFIG_DIR / "settings.toml", "rb") as fh:
        cfg = tomllib.load(fh)

    return Settings(
        user_agent=os.environ.get("SP500LAB_USER_AGENT")
        or cfg["identity"]["user_agent"],
        history_start=cfg["universe"]["history_start"],
        price_start=cfg["universe"]["price_start"],
        rate_limits={k: float(v) for k, v in cfg["rate_limits"].items()},
        max_retries=int(cfg["http"]["max_retries"]),
        backoff_base_seconds=float(cfg["http"]["backoff_base_seconds"]),
        timeout_seconds=int(cfg["http"]["timeout_seconds"]),
        compression=cfg["storage"]["compression"],
        raw=cfg,
    )
