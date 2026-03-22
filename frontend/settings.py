"""
Optional API URL / password without using ``st.secrets``.

Accessing ``st.secrets`` when ``secrets.toml`` is absent makes Streamlit show a
"No secrets found" notice on the page. For local dev, use env vars or an
optional TOML file read here instead.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Project root (parent of ``frontend/``)
ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=8)
def _toml_string_dict() -> dict[str, str]:
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return {}
    try:
        import tomllib

        with path.open("rb") as f:
            raw = tomllib.load(f)
    except Exception:
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float)):
            out[str(k)] = str(v)
    return out


def get_setting(key: str, default: str | None = None) -> str | None:
    """
    Resolve a setting: environment variable first, then ``.streamlit/secrets.toml``.

    Does not use ``st.secrets`` (avoids Streamlit's missing-secrets banner).
    """
    env = os.environ.get(key)
    if env is not None and env != "":
        return env
    return _toml_string_dict().get(key, default)
