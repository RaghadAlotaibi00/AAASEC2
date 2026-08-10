"""Lightweight local dotenv shim used for the lab.

This provides a minimal `load_dotenv()` implementation so the
repository can run even when the external `python-dotenv` package
isn't installed. If `python-dotenv` is installed, it will shadow
this module automatically, so this shim is safe.

It supports simple KEY=VALUE lines and ignores comments and blanks.
"""
import os
from pathlib import Path
from typing import Optional


def _parse_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip()
    if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in ('"', "'")):
        val = val[1:-1]
    return key, val


def load_dotenv(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """Load environment variables from a .env file.

    Returns True if a .env file was found and parsed.
    """
    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.exists():
        return False

    changed = False
    with dotenv_path.open("r", encoding="utf8") as f:
        for raw in f:
            parsed = _parse_line(raw)
            if not parsed:
                continue
            k, v = parsed
            if override or k not in os.environ:
                os.environ[k] = v
                changed = True

    return changed


__all__ = ["load_dotenv"]
