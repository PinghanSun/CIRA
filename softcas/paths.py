"""Locate bundled configuration templates."""

from pathlib import Path


def configuration_root():
    package = Path(__file__).resolve().parent
    for path in (package.parent / "configs", package / "_configs"):
        if path.is_dir():
            return path
    raise FileNotFoundError("Configuration templates are unavailable")
