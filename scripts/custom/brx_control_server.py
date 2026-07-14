#!/usr/bin/env python3
"""Compatibility launcher for :mod:`AppleVisionPro.brx_control_server`."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AppleVisionPro.brx_control_server import main, simulation_app  # noqa: E402


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
