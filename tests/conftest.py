"""Configuration pytest partagee : rend le package src/structrisk importable
sans installation prealable (pas de dependance a un `pip install -e .`)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
