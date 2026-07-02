"""Shared pytest setup: put the repo root on sys.path so `src.rag` imports work
without installing the package. Loaded by pytest before any test module."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
