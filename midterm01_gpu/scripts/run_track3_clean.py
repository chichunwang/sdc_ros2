#!/usr/bin/env python3
"""Thin entrypoint for the independent track3_clean package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track3_clean.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
