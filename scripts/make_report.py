#!/usr/bin/env python
from pathlib import Path

report = Path("results/quick_v1/REPORT.md")
if not report.exists():
    raise SystemExit("Run scripts/run_quick_suite.py first.")
print(report.resolve())
