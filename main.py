#!/usr/bin/env python3
"""Command-line entry point for the YarnBreak detector.

The pipeline itself lives in yarnbreak/pipeline.py; this file only keeps the
historical `python3 main.py ...` command working (scripts/chay.sh,
scripts/chay_nen.sh and the operator guide use it).

    python3 main.py --config config.yaml
    python3 main.py --config config.yaml --headless
"""
from yarnbreak.pipeline import main

if __name__ == "__main__":
    main()
