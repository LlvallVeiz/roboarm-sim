#!/usr/bin/env python3
"""Entry point: run the roboarm CLI, e.g. ``python3 main.py fk --angles 0 0 0``."""

import sys

from roboarm.cli import main

if __name__ == "__main__":
    sys.exit(main())
