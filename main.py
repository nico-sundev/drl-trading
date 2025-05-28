#!/usr/bin/env python3
"""Main bootstrap script that adapts to deployment mode."""
# Added another comment for testing pre-commit hooks workflow

import sys
from pathlib import Path

# Add framework to path
framework_path = Path(__file__).parent / "drl-trading-framework" / "src"
sys.path.insert(0, str(framework_path))


def main():
    pass

if __name__ == "__main__":
    main()
