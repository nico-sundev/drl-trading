"""
Root conftest for drl-trading-preprocess tests.

This file configures pytest for the entire test suite by adding the tests
directory to Python path, allowing test-specific packages (like 'builders')
to be imported cleanly without sys.path manipulation in individual test files.

This is the standard Python/pytest pattern for handling test utilities.
"""
import sys
from pathlib import Path

# Add tests directory to Python path so test packages can be imported
# This runs once when pytest starts, before any test collection
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))
