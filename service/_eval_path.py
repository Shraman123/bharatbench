"""
eval/ is not an installed package -- runner.py, config.py, etc. are top-level
modules meant to be imported with eval/ on sys.path directly (see
tests/conftest.py, which does the same thing). Importing this module once,
before anything else in service/ imports runner/config/analyze, makes that
work from here too.
"""

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).parent.parent / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
