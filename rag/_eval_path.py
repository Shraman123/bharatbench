"""
Same convention as service/_eval_path.py: eval/ is not an installed package,
its modules are meant to be imported with eval/ on sys.path directly.
Importing this once, before anything else in rag/ imports config/providers/
runner, makes that work from here too.
"""

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).parent.parent / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
