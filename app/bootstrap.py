# app/bootstrap.py
from pathlib import Path
import sys

# Ensure top-level package (luminai) is importable from any page
ROOT = Path(__file__).resolve().parents[1]  # .../LuminAI
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
