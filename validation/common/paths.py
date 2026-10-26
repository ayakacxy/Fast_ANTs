"""Stable repository locations for developer tools, independent of their group."""

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
VALIDATION = PROJECT / "validation"

# Direct invocations work before editable installation. Explicit snapshot roots
# inserted by a source-selectable tool retain priority over this default.
if str(PROJECT / "src") not in sys.path:
    sys.path.append(str(PROJECT / "src"))
