"""Write model artifacts so a reader never sees a half-written file.

Training writes files that the dashboard may be reading at the same moment.
Writing straight to the final path means a refresh that lands mid-write loads
a truncated file. Instead each file is written next to its destination under
a temporary name and then swapped in with os.replace, which is atomic on the
same filesystem: a reader sees either the old file or the new one, never half.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def _temp_path(path: Path) -> Path:
    # Keep the real extension last: Keras and joblib pick formats by suffix.
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def atomic_save(path: Path, write: Callable[[Path], object]) -> None:
    """Call ``write(temp_path)``, then move the result onto ``path``."""
    path = Path(path)
    temp = _temp_path(path)
    try:
        write(temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    atomic_save(path, lambda temp: temp.write_text(text, encoding="utf-8"))
