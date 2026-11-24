"""Convert notebooks/_src/*.py (jupytext percent format) to .ipynb and execute them in order.

Usage:  python scripts/run_notebooks.py            # all notebooks
        python scripts/run_notebooks.py 03 04      # only notebooks whose name starts with 03 / 04
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"


def run(path: Path) -> None:
    nb = jupytext.read(path)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    t0 = time.time()
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(OUT)}})
    try:
        client.execute()
    finally:
        target = OUT / (path.stem + ".ipynb")
        nbformat.write(nb, target)
    print(f"OK  {path.stem}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for p in sorted(SRC.glob("*.py")):
        if not prefixes or any(p.stem.startswith(x) for x in prefixes):
            run(p)
