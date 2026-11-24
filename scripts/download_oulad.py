"""Download the OULAD archive (UCI mirror of the Kaggle dataset) into data/raw and verify hashes."""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edtech import utils as U  # noqa: E402

RAW = ROOT / "data" / "raw"
URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    print("downloading", URL)
    blob = urllib.request.urlopen(URL, timeout=300).read()
    outer = zipfile.ZipFile(io.BytesIO(blob))
    inner_name = next((n for n in outer.namelist() if n.endswith(".zip")), None)
    z = zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) if inner_name else outer
    for n in z.namelist():
        if n.endswith(".csv"):
            (RAW / Path(n).name).write_bytes(z.read(n))
    manifest = json.loads((RAW / "raw_manifest.json").read_text())
    for name, meta in manifest["files"].items():
        p = RAW / name
        if p.suffix == ".csv":
            print(f"{name:25s} {'OK' if p.exists() and U.sha256(p) == meta['sha256'] else 'MISMATCH'}")
