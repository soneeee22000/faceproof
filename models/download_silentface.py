"""Download the Silent-Face anti-spoofing weights into this directory.

The two MiniFASNet weight files are Apache-2.0 licensed (see the repository-root
``NOTICE``). They are small (~2 MB each) and gitignored — never committed.

Usage (from the repository root):
    python models/download_silentface.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

_BASE_URL = (
    "https://raw.githubusercontent.com/minivision-ai/"
    "Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models"
)
_WEIGHT_FILES = ("2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth")
_TARGET_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Download each weight file that is not already present."""
    for filename in _WEIGHT_FILES:
        destination = _TARGET_DIR / filename
        if destination.exists():
            print(f"already present: {filename}")
            continue
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(f"{_BASE_URL}/{filename}", destination)
        print(f"  saved to {destination}")


if __name__ == "__main__":
    main()
