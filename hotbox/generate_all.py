#!/usr/bin/env python3
"""Regenerate the full set of Hot Box decks.

Builds nominal- and measured-density decks for the benchmark 8x4 cross section
at every stack height from 1 to 12 layers (the approach-to-critical series).
Run from the ``hotbox/`` directory:

    python generate_all.py
"""

import os

import hbgen

WIDTH, LENGTH = 8, 4
HEIGHTS = range(1, 13)
OUTDIR = "decks"
DENSITY_CSV = "densities.csv"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    os.makedirs(OUTDIR, exist_ok=True)
    densities = hbgen.read_density_csv(DENSITY_CSV)

    for height in HEIGHTS:
        for mode in ("nominal", "measured"):
            text = hbgen.generate_deck(
                WIDTH, LENGTH, height,
                mode=mode,
                densities=densities if mode == "measured" else None,
            )
            path = os.path.join(OUTDIR, f"hb_{mode}_{height}_{WIDTH}_{LENGTH}.i")
            with open(path, "w") as fh:
                fh.write(text)
            print(f"wrote {path:42s} ({text.count(chr(10)):5d} lines)")


if __name__ == "__main__":
    main()
