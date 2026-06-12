#!/usr/bin/env python3
"""Hot Box critical-assembly MCNP deck generator.

A standalone rewrite of the original ``HBMCNPgen.py`` script.  Instead of
emitting one ``LIKE n BUT TRCL=(...)`` card for every foil, cladding and
graphite cell (which produced ~6000-card decks), this module builds the
assembly with **MCNP repeated structures**: each unit cell is described once
as a *universe*, and the whole stack is placed with a single rectangular
``LAT=1`` lattice driven by a ``FILL`` array.

Two density treatments are supported:

* ``mode="measured"`` -- preserve the individually measured foil/cladding
  densities (one fuel universe per fuelled slot).  This is the high-fidelity
  benchmark model.
* ``mode="nominal"``  -- use a single average density for every foil and a
  single average cladding density (one fuel universe total).  This collapses
  the whole assembly to a handful of cells and is the "simple" model.

The assembly geometry itself is identical in both modes.

Geometry summary (one lattice element, foil plane at local z = 0)
-----------------------------------------------------------------
* X pitch  PX = 30.45968 cm (12")  -- columns  (index i)
* Y pitch  PY = 60.91936 cm (24")  -- rows      (index j)
* Z pitch  PZ =  7.62000 cm ( 3")  -- layers    (index k)

Each element contains, bottom to top: a 1" graphite block, the fuel-element
plane (two U-235 foils inside a thin stainless-steel sheath), and a 2"
graphite block, flanked by graphite side rails.  Fuelled slots follow a
checkerboard that flips parity every layer:

    fuelled  <=>  (layer + col + row) is even        (1-based indices)
             <=>  (i + j + k) is odd                  (0-based indices)

Simplifications vs. the original deck (all neutronically negligible, but worth
a one-time k-eff confirmation against the reference deck):

* The folded stainless-steel cladding (40+ ``ARB`` polyhedra in the original)
  is modelled as two thin SS sheets above/below the foils, preserving the SS
  *mass* rather than the exact fold geometry.
* The graphite side-rail bolt holes (24 ``RCC`` cylinders of air per block)
  are omitted -- small air voids in graphite, far from the fuel.

Usage
-----
    # Library
    from hbgen import generate_deck, parse_deck_densities
    text = generate_deck(8, 4, 12, mode="nominal")

    # CLI -- regenerate both reference decks from the measured-density table
    python hbgen.py --densities densities.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Physical constants (cm, g/cc).  All taken from the reference Hot Box deck.
# --------------------------------------------------------------------------
PX = 30.45968   # column pitch  (X, 12")
PY = 60.91936   # row pitch     (Y, 24")
PZ = 7.62000    # layer pitch   (Z,  3")

RHO_GRAPHITE = 2.7338     # graphite block density (from reference deck)
RHO_FOIL_NOMINAL = 18.5835  # mean of the 382 measured foil densities
RHO_CLAD_NOMINAL = 7.2694   # mean of the 191 measured cladding densities

KSRC_FOIL_OFFSET = 5.55625  # +/- x offset of the two foils about a slot centre

# Material numbers
M_RFOIL, M_LFOIL, M_CLAD, M_GRAPH = 1, 2, 3, 4

# Universe numbers
U_LATTICE = 1
U_VOID = 2
U_FUEL_NOMINAL = 3
U_FUEL_MEASURED_BASE = 100  # measured fuel universes are 100, 101, 102, ...


@dataclass
class FoilDensities:
    """Measured densities for one fuelled slot."""
    rho_rf: float    # right foil
    rho_lf: float    # left foil
    rho_clad: float  # stainless-steel cladding


# --------------------------------------------------------------------------
# Checkerboard occupancy
# --------------------------------------------------------------------------
def is_fuelled(i: int, j: int, k: int) -> bool:
    """Return True if lattice slot (i, j, k) holds a real fuel element.

    0-based indices.  Equivalent to (layer + col + row) being even in the
    1-based convention used by the original benchmark deck.
    """
    return (i + j + k) % 2 == 1


# --------------------------------------------------------------------------
# Density-table parsing
# --------------------------------------------------------------------------
_DENSITY_RE = re.compile(
    r"mat=(\d+)\s+rho=-([\d.]+).*?\$\s*(right foil|left foil|cladding)\s+"
    r"(\d+)_(\d+)_(\d+)"
)


def parse_deck_densities(deck_path: str) -> dict[tuple[int, int, int], FoilDensities]:
    """Extract measured densities from an original auto-generated Hot Box deck.

    Reads the ``... mat=N rho=-X $right foil L_C_R`` comment lines and returns
    a mapping ``(layer, col, row) -> FoilDensities`` for every fuelled slot.
    """
    rf: dict = {}
    lf: dict = {}
    cl: dict = {}
    with open(deck_path) as fh:
        for line in fh:
            m = _DENSITY_RE.search(line)
            if not m:
                continue
            mat = int(m.group(1))
            if mat == 0:  # void placeholder slot -- no real material
                continue
            rho = float(m.group(2))
            kind = m.group(3)
            key = (int(m.group(4)), int(m.group(5)), int(m.group(6)))
            if kind == "right foil":
                rf[key] = rho
            elif kind == "left foil":
                lf[key] = rho
            else:
                cl[key] = rho

    table: dict[tuple[int, int, int], FoilDensities] = {}
    for key in rf:
        table[key] = FoilDensities(
            rho_rf=rf[key],
            rho_lf=lf.get(key, RHO_FOIL_NOMINAL),
            rho_clad=cl.get(key, RHO_CLAD_NOMINAL),
        )
    return table


def write_density_csv(table: dict, path: str) -> None:
    """Write a measured-density table to CSV (layer, col, row, rf, lf, clad)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "col", "row", "rho_rf", "rho_lf", "rho_clad"])
        for (layer, col, row), d in sorted(table.items()):
            w.writerow([layer, col, row, d.rho_rf, d.rho_lf, d.rho_clad])


def read_density_csv(path: str) -> dict[tuple[int, int, int], FoilDensities]:
    """Read a measured-density table written by :func:`write_density_csv`."""
    table: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["layer"]), int(r["col"]), int(r["row"]))
            table[key] = FoilDensities(
                rho_rf=float(r["rho_rf"]),
                rho_lf=float(r["rho_lf"]),
                rho_clad=float(r["rho_clad"]),
            )
    return table


# --------------------------------------------------------------------------
# Surface block (shared by every universe -- defined exactly once)
# --------------------------------------------------------------------------
def _surface_cards(width: int, length: int, height: int) -> list[str]:
    cx = -15.22984 + width * PX     # container x-max
    cy = -30.45968 + length * PY    # container y-max
    cz = -2.44348 + height * PZ     # container z-max
    return [
        "C  Surface cards",
        "C  --- fuel element (local coords, foil plane at z = 0) ---",
        "10 rpp   0.23749  11.35126  -29.1211  29.1211  -0.00254  0.00254  $ right U foil",
        "11 rpp -11.35126  -0.23749  -29.1211  29.1211  -0.00254  0.00254  $ left U foil",
        "12 rpp -11.7475   11.7475  -30.45968 30.45968  -0.00762  0.00762  $ clad envelope",
        "13 rpp -11.7475   11.7475  -30.45968 30.45968  -0.00762 -0.00254  $ bottom clad sheet",
        "14 rpp -11.7475   11.7475  -30.45968 30.45968   0.00254  0.00762  $ top clad sheet",
        "15 rpp -11.7475   11.7475  -30.45968 30.45968  -0.00254  0.00254  $ foil mid-plane",
        "C  --- graphite ---",
        "20 rpp -12.065    12.065   -30.45968 30.45968  -2.44348 -0.00762  $ 1in block (below)",
        "21 rpp -12.065    12.065   -30.45968 30.45968   0.09652  5.08762  $ 2in block (above)",
        "22 rpp -12.065    12.065   -30.45968 30.45968  -0.00762  0.09652  $ mid slab footprint",
        "23 rpp -12.065    12.065   -30.45968 30.45968   5.08762  5.17652  $ top gap",
        "30 rpp -15.22984 -12.065   -30.45968 30.45968  -2.44348  5.17652  $ left rail",
        "31 rpp  12.065    15.22984 -30.45968 30.45968  -2.44348  5.17652  $ right rail",
        "C  --- lattice element, container, world ---",
        "99 rpp -15.22984  15.22984 -30.45968 30.45968  -2.44348  5.17652  $ lattice element",
        f"98 rpp -15.22984 {cx:.5f} -30.45968 {cy:.5f} -2.44348 {cz:.5f}  $ lattice container",
        "999 rpp -500 500 -500 600 -500 600  $ world",
    ]


# --------------------------------------------------------------------------
# Universe cell blocks
# --------------------------------------------------------------------------
def _void_universe_cells(start: int) -> list[str]:
    """Graphite-only placeholder universe (no foils, no cladding)."""
    n = start
    g = f"{M_GRAPH} -{RHO_GRAPHITE}"
    return [
        "C  === void (placeholder) universe ===",
        f"{n+0} {g}  -20  u={U_VOID} imp:n=1  $ 1in graphite",
        f"{n+1} {g}  -21  u={U_VOID} imp:n=1  $ 2in graphite",
        f"{n+2} {g}  -30  u={U_VOID} imp:n=1  $ left rail",
        f"{n+3} {g}  -31  u={U_VOID} imp:n=1  $ right rail",
        f"{n+4} 0       -22  u={U_VOID} imp:n=1  $ void slab",
        f"{n+5} 0       -23  u={U_VOID} imp:n=1  $ void top gap",
    ]


def _fuel_universe_cells(start: int, universe: int, d: FoilDensities,
                         label: str = "") -> list[str]:
    """One fuel universe: U-235 foils + SS sheath embedded in graphite."""
    n = start
    g = f"{M_GRAPH} -{RHO_GRAPHITE}"
    tag = f"  $ fuel universe {universe}{(' ' + label) if label else ''}"
    return [
        f"C  === fuel universe {universe}{(' ' + label) if label else ''} ===",
        f"{n+0} {M_RFOIL} -{d.rho_rf}  -10        u={universe} imp:n=1  $ right foil",
        f"{n+1} {M_LFOIL} -{d.rho_lf}  -11        u={universe} imp:n=1  $ left foil",
        f"{n+2} {M_CLAD} -{d.rho_clad}  -13        u={universe} imp:n=1  $ bottom clad",
        f"{n+3} {M_CLAD} -{d.rho_clad}  -14        u={universe} imp:n=1  $ top clad",
        f"{n+4} 0             -15 10 11  u={universe} imp:n=1  $ foil-plane void",
        f"{n+5} 0             -22 12     u={universe} imp:n=1  $ void around clad",
        f"{n+6} {g}  -20        u={universe} imp:n=1  $ 1in graphite",
        f"{n+7} {g}  -21        u={universe} imp:n=1  $ 2in graphite",
        f"{n+8} {g}  -30        u={universe} imp:n=1  $ left rail",
        f"{n+9} {g}  -31        u={universe} imp:n=1  $ right rail",
        f"{n+10} 0            -23        u={universe} imp:n=1  $ top gap",
    ]


# --------------------------------------------------------------------------
# Deck assembly
# --------------------------------------------------------------------------
def _fill_array(width: int, length: int, height: int,
                slot_universe) -> list[str]:
    """Build the FILL array lines (i fastest, then j, then k)."""
    lines = [
        f"1 0 -99 lat=1 u={U_LATTICE} imp:n=1",
        f"      fill=0:{width-1} 0:{length-1} 0:{height-1}",
    ]
    for k in range(height):
        for j in range(length):
            row = " ".join(f"{slot_universe(i, j, k):>4d}" for i in range(width))
            lines.append(f"      {row}  $ row j={j} layer k={k}")
    return lines


def _material_cards() -> list[str]:
    return [
        "C  Materials cards",
        "m1   92234.80c 5.93E-04   92235.80c 4.57E-02   $ U for right foil",
        "     92236.80c 3.43E-03   92238.80c 2.94E-03",
        "m2   92234.80c 5.87E-04   92235.80c 4.53E-02   $ U for left foil",
        "     92236.80c 3.40E-03   92238.80c 2.91E-03",
        "m3   6000.80c  3.11E-03   25055.80c 1.70E-03   $ stainless-steel cladding",
        "     14028.80c 1.54E-03   14029.80c 7.54E-05   14030.80c 4.81E-05",
        "     24050.80c 7.31E-04   24052.80c 1.35E-02   24053.80c 1.51E-03",
        "     24054.80c 3.68E-04   28058.80c 6.03E-03   28060.80c 2.25E-03",
        "     28061.80c 9.60E-05   28062.80c 3.01E-04   28064.80c 7.43E-05",
        "     15031.80c 6.78E-04   16032.80c 4.16E-04   16033.80c 3.18E-06",
        "     16034.80c 1.75E-05   16036.80c 3.89E-08   26054.80c 3.36E-03",
        "     26056.80c 5.09E-02   26057.80c 1.15E-03   26058.80c 1.51E-04",
        "m4   6000.80c  1   $ graphite",
        "mt4  grph.20t",
    ]


def _ksrc_cards(fuelled: list[tuple[int, int, int]]) -> list[str]:
    """One ksrc point pair per fuelled slot, at the two foil centres."""
    pts = []
    for (i, j, k) in fuelled:
        xc, yc, zc = i * PX, j * PY, k * PZ
        pts.append((xc - KSRC_FOIL_OFFSET, yc, zc))
        pts.append((xc + KSRC_FOIL_OFFSET, yc, zc))
    lines = [f"ksrc {pts[0][0]:.5f} {pts[0][1]:.5f} {pts[0][2]:.5f}"]
    for x, y, z in pts[1:]:
        lines.append(f"     {x:.5f} {y:.5f} {z:.5f}")
    return lines


def generate_deck(width: int, length: int, height: int, *,
                  mode: str = "nominal",
                  densities: dict | None = None,
                  nominal_rho: float = RHO_FOIL_NOMINAL,
                  nominal_clad: float = RHO_CLAD_NOMINAL,
                  kcode: tuple = (10000, 1.0, 30, 130),
                  title: str | None = None) -> str:
    """Return a complete MCNP Hot Box input deck as a string.

    Parameters
    ----------
    width, length, height : int
        Number of columns (X), rows (Y) and layers (Z).  Max benchmark size
        is 8 x 4 x 12.
    mode : {"nominal", "measured"}
        ``nominal`` uses one averaged density everywhere (compact deck).
        ``measured`` preserves per-foil densities from ``densities``.
    densities : dict, optional
        Mapping ``(layer, col, row) -> FoilDensities``.  Required for
        ``mode="measured"``; slots missing from the table fall back to the
        nominal densities (with a comment in the deck).
    """
    if mode not in ("nominal", "measured"):
        raise ValueError(f"mode must be 'nominal' or 'measured', got {mode!r}")
    if mode == "measured" and not densities:
        raise ValueError("mode='measured' requires a densities table")

    if title is None:
        title = (f"Hot Box {width}x{length}x{height} "
                 f"({'measured' if mode == 'measured' else 'nominal'} density)")

    cells: list[str] = ["C  Cell cards"]

    # -- World / container / lattice ---------------------------------------
    # Slot -> universe mapping, plus per-slot fuel universe definitions.
    fuelled: list[tuple[int, int, int]] = []
    fuel_universe_of: dict[tuple[int, int, int], int] = {}
    fuel_cell_blocks: list[str] = []
    cell_no = 1000  # fuel-universe cell numbers start here

    next_universe = U_FUEL_MEASURED_BASE
    for k in range(height):
        for j in range(length):
            for i in range(width):
                if not is_fuelled(i, j, k):
                    continue
                fuelled.append((i, j, k))
                if mode == "nominal":
                    fuel_universe_of[(i, j, k)] = U_FUEL_NOMINAL
                else:
                    key = (k + 1, i + 1, j + 1)  # (layer, col, row), 1-based
                    d = densities.get(key)
                    note = ""
                    if d is None:
                        d = FoilDensities(nominal_rho, nominal_rho, nominal_clad)
                        note = "(no measured data -> nominal)"
                    u = next_universe
                    next_universe += 1
                    fuel_universe_of[(i, j, k)] = u
                    fuel_cell_blocks += _fuel_universe_cells(
                        cell_no, u, d, label=f"slot {key} {note}".strip())
                    cell_no += 11

    def slot_universe(i: int, j: int, k: int) -> int:
        return fuel_universe_of.get((i, j, k), U_VOID)

    # Lattice + container + world (universe 0)
    cells += _fill_array(width, length, height, slot_universe)
    cells.append(f"2 0 -98 fill={U_LATTICE} imp:n=1  $ assembly container")
    cells.append("3 0 98 -999 imp:n=1  $ world void")
    cells.append("4 0 999 imp:n=0  $ graveyard")

    # Universe definitions
    cells += _void_universe_cells(500)
    if mode == "nominal":
        d = FoilDensities(nominal_rho, nominal_rho, nominal_clad)
        cells += _fuel_universe_cells(600, U_FUEL_NOMINAL, d, label="(nominal)")
    else:
        cells += fuel_cell_blocks

    # -- Data cards --------------------------------------------------------
    npg, k0, skip, cycles = kcode
    data = [
        "C  Data cards",
        "mode n",
        f"kcode {npg} {k0} {skip} {cycles}",
    ]
    data += _ksrc_cards(fuelled)
    data += _material_cards()

    blocks = [title] + cells + [""] + _surface_cards(width, length, height) + [""] + data
    return "\n".join(blocks) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate Hot Box MCNP decks.")
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--length", type=int, default=4)
    p.add_argument("--height", type=int, default=12)
    p.add_argument("--mode", choices=["nominal", "measured", "both"],
                   default="both")
    p.add_argument("--densities",
                   help="CSV density table, or an original auto-generated "
                        ".sim deck to parse densities from.")
    p.add_argument("--outdir", default="decks")
    p.add_argument("-o", "--output",
                   help="Single output file (only with --mode nominal|measured).")
    args = p.parse_args(argv)

    densities = None
    if args.densities:
        if args.densities.lower().endswith(".csv"):
            densities = read_density_csv(args.densities)
        else:
            densities = parse_deck_densities(args.densities)

    import os
    os.makedirs(args.outdir, exist_ok=True)
    modes = ["nominal", "measured"] if args.mode == "both" else [args.mode]
    for mode in modes:
        if mode == "measured" and densities is None:
            p.error("--mode measured/both requires --densities")
        text = generate_deck(args.width, args.length, args.height,
                             mode=mode, densities=densities)
        if args.output and len(modes) == 1:
            path = args.output
        else:
            path = os.path.join(
                args.outdir,
                f"hb_{mode}_{args.width}_{args.length}_{args.height}.i")
        with open(path, "w") as fh:
            fh.write(text)
        print(f"wrote {path} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
