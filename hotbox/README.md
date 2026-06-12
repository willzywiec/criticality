# Hot Box MCNP deck generator

`hbgen.py` is a standalone rewrite of the original `HBMCNPgen.py`. It builds
the Hot Box critical-assembly MCNP input deck using **repeated structures**
(universes + a `LAT=1` lattice with a `FILL` array) instead of emitting one
`LIKE n BUT TRCL=(...)` card per cell.

| Deck | Original (`indeckHBfullstack_12_8_4.sim`) | New nominal | New measured |
|------|------------------------------------------:|------------:|-------------:|
| Lines | 5824 | **497** | **2789** |

Both new decks describe the *same* geometry; they differ only in how foil
densities are treated.

## The two models

* **Nominal** (`hb_nominal_*.i`) — every foil uses one averaged density
  (18.5835 g/cc) and every cladding one averaged density (7.2694 g/cc). The
  whole assembly collapses to **2 universes** (fuel + void placeholder) placed
  by the lattice. This is the "simple" model.
* **Measured** (`hb_measured_*.i`) — preserves each individually measured
  foil/cladding density. Because MCNP cannot vary material between instances of
  one universe, each fuelled slot gets its own fuel universe (191 of them), but
  the geometry surfaces are still defined **once** and placement is still a
  single lattice. This is the high-fidelity benchmark model.

## How the geometry maps to a lattice

One lattice element (pitch 30.45968 × 60.91936 × 7.62 cm = 12" × 24" × 3")
contains, bottom to top: a 1" graphite block, the fuel-element plane (two
U-235 foils in a stainless sheath), and a 2" graphite block, flanked by
graphite side rails. The fuelled checkerboard — which flips parity every
layer, exactly as in the benchmark slides — becomes the `FILL` array:

```
fuelled  <=>  (layer + col + row) even   (1-based)
         <=>  (i + j + k) odd            (0-based)
```

This parity rule was validated against all 1149 fuel/clad cells in the
reference deck (0 mismatches).

## Simplifications

These reduce card count with expected-negligible reactivity impact. Confirm
once with a k-eff comparison against the reference deck before production use.

1. **Cladding** — the original folded stainless sheath (40+ `ARB` polyhedra)
   is modelled as two thin SS sheets above/below the foils. This preserves the
   stainless **mass** (~14 cm³/element) rather than the exact fold geometry. (A
   naive "fill the cladding box with solid SS" model would have ~15× too much
   steel and is *not* used.)
2. **Bolt holes** — the 24 air-filled `RCC` holes per graphite block are
   omitted (small air voids in graphite, far from the fuel).

Everything else — pitches, block/foil dimensions, materials (including the
`grph.20t` thermal scattering card), and the measured densities — is preserved.

## Generated decks

`decks/` holds the full approach-to-critical series: nominal and measured
decks for the benchmark 8x4 cross section at every stack height from 1 to 12
layers, named `hb_<mode>_<height>_<width>_<length>.i` (matching the reference
`12_8_4` convention). The named reference cases `3_8_4` and `12_8_4` are both
included. Regenerate them all with:

```bash
python generate_all.py
```

## Usage

```bash
# Regenerate both decks at a single size (8x4x12) from the density table
python hbgen.py --width 8 --length 4 --height 12 --densities densities.csv

# A single nominal deck of a custom size
python hbgen.py --width 8 --length 4 --height 6 --mode nominal -o small.i

# Parse densities straight from an original auto-generated deck
python hbgen.py --densities ../indeckHBfullstack_12_8_4.sim --mode measured
```

```python
# As a library
from hbgen import generate_deck, parse_deck_densities, read_density_csv

dens = read_density_csv("densities.csv")
text = generate_deck(8, 4, 12, mode="measured", densities=dens)
```

## Files

* `hbgen.py` — the generator module (Python 3, standard library only).
* `generate_all.py` — regenerates the full 1–12 layer deck series.
* `densities.csv` — measured per-slot densities, parsed from the reference
  deck (`layer, col, row, rho_rf, rho_lf, rho_clad`).
* `reference_indeck_12_8_4.sim` — the original full-stack deck, kept for
  comparison.
* `decks/` — the generated series (`hb_nominal_*` and `hb_measured_*`,
  heights 1–12 at 8×4).
