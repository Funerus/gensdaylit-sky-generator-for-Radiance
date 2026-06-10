# gensdaylit — Spectral, penumbra-aware sky generator for Radiance

`gensdaylit` is a Python-based alternative to the classic Radiance sky generator
`gendaylit`. It produces Radiance-compatible sun–sky descriptions that model both
the **spectral composition** of daylight (across the full solar spectrum, 
not just the visible band) and the **angular extent of the solar disc**,
enabling physically based **penumbra** (soft-shadow) rendering.

The purpose of the script is to create a `.rad` file containing the description
of the sun and sky in a format that Radiance (v6) understands. The typical
workflow of setting up the scene is unchanged — the only difference is that
instead of using another sky generator such as `gendaylit`, `gensky`, or
`genssky` to create the `.rad` file, you use `gensdaylit`. Ray tracing and
rendering then proceed as usual.

This code accompanies the paper:

> O. A. Katsikogiannis, O. Isabella, R. Santbergen, H. Ziar.
> *Tracing rays from leaves to sky: Multispectral, penumbra-aware irradiance
> modeling for agrivoltaic orchards.* Applied Energy 420 (2026) 128087.
> https://doi.org/10.1016/j.apenergy.2026.128087

---

## Features

- **Atmosphere-specific, full-spectrum sun–sky generation.** Rather than RGB
  triples, the sun and sky use Radiance's `specfile` spectral pattern primitive,
  reading per-band `(wavelength, value)` samples so an arbitrary number of
  wavebands can be represented.
- **Penumbra via proxy suns.** The finite solar disc is sampled with an
  equal-area Fibonacci lattice; proxy-sun radiances sum to the target DNI,
  yielding soft shadows.
- **Broadband mode.** Setting `n_bands = None` falls back to a conventional
  (non-spectral) sky.

---

## Repository structure

```
.
├── gensdaylit.py            # Core module — the Apollo sky generator class
├── README.md                # This file
├── LICENSE                  # Code license (choose one, e.g. MIT / BSD-3-Clause)
├── CITATION.cff             # Machine-readable citation metadata
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignores generated .rad/.dat files, caches, etc.
└── skies/                   # Output directory (created at runtime)
    └── spectrum/            # Per-band spectral .dat files (created at runtime)
```

> **Note:** `skies/` and `skies/spectrum/` are working directories that the
> generator writes into. They are excluded from version control via
> `.gitignore`; create them (empty) before the first run

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/Funerus/gensdaylit-sky-generator-for-Radiance.git
cd gensdaylit-sky-generator-for-Radiance
pip install -r requirements.txt
```

### Dependencies

`numpy`
`pandas`
`pvlib`
`loguru`

A working **[Radiance v6](https://www.radiance-online.org/)** installation is
required to render the generated `.rad` scene descriptions. `gensdaylit`
only *produces* the sun–sky description; rendering (e.g. `rtrace`, `rpict`) is
done by Radiance.

---

## Usage

The main entry point is the `Apollo` class. It is initialized with a
`daylight_params` object and called once per timestep via `generate_daylight`.

### 1. Configure daylight parameters

`daylight_params` must expose the following attributes:

| Attribute       | Type             | Meaning                                                        |
|-----------------|------------------|----------------------------------------------------------------|
| `n_bands`       | `int` or `None`  | Number of spectral bands (`None` → luminance-only sky)         |
| `n_suns`        | `int` or `None`  | Number of proxy suns (`None` → single point-source sun)        |
| `material`      | `str`            | Radiance material definition used for the ground plane         |
| `ground_radius` | `float`          | Radius of the ground ring (m)                                  |
| `albedo`        | `float` in [0,1] | Broadband ground albedo used by the Perez model                |

### 2. Provide per-timestep input

`generate_daylight` reads a single row, `comb_df`, with these fields:

| Field | Meaning                                  |
|-------|------------------------------------------|
| `DNI` | Direct normal irradiance (W/m²)          |
| `DHI` | Diffuse horizontal irradiance (W/m²)     |
| `Z`   | Solar zenith angle (degrees)             |
| `A`   | Solar azimuth (degrees, clockwise from N)|
| `m`   | Relative air mass                        |
| `DH`  | Daylight-hour index (used in filenames)  |

The timestamp `dt_index` **must be timezone-aware**.

### 3. Generate a sky

```python
import pandas as pd
from gensdaylit import Apollo

# Create a params object (e.g. a dataclass)
params = DaylightParams(
    n_bands=18,            # e.g. 18 bands over 300–1200 nm; or None
    n_suns=66,             # proxy suns for penumbra; or None for a point sun
    material="void",       # ground material definition
    ground_radius=100.0,   # m
    albedo=0.2,            # broadband albedo
)

sky = Apollo(params)

dt = pd.Timestamp("2023-06-21 14:00:00", tz="Europe/Rome")
comb_row = pd.Series({
    "DNI": 820.0, "DHI": 110.0,
    "Z": 28.0, "A": 215.0, "m": 1.13, "DH": 7,
})

daylight_errors = {}
sky_files, daylight_errors = sky.generate_daylight(dt, comb_row, daylight_errors)
print(sky_files)   # e.g. ['skies/gensdaylit_DH7_N18_Ns66.rad']
```

The returned `.rad` file can then be compiled into an octree and rendered with
Radiance.

### Spectral input files

When `n_bands` is set, the generator references per-band spectral files at:

```
skies/spectrum/sun_spectrum_N<bands>_<YYYYMMDD_HHMMSS>.dat
skies/spectrum/sky_spectrum_N<bands>_<YYYYMMDD_HHMMSS>.dat
```

These contain the SMARTS-derived, per-band spectral fractions for the sun and
sky and must be present for spectral renders. Their generation (SMARTS →
CAELUS/GISPLIT decomposition → per-band scaling) is part of the upstream
pipeline described in the paper and is **not** included in this repository.

---

## Citation

If you use this software, please cite the accompanying paper (see top) and, if
desired, the repository via `CITATION.cff`.

---

## License

The paper is published open access under CC BY 4.0. The **code** in this
repository is released under the [MIT License](LICENSE).

---

## Acknowledgements

Built on the [Radiance](https://www.radiance-online.org/) lighting-simulation
suite. Thanks to the Radiance community — in particular Greg Ward and Taoning
Wang. Atmospheric inputs in the accompanying study drew on AERONET and the KIT
KITcube observation system.
