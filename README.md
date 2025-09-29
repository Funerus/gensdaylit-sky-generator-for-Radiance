[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17228395.svg)](https://doi.org/10.5281/zenodo.17228395)

# gensdaylit

`gensdaylit.py` is a Python-based alternative to the Radiance `gendaylit` sky generator.  
It models both the spectral composition of daylight and the angular extent of the solar disc,
providing `.rad` scene files for Radiance simulations.  
This repository includes the initial Python script, sample input files (`.dat`),
and example output files (`.rad`) to reproduce basic runs.

## Requirements
Python ≥ 3.10

```txt
numpy>=1.23
pandas>=1.5
pvlib>=0.10
loguru>=0.7
