#!/usr/bin/env python3
"""Fetch random non-magnetic structures from the MC3D database (Materials Cloud)
via its OPTIMADE API and write each one as a .cif file.

Selection criteria:
  - all elements have atomic number < 37 (H ... Kr)
  - number of atoms (nsites) between 10 and 120 (inclusive)
  - non-magnetic: absolute magnetization below MAG_TOL

100 structures are sampled at random from the matching candidates and written
to the ``mc3d_optimade`` directory as CIF files.
"""

import os
import random

import requests
from ase import Atoms
from ase.io import write

BASE_URL = "https://optimade.materialscloud.org/main/mc3d-pbe-v1/v1/structures"
OUT_DIR = "mc3d_optimade"
N_STRUCTURES = 100
MIN_SITES = 10
MAX_SITES = 120
MAG_TOL = 1e-2  # |magnetization| below this is considered non-magnetic
PAGE_LIMIT = 100
SEED = 1234

# Elements with atomic number < 37 (Z = 1..36, H through Kr).
ELEMENTS_LT_37 = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
]

RESPONSE_FIELDS = ",".join([
    "chemical_formula_reduced",
    "cartesian_site_positions",
    "lattice_vectors",
    "species",
    "species_at_sites",
    "nsites",
    "_mcloud_absolute_magnetization",
])


def build_filter():
    elements = ",".join(f'"{e}"' for e in ELEMENTS_LT_37)
    return (
        f"nsites>={MIN_SITES} AND nsites<={MAX_SITES} "
        f"AND elements HAS ONLY {elements} "
        f"AND _mcloud_absolute_magnetization<{MAG_TOL}"
    )


def fetch_all_candidates():
    """Page through the OPTIMADE endpoint and return all matching entries."""
    entries = []
    params = {
        "filter": build_filter(),
        "response_fields": RESPONSE_FIELDS,
        "page_limit": PAGE_LIMIT,
    }
    url = BASE_URL
    while url:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        entries.extend(payload["data"])
        # follow pagination; subsequent "next" links already carry the query
        url = payload.get("links", {}).get("next")
        params = None
        print(f"  fetched {len(entries)} candidates...")
    return entries


def optimade_to_atoms(entry):
    """Convert one OPTIMADE structure entry into an ASE Atoms object."""
    attrs = entry["attributes"]

    # map each species name to its single chemical symbol
    symbol_of = {}
    for sp in attrs["species"]:
        symbols = sp["chemical_symbols"]
        # skip disordered sites / vacancies
        if len(symbols) != 1 or symbols[0] == "vacancy":
            return None
        symbol_of[sp["name"]] = symbols[0]

    try:
        symbols = [symbol_of[name] for name in attrs["species_at_sites"]]
    except KeyError:
        return None

    return Atoms(
        symbols=symbols,
        positions=attrs["cartesian_site_positions"],
        cell=attrs["lattice_vectors"],
        pbc=True,
    )


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Querying MC3D OPTIMADE API...")
    candidates = fetch_all_candidates()
    print(f"Found {len(candidates)} matching structures.")

    random.shuffle(candidates)

    written = 0
    for entry in candidates:
        if written >= N_STRUCTURES:
            break
        atoms = optimade_to_atoms(entry)
        if atoms is None:
            continue
        mc3d_id = entry["id"]
        formula = entry["attributes"].get("chemical_formula_reduced", "structure")
        fname = os.path.join(OUT_DIR, f"{formula}_{mc3d_id}.cif")
        write(fname, atoms)
        written += 1
        print(f"[{written:3d}/{N_STRUCTURES}] wrote {fname} ({len(atoms)} atoms)")

    print(f"\nDone. Wrote {written} CIF files to {OUT_DIR}/")
    if written < N_STRUCTURES:
        print(
            f"Warning: only {written} structures available/convertible "
            f"(requested {N_STRUCTURES})."
        )


if __name__ == "__main__":
    main()
