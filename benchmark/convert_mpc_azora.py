#!/usr/bin/env python3
"""Batch-convert mpc-azora-psp CPMD psppar files to UPF v2.0.1 using hgh2qe.

Usage:
    python3 convert_mpc_azora.py

This script:
  1. Reads every psppar.<ELEM> in new_pseudos/SI/mpc_azora_psp/.
  2. Generates a rhoatom.dat file on the hgh2qe radial grid.
  3. Runs a patched hgh2qe.x (long output lines, correct XML attributes).
  4. Replaces any Fortran '***' overflow markers with zeros.
  5. Validates each UPF by loading it with DFTK.

The converted files are written to pseudos/mpc_azora_upf/<ELEM>.upf.
"""

import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Project-relative paths
REPO_ROOT = Path(__file__).resolve().parent.parent
PSPPAR_DIR = REPO_ROOT / "new_pseudos" / "SI" / "mpc_azora_psp"
OUT_DIR = REPO_ROOT / "benchmark" / "pseudos" / "mpc_azora_upf"
HGH2QE = REPO_ROOT / "benchmark" / "tools" / "hgh2qe_mpc.x"

def parse_psppar(path):
    """Extract znuc, zion, rcov, and pspcod from a psppar file."""
    with open(path, "r") as f:
        lines = f.readlines()

    # First line: "spin-polarized 1  rcov  rprb"
    parts = lines[0].split()
    rcov = float(parts[2])

    # Second line: znuc zion date
    znuc, zion = map(float, lines[1].split()[:2])

    # Third line: pspcod ixc ...
    pspcod = int(lines[2].split()[0])

    return znuc, zion, rcov, pspcod


def generate_rhoatom(znuc, zion, rcov):
    """Generate rhoatom.dat values on the hgh2qe radial grid.

    The values represent 4*pi*r^2*rho(r) and are normalized to integrate to zion.
    A simple Gaussian-like model is used; this is only the initial guess density
    for DFTK's SCF and does not affect the converged result.
    """
    xmin = -7.0
    amesh = 0.0125
    rmax = 100.0
    mesh = 1 + int((math.log(znuc * rmax) - xmin) / amesh)
    mesh = (mesh // 2) * 2 + 1

    rr = [math.exp(xmin + i * amesh) / znuc for i in range(mesh)]
    rab = [r * amesh for r in rr]

    fourpi = 4.0 * math.pi
    rho = []
    for r in rr:
        if r == 0.0:
            rho.append(0.0)
        else:
            # Gaussian-like valence density; smooth and non-negative
            sigma = max(rcov, 0.5)
            val = fourpi * r**2 * math.exp(-0.5 * (r / sigma) ** 2)
            rho.append(val)

    # Normalize to zion
    integral = sum(rho[i] * rab[i] for i in range(mesh))
    if integral <= 0.0:
        raise ValueError("rhoatom integral is non-positive")
    rho = [v * (zion / integral) for v in rho]

    # Pad to a multiple of 4 (required by hgh2qe's read loop)
    while len(rho) % 4 != 0:
        rho.append(0.0)

    return rho


def write_rhoatom_dat(path, rho):
    with open(path, "w") as f:
        for i in range(0, len(rho), 4):
            f.write("  ".join(f"{v:22.15e}" for v in rho[i : i + 4]) + "\n")


def strip_stars(text):
    """Replace Fortran '***' overflow markers with a valid zero field."""
    return re.sub(r"\*+", "0.000000000000000E+000", text)


def validate_upf(path, element):
    """Check that DFTK can parse the UPF and report its Zion."""
    cmd = [
        "julia",
        "--project=benchmark/cpu_project",
        "-e",
        f'using DFTK; psp=load_psp("{path}"); println("OK ", psp.Zion)',
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"DFTK failed to load {path}:\n{result.stderr}")
    print(f"  validated ({result.stdout.strip()})")


def convert_one(psppar_path, out_path):
    element = psppar_path.name.replace("psppar.", "")
    print(f"Converting {element} ...")

    znuc, zion, rcov, pspcod = parse_psppar(psppar_path)
    print(f"  znuc={znuc}, zion={zion}, rcov={rcov}, pspcod={pspcod}")

    if pspcod not in (2, 3, 10, 12):
        raise ValueError(f"pspcod={pspcod} not supported by hgh2qe")

    rho = generate_rhoatom(znuc, zion, rcov)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "psppar").write_text(psppar_path.read_text())
        write_rhoatom_dat(tmpdir / "rhoatom.dat", rho)

        if not HGH2QE.exists():
            build_script = HGH2QE.with_suffix("").parent / "build_hgh2qe.sh"
            print(f"  {HGH2QE.name} not found, building ...")
            build = subprocess.run([str(build_script)], cwd=REPO_ROOT)
            if build.returncode != 0 or not HGH2QE.exists():
                raise FileNotFoundError(
                    f"{HGH2QE} not found and build failed. "
                    f"Run benchmark/tools/build_hgh2qe.sh manually."
                )

        result = subprocess.run(
            [str(HGH2QE)], cwd=tmpdir, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"hgh2qe failed for {element}:\n{result.stderr}")

        upf_text = strip_stars(result.stdout)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(upf_text)

    validate_upf(out_path, element)


def main():
    if not PSPPAR_DIR.exists():
        print(f"Error: source directory not found: {PSPPAR_DIR}", file=sys.stderr)
        sys.exit(1)

    psppar_files = sorted(PSPPAR_DIR.glob("psppar.*"))
    print(f"Found {len(psppar_files)} psppar files")

    failures = []
    for psppar_path in psppar_files:
        element = psppar_path.name.replace("psppar.", "")
        out_path = OUT_DIR / f"{element}.upf"
        try:
            convert_one(psppar_path, out_path)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append(element)

    print()
    if failures:
        print(f"Failures: {failures}")
        sys.exit(1)
    print(f"All {len(psppar_files)} pseudopotentials converted successfully.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
