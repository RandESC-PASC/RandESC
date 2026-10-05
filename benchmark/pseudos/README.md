# mpc-azora-psp UPF pseudopotentials

This directory contains the `mpc-azora-psp` pseudopotential set converted to
Quantum ESPRESSO UPF v2.0.1 format for use with DFTK.

## Source

- Original files: `new_pseudos/SI/mpc_azora_psp/`
- Reference: A. Willand et al., *J. Chem. Phys.* **138**, 104109 (2013)
- Original format: CPMD/FHI-aims analytical `psppar.*` files
- XC functional: PBE (`SLA-PW-PBX-PBC`)
- Pseudopotential type: norm-conserving (NC), with NLCC where present

## Elements covered

`H`, `Li`, `Be`, `B`, `C`, `N`, `O`, `F`, `Na`, `Mg`, `Al`, `Si`, `P`, `S`,
`Cl`, `K`, `Ca`, `Sc`, `Ti`, `V`, `Cr`, `Mn`, `Fe`, `Co`, `Ni`, `Cu`, `Zn`,
`Ga`, `Ge`, `As`, `Se`, `Br`.

## Conversion

The conversion was performed with a patched `hgh2qe.x` (Quantum ESPRESSO 7.5)
compiled from `benchmark/tools/hgh2qe_mpc.f90`. The patch fixes the original
converter's 80-character output-line truncation that otherwise produces invalid
UPF XML.

A simple analytical model density is supplied as `rhoatom.dat` for each element
and normalized to the valence charge; this is used only as the initial SCF
guess and does not affect converged results.

To regenerate the UPFs from the source `psppar` files:

```bash
cd benchmark
python3 convert_mpc_azora.py
```

Any Fortran `***` overflow markers produced by the spline interpolation of the
initial density are replaced by zeros before validation.

## Validation

Each UPF file is loaded with `DFTK.load_psp` as part of the conversion script,
and its valence charge is checked against the value declared in the source
`psppar`.

## Note on Si

The `Si.upf` file here was regenerated consistently with the same converter and
reproduces the total energy of the pre-existing `new_pseudos/SI/mpc_azora_psp/Si.upf`
to better than 1e-9 Ha in a silicon unit-cell test.
