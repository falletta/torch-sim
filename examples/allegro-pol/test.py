"""Test the deployed allegro-pol SiO2 model on a quartz structure."""

from __future__ import annotations

import torch
from allegro_pol._keys import POLARIZABILITY_KEY
from ase.spacegroup import crystal
from nequip.data import AtomicDataDict
from nequip.integrations.ase import NequIPCalculator

MODEL_PATH = "allegro-pol-SiO2.nequip.zip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Alpha-quartz SiO2 (spacegroup P3_121, 9 atoms: 3 Si + 6 O)
atoms = crystal(
    ["Si", "O"],
    basis=[(0.4697, 0.0, 0.0), (0.4135, 0.2669, 0.1191)],
    spacegroup=152,
    cellpar=[4.916, 4.916, 5.405, 90, 90, 120],
)

calc = NequIPCalculator._from_saved_model(model_path=MODEL_PATH, device=DEVICE)

data = calc.atoms_to_data(atoms)
out = calc.call_model(data)

print("=== Model output keys ===")
for key in sorted(out):
    val = out[key]
    if isinstance(val, torch.Tensor):
        print(f"  {key:30s}  shape={str(list(val.shape)):20s}  dtype={val.dtype}")

energy = out[AtomicDataDict.TOTAL_ENERGY_KEY].detach().cpu().item()
forces = out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
print(f"\nEnergy: {energy:.6f} eV")
print(f"Forces (shape {forces.shape}):\n{forces}")

if AtomicDataDict.POLARIZATION_KEY in out:
    polarization = out[AtomicDataDict.POLARIZATION_KEY].detach().cpu().numpy()
    print(f"\nPolarization (shape {polarization.shape}):\n{polarization}")

if AtomicDataDict.BORN_CHARGE_KEY in out:
    born_charges = out[AtomicDataDict.BORN_CHARGE_KEY].detach().cpu().numpy()
    print(f"\nBorn effective charges (shape {born_charges.shape}):\n{born_charges}")

if POLARIZABILITY_KEY in out:
    polarizability = out[POLARIZABILITY_KEY].detach().cpu().numpy()
    print(f"\nPolarizability (shape {polarizability.shape}):\n{polarizability}")
