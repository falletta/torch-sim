"""Test the deployed allegro-pol SiO2 model on a quartz structure via TorchSim."""

from __future__ import annotations

import torch
import torch_sim as ts
from ase.spacegroup import crystal
from pymatgen.io.ase import AseAtomsAdaptor
from allegro_pol.integrations.torchsim import NequIPPolTorchSimCalc

MODEL_PATH = "allegro-pol-SiO2.nequip.pt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alpha-quartz SiO2 (spacegroup P3_121, 9 atoms: 3 Si + 6 O)
atoms = crystal(
    ["Si", "O"],
    basis=[(0.4697, 0.0, 0.0), (0.4135, 0.2669, 0.1191)],
    spacegroup=152,
    cellpar=[4.916, 4.916, 5.405, 90, 90, 120],
)
structure = AseAtomsAdaptor.get_structure(atoms)

model = NequIPPolTorchSimCalc.from_compiled_model(
    MODEL_PATH, device=DEVICE, chemical_species_to_atom_type_map=True
)

state = ts.initialize_state([structure], model.device, model.dtype)
out = model(state)

print("=== Model output keys ===")
for key in sorted(out):
    val = out[key]
    if isinstance(val, torch.Tensor):
        print(f"  {key:30s}  shape={str(list(val.shape)):20s}  dtype={val.dtype}")

energy = out["energy"].cpu().item()
forces = out["forces"].cpu().numpy()
print(f"\nEnergy: {energy:.6f} eV")
print(f"Forces (shape {forces.shape}):\n{forces}")

if "stress" in out:
    stress = out["stress"].cpu().numpy()
    print(f"\nStress (shape {stress.shape}):\n{stress}")

for extra_key in ("polarization", "born_charges", "polarizability"):
    if extra_key in out:
        val = out[extra_key].cpu().numpy()
        print(f"\n{extra_key} (shape {val.shape}):\n{val}")
