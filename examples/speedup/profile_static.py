"""Profile TorchSim static path for n in [1, 10, 100, 500]; plot breakdown and total time.

Run: uv run profile_static.py
Output: profile_static.html (single file with all plots)
"""

# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[mace,test]",
#     "plotly",
# ]
# ///

import time
import typing
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls


warnings.filterwarnings(
    "ignore",
    message=(
        "The TorchScript type system doesn't support instance-level "
        "annotations on empty non-base types"
    ),
    category=UserWarning,
    module="torch.jit._check",
)


def profile_torchsim_static(n: int, base_structure: typing.Any) -> dict[str, float]:
    """Time initialize_state, load_states, and model loop separately for one n.

    Returns:
        Dict with keys: initialize_state, load_states, model_loop, total,
        neighbor_list, cell_shifts, mace_forward, n_batches (as float).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        return_raw_model=True,
        default_dtype="float64",
        device=str(device),
    )
    model = MaceModel(
        model=typing.cast("torch.nn.Module", loaded_model),
        device=device,
        compute_forces=True,
        compute_stress=True,
        dtype=torch.float64,
        enable_cueq=False,
    )
    batcher = ts.BinningAutoBatcher(
        model=model,
        max_memory_scaler=400_000,
        memory_scales_with="n_atoms_x_density",
    )
    structures = [base_structure] * n

    t0 = time.perf_counter()
    state = ts.initialize_state(structures, model.device, model.dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_init = time.perf_counter() - t0

    t0 = time.perf_counter()
    batcher.load_states(state)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    for sub_state, _ in batcher:
        model(sub_state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_loop = time.perf_counter() - t0

    total = t_init + t_load + t_loop
    print(f"n={n} profile:")
    print(f"  initialize_state: {t_init:.4f}s ({100 * t_init / total:.1f}%)")
    print(f"  load_states:       {t_load:.4f}s ({100 * t_load / total:.1f}%)")
    print(f"  model loop:        {t_loop:.4f}s ({100 * t_loop / total:.1f}%)")
    print(f"  total:             {total:.4f}s")

    batcher.load_states(state)
    sub_state, _ = next(iter(batcher))
    model(sub_state)
    t_nl = time.perf_counter()
    edge_index, mapping_system, unit_shifts = model.neighbor_list_fn(
        sub_state.positions,
        sub_state.row_vector_cell,
        sub_state.pbc,
        model.r_max,
        sub_state.system_idx,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_nl = time.perf_counter() - t_nl
    t_shifts = time.perf_counter()
    shifts = ts.transforms.compute_cell_shifts(
        sub_state.row_vector_cell, unit_shifts, mapping_system
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_shifts = time.perf_counter() - t_shifts
    data_dict = dict(
        ptr=model.ptr,
        node_attrs=model.node_attrs,
        batch=sub_state.system_idx,
        pbc=sub_state.pbc,
        cell=sub_state.row_vector_cell,
        positions=sub_state.positions,
        edge_index=edge_index,
        unit_shifts=unit_shifts,
        shifts=shifts,
        total_charge=sub_state.charge,
        total_spin=sub_state.spin,
    )
    t_fwd = time.perf_counter()
    model.model(data_dict, compute_force=True, compute_stress=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t_fwd
    n_batches = len(batcher.index_bins)
    print(f"  (first batch: neighbor_list={t_nl:.4f}s, "
          f"cell_shifts={t_shifts:.4f}s, mace_forward={t_fwd:.4f}s; n_batches={n_batches})")

    return {
        "initialize_state": t_init,
        "load_states": t_load,
        "model_loop": t_loop,
        "total": total,
        "neighbor_list": t_nl,
        "cell_shifts": t_shifts,
        "mace_forward": t_fwd,
        "n_batches": float(n_batches),
    }


N_STRUCTURES = [1, 1, 10, 100, 500]


def plot_profile_sweep(
    timings_by_n: list[dict[str, float]],
    n_list: list[int],
    output_path: str = "profile_static.html",
) -> None:
    """Single HTML: grouped bar (phase breakdown by n) + total time vs n."""
    phases = ["initialize_state", "load_states", "model_loop"]
    colors = {"initialize_state": "#2ca02c", "load_states": "#ff7f0e", "model_loop": "#1f77b4"}
    n_cats = len(n_list)
    x_centers = list(range(n_cats))
    offset = 0.25
    bar_width = 0.2

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Profile breakdown by n_structures", "Total time vs n_structures"),
        vertical_spacing=0.12,
    )

    for i, phase in enumerate(phases):
        phase_offset = offset * (i - 1)
        x_pos = [c + phase_offset for c in x_centers]
        y = [t[phase] for t in timings_by_n]
        fig.add_trace(
            go.Bar(
                name=phase,
                x=x_pos,
                y=y,
                marker_color=colors[phase],
                text=[f"{v:.3f}s" for v in y],
                textposition="outside",
                width=bar_width,
            ),
            row=1,
            col=1,
        )

    totals = [t["total"] for t in timings_by_n]
    fig.add_trace(
        go.Scatter(
            x=x_centers,
            y=totals,
            mode="lines+markers+text",
            text=[f"{t:.2f}s" for t in totals],
            textposition="top center",
            line={"width": 2},
            marker={"size": 12},
            name="total",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text="n_structures",
        row=1,
        col=1,
        tickvals=x_centers,
        ticktext=[str(n) for n in n_list],
    )
    fig.update_yaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(
        title_text="n_structures",
        row=2,
        col=1,
        tickvals=x_centers,
        ticktext=[str(n) for n in n_list],
    )
    fig.update_yaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(
        height=700,
        legend={"x": 0.01, "y": 0.99},
    )
    fig.write_html(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    mgo_ase = bulk(name="MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    base_structure = AseAtomsAdaptor.get_structure(atoms=mgo_ase)  # pyright: ignore[reportArgumentType]
    timings_by_n = [profile_torchsim_static(n_val, base_structure) for n_val in N_STRUCTURES]
    plot_profile_sweep(timings_by_n, N_STRUCTURES, output_path="profile_static.html")
