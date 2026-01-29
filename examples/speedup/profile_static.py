"""Profile TorchSim and ASE static path for n in [1, 10, 100, 500];
plot breakdown and total time.

Run: uv run profile_static.py
Output: profile_static.html (single file with all plots;
includes ASE total for comparison)
"""

# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[mace,test]",
#     "plotly",
#     "pydantic",
# ]
# ///

import time
import typing
import warnings

import plotly.graph_objects as go
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from plotly.subplots import make_subplots
from pydantic import BaseModel
from pymatgen.io.ase import AseAtomsAdaptor

import torch_sim as ts
from torch_sim.autobatching import calculate_memory_scaler, to_constant_volume_bins
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


class TorchSimStaticProfile(BaseModel):
    """Timing breakdown for one static profile run."""

    initialize_state: float
    load_states: float
    load_states_split: float
    load_states_memory_scalers: float
    load_states_binning: float
    model_loop: float
    model_loop_get_batch: float
    model_loop_forward: float
    total: float
    neighbor_list: float
    cell_shifts: float
    mace_forward: float
    n_batches: float


class AseStaticProfile(BaseModel):
    """Timing breakdown for one ASE static profile run (same job as TorchSim)."""

    setup: float  # copy atoms + attach calculator
    model_loop: float  # get_potential_energy + get_forces + get_stress for each
    total: float


def profile_torchsim_static(n: int, base_structure: typing.Any) -> TorchSimStaticProfile:  # noqa: C901, PLR0915
    """Time initialize_state, load_states, and model loop separately for one n.

    Returns:
        TorchSimStaticProfile with timing fields.
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

    # Single run for total model_loop time (one sync at end so GPU can overlap work).
    t0 = time.perf_counter()
    get_batch_times: list[float] = []
    model_times: list[float] = []
    it = iter(batcher)
    t_prev = t0
    while True:
        try:
            sub_state, _ = next(it)
            t_after_get = time.perf_counter()
            get_batch_times.append(t_after_get - t_prev)
            model(sub_state)
            t_prev = time.perf_counter()
            model_times.append(t_prev - t_after_get)
        except StopIteration:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_loop_elapsed = time.perf_counter() - t0
    t_loop_get_batch = sum(get_batch_times)
    t_loop_forward = sum(model_times)
    # First breakdown bar = sum of model_loop breakdown plot (get_batch + forward).
    model_loop = t_loop_get_batch + t_loop_forward
    total = t_init + t_load + model_loop

    if t_loop_elapsed < model_loop - 1e-9:
        raise ValueError("elapsed loop time must be >= breakdown sum")

    t0 = time.perf_counter()
    state_slices = state.split() if isinstance(state, ts.SimState) else list(state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_load_split = time.perf_counter() - t0
    t0 = time.perf_counter()
    memory_scalers = [
        calculate_memory_scaler(s, batcher.memory_scales_with) for s in state_slices
    ]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_load_scalers = time.perf_counter() - t0
    index_to_scaler = dict(enumerate(memory_scalers))
    t0 = time.perf_counter()
    index_bins = to_constant_volume_bins(
        index_to_scaler, max_volume=batcher.max_memory_scaler
    )
    index_bins = [list(batch.keys()) for batch in index_bins]
    _ = [[state_slices[i] for i in bin_idx] for bin_idx in index_bins]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_load_binning = time.perf_counter() - t0

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

    return TorchSimStaticProfile(
        initialize_state=t_init,
        load_states=t_load,
        load_states_split=t_load_split,
        load_states_memory_scalers=t_load_scalers,
        load_states_binning=t_load_binning,
        model_loop=model_loop,
        model_loop_get_batch=t_loop_get_batch,
        model_loop_forward=t_loop_forward,
        total=total,
        neighbor_list=t_nl,
        cell_shifts=t_shifts,
        mace_forward=t_fwd,
        n_batches=float(n_batches),
    )


def profile_ase_static(n: int, ase_atoms: typing.Any) -> AseStaticProfile:
    """Time ASE static (same job as TorchSim): n structures, energy+forces+stress.

    Uses the same MACE model and device as profile_torchsim_static for comparison.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ase_calc = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        default_dtype="float64",
        device=str(device),
        enable_cueq=False,
    )
    t0 = time.perf_counter()
    ase_atoms_list = [ase_atoms.copy() for _ in range(n)]
    for at in ase_atoms_list:
        at.calc = ase_calc
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    for at in ase_atoms_list:
        at.get_potential_energy()
        at.get_forces()
        at.get_stress()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_loop = time.perf_counter() - t0

    return AseStaticProfile(
        setup=t_setup,
        model_loop=t_loop,
        total=t_setup + t_loop,
    )


N_STRUCTURES = [1, 1, 10, 100, 500]


def plot_profile_sweep(
    timings_by_n: list[TorchSimStaticProfile],
    n_list: list[int],
    output_path: str = "profile_static.html",
    ase_timings_by_n: list[AseStaticProfile] | None = None,
) -> None:
    """Single HTML: main phases, load_states/model_loop breakdowns, total vs n."""
    phases = ["initialize_state", "load_states", "model_loop"]
    colors = {
        "initialize_state": "#2ca02c",
        "load_states": "#ff7f0e",
        "model_loop": "#1f77b4",
    }
    load_phases = [
        "load_states_split",
        "load_states_memory_scalers",
        "load_states_binning",
    ]
    load_colors = {
        "load_states_split": "#8c564b",
        "load_states_memory_scalers": "#e377c2",
        "load_states_binning": "#bcbd22",
    }
    loop_phases = ["model_loop_get_batch", "model_loop_forward"]
    loop_colors = {"model_loop_get_batch": "#17becf", "model_loop_forward": "#1f77b4"}
    n_cats = len(n_list)
    x_centers = list(range(n_cats))
    offset = 0.25
    bar_width = 0.2

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Profile breakdown by n_structures",
            "load_states breakdown",
            "model_loop breakdown",
            "Total time vs n_structures",
        ),
        vertical_spacing=0.08,
    )

    for i, phase in enumerate(phases):
        phase_offset = offset * (i - 1)
        x_pos = [c + phase_offset for c in x_centers]
        y = [getattr(t, phase) for t in timings_by_n]
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

    for i, phase in enumerate(load_phases):
        phase_offset = offset * (i - 1)
        x_pos = [c + phase_offset for c in x_centers]
        y = [getattr(t, phase, 0.0) for t in timings_by_n]
        fig.add_trace(
            go.Bar(
                name=phase.replace("load_states_", ""),
                x=x_pos,
                y=y,
                marker_color=load_colors[phase],
                text=[f"{v:.3f}s" for v in y],
                textposition="outside",
                width=bar_width,
            ),
            row=2,
            col=1,
        )

    for i, phase in enumerate(loop_phases):
        phase_offset = offset * (i - 0.5)
        x_pos = [c + phase_offset for c in x_centers]
        y = [getattr(t, phase, 0.0) for t in timings_by_n]
        fig.add_trace(
            go.Bar(
                name=phase.replace("model_loop_", ""),
                x=x_pos,
                y=y,
                marker_color=loop_colors[phase],
                text=[f"{v:.3f}s" for v in y],
                textposition="outside",
                width=bar_width,
            ),
            row=3,
            col=1,
        )

    totals = [t.total for t in timings_by_n]
    fig.add_trace(
        go.Scatter(
            x=x_centers,
            y=totals,
            mode="lines+markers+text",
            text=[f"{t:.2f}s" for t in totals],
            textposition="top center",
            line={"width": 2},
            marker={"size": 12},
            name="TorchSim total",
        ),
        row=4,
        col=1,
    )
    if ase_timings_by_n is not None and len(ase_timings_by_n) == len(n_list):
        ase_totals = [t.total for t in ase_timings_by_n]
        fig.add_trace(
            go.Scatter(
                x=x_centers,
                y=ase_totals,
                mode="lines+markers+text",
                text=[f"{t:.2f}s" for t in ase_totals],
                textposition="bottom center",
                line={"width": 2, "dash": "dash"},
                marker={"size": 12, "symbol": "square"},
                name="ASE total",
            ),
            row=4,
            col=1,
        )

    tickvals = dict(tickvals=x_centers, ticktext=[str(n) for n in n_list])
    for row in range(1, 5):
        fig.update_xaxes(title_text="n_structures", row=row, col=1, **tickvals)
        fig.update_yaxes(title_text="Time (s)", row=row, col=1)
    fig.update_layout(
        height=1000,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        margin=dict(t=120),
    )
    fig.write_html(output_path)


if __name__ == "__main__":
    mgo_ase = bulk(name="MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    base_structure = AseAtomsAdaptor.get_structure(atoms=mgo_ase)  # pyright: ignore[reportArgumentType]
    timings_by_n = [
        profile_torchsim_static(n_val, base_structure) for n_val in N_STRUCTURES
    ]
    ase_timings_by_n = [profile_ase_static(n_val, mgo_ase) for n_val in N_STRUCTURES]
    plot_profile_sweep(
        timings_by_n,
        N_STRUCTURES,
        output_path="profile_static.html",
        ase_timings_by_n=ase_timings_by_n,
    )
