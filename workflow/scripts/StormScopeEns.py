#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script implements a refactored version of the StormScope ensemble nowcast pipeline, which generates 6-hour forecasts of GOES satellite imagery and MRMS radar reflectivity. The script is designed to be run on an NVIDIA GPU and uses PyTorch for model inference. It includes functionality for loading pre-trained StormScope models, preparing input data, performing autoregressive rollouts, computing derived precipitation fields, and generating quick-look plots. The script is structured to allow for multiprocessing across multiple GPUs to speed up the rollout process. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: May 2026
Version: 1.0.0
"""
from __future__ import annotations

import os
import sys
import time
import shutil
import logging
import tempfile
import torch
import torch.multiprocessing as mp
import argparse
import numpy as np
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, NamedTuple, NoReturn
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_default_cache: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.environ.setdefault("EARTH2STUDIO_CACHE", _default_cache)

# --- Type aliases --------------------------------------------------------

CoordSystem = "OrderedDict[str, np.ndarray]"

Model = Any        # earth2studio prognostic model (StormScopeGOES/MRMS)
Package = Any      # earth2studio model package
DataSource = Any   # earth2studio data source (GFS_FX, GOES, MRMS)

# --- Status logging ------------------------------------------------------

START: float = time.time() 
PHASE_TIMES: list[tuple[str, float]] = [] 

log: logging.Logger = logging.getLogger("stormscope")


def _fmt_hms(seconds: float) -> str:
    """
    This helper formats a duration in seconds into a human-readable H:MM:SS string. It's used for logging elapsed time and ETA during the rollout. 

    Parameters:
        seconds (float): The duration in seconds to format.

    Returns:
        str: A string representing the duration in H:MM:SS format.
    """
    whole = int(seconds)
    return f"{whole // 3600}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"


class _ElapsedFormatter(logging.Formatter):
    """ Prefix every record with wall-clock time + elapsed since START. """

    def format(self: '_ElapsedFormatter', 
               record: logging.LogRecord) -> str:
        """
        This method formats a logging record by prefixing it with the current wall-clock time and the elapsed time since the START of the process. The timestamp is formatted as HH:MM:SS, and the elapsed time is formatted using the _fmt_hms helper function. 

        Parameters:
            record (logging.LogRecord): The logging record to format.

        Returns:
            str: A formatted string with the current wall-clock time and elapsed time since START.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = _fmt_hms(time.time() - START)
        return f"{timestamp} [+{elapsed}] {record.getMessage()}"


def setup_logging() -> None:
    """
    This function sets up the logging configuration for the process. It checks if any logging handlers are already configured (which would be the case in the main process), and if not, it creates a new StreamHandler that outputs to standard output (sys.stdout). The handler is configured to use the _ElapsedFormatter defined above, which prefixes log messages with timestamps and elapsed time. The logging level is set to INFO, and propagation is disabled to prevent duplicate log messages in child processes. This function is idempotent and can be safely called multiple times, which is necessary because worker processes spawned with the "spawn" method do not inherit logging handlers from the parent process. 

    Parameters:
        None

    Returns:
        None
    """
    if log.handlers:
        return
        
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ElapsedFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


@contextmanager
def phase(label: str) -> Iterator[None]:
    """
    This context manager is used to log the start and end of a phase of the pipeline, along with the time taken for that phase. When entering the context, it logs a message indicating that the phase has started. It then yields control back to the caller, allowing the code within the context to execute. After the code in the context has finished executing, it calculates the duration of the phase and logs a message indicating that the phase is done along with the time taken. The duration is also appended to the PHASE_TIMES list for an end-of-run summary. 

    Parameters:
        label (str): The label for the phase.

    Returns:
        An iterator that yields control back to the caller.
    """
    log.info(f"{label} ...")
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    PHASE_TIMES.append((label, duration))
    log.info(f"{label} done in {duration:.1f}s")


GOES_KNOWN: list[str] = [
    "3km_10min_natten_pure_obs_cos_zenith_input_eoe",
    "6km_10min_natten_pure_obs_zenith_eoe",
    "6km_10min_natten_pure_obs_zenith_6steps",
    "6km_60min_natten_cos_zenith_input_eoe_v2",
]

MRMS_KNOWN: list[str] = [
    "6km_10min_natten_pure_obs_mrms_obs_6steps",
    "6km_60min_natten_cos_zenith_input_mrms_eoe",
]


def refc_to_rain_rate(reflectivity_dbz: np.ndarray,
                       a: float = 300.0,
                       b: float = 1.4,
                       cap_dbz: float = 53.0,
                       floor_dbz: float = 5.0) -> np.ndarray:
    """
    This function converts composite reflectivity in dBZ to rain rate in mm/hr using the Z-R relationship defined by the equation Z = a * R^b, where Z is the reflectivity in linear units (not dBZ), R is the rain rate, and a and b are empirically derived coefficients. The function takes an array of reflectivity values in dBZ and applies the following steps to compute the rain rate:
    1. It caps the input reflectivity values at a specified maximum (cap_dbz) to limit the influence of hail contamination, which can lead to unrealistically high reflectivity values that do not correspond to rain.
    2. It converts the capped reflectivity from dBZ to linear units by using the formula Z = 10^(dBZ/10).
    3. It applies the Z-R relationship to compute the rain rate R from the reflectivity Z using the formula R = (Z/a)^(1/b).
    4. It sets the rain rate to 0 for reflectivity values below a specified minimum (floor_dbz).
    5. It preserves NaN values in the input reflectivity array.

    Parameters:
        reflectivity_dbz (np.ndarray): The input reflectivity in dBZ.
        a (float): Z-R prefactor. Default is 300.0.
        b (float): Z-R exponent. Default is 1.4.
        cap_dbz (float): Maximum dBZ value to consider. Default is 53.0.
        floor_dbz (float): Minimum dBZ value to consider. Default is 5.0.

    Returns:
        np.ndarray: The calculated rain rate in mm/hr.
    """
    capped_dbz = np.clip(reflectivity_dbz, None, cap_dbz)
    rain_rate = (10.0 ** (capped_dbz / 10.0) / a) ** (1.0 / b)
    rain_rate = np.where(reflectivity_dbz < floor_dbz, 0.0, rain_rate)
    return np.where(np.isnan(reflectivity_dbz), np.nan, rain_rate)


def parse_args() -> argparse.Namespace:
    """
    This function parses command-line arguments for the StormScope ensemble nowcast script. It uses the argparse library to define various options that can be specified when running the script.

    Parameters:
        None

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="2023-12-05T12:00",
                        help="Initialization time, ISO (UTC). Default: 2023-12-05T12:00")
    parser.add_argument("--n-ensemble", type=int, default=4,
                        help="Ensemble members (batch dim). Default: 4")
    parser.add_argument("--n-steps", type=int, default=36,
                        help="Autoregressive steps; 10 min each -> 36 ~= 6 h. Default: 36")
    parser.add_argument("--goes-model", default=GOES_KNOWN[2])
    parser.add_argument("--mrms-model", default=MRMS_KNOWN[0])
    parser.add_argument("--goes-satellite", default="goes16")
    parser.add_argument("--scan-mode", default="C", help="GOES scan mode (C=CONUS)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir",
                        default=os.environ.get("STORMSCOPE_OUTPUT_DIR", "outputs"))
    parser.add_argument("--plot-channel", default="abi13c",
                        help="GOES band to map (abi13c = clean IR 10.35 um)")
    parser.add_argument("--no-precip", action="store_true",
                        help="Skip derived precip (rate + accumulation) outputs")
    parser.add_argument("--zr-a", type=float, default=300.0,
                        help="Z-R prefactor a in Z=a*R^b (MRMS radar-only: 300)")
    parser.add_argument("--zr-b", type=float, default=1.4,
                        help="Z-R exponent b in Z=a*R^b (MRMS radar-only: 1.4)")
    parser.add_argument("--refc-cap-dbz", type=float, default=53.0,
                        help="Cap dBZ before Z-R to limit hail contamination")
    parser.add_argument("--refc-floor-dbz", type=float, default=5.0,
                        help="dBZ below this maps to zero rain rate")
    return parser.parse_args()


def load_stormscope_model(model_cls: type,
                          *,
                          package: Package,
                          conditioning: DataSource,
                          model_name: str,
                          device: torch.device,
                          kind: str | None = None,
                          known: list[str] | None = None) -> Model:
    """
    This function loads a StormScope model of the specified class (model_cls) from the given package, using the specified conditioning data source and model name. It attempts to load the model and move it to the specified device (e.g., GPU). If the loading fails and a kind and known list are provided, it calls the fail_with_available function to log the error and available options before exiting. This function abstracts away the details of loading different types of StormScope models (e.g., GOES vs MRMS) and provides a common interface for doing so.

    Parameters:
        model_cls (type): The class of the model to load (e.g., StormScopeGOES or StormScopeMRMS).
        package (Package): The model package from which to load the model.
        conditioning (DataSource): The data source used for conditioning the model.
        model_name (str): The name of the model checkpoint to load.
        device (torch.device): The device to which the loaded model should be moved.
        kind (str | None): An optional string indicating the kind of model (e.g., "GOES" or "MRMS") for error reporting purposes.
        known (list[str] | None): An optional list of known valid model names for error reporting purposes.

    Returns:
        Model: The loaded and device-moved model instance.
    """
    def _load() -> Model:
        """
        This inner function performs the actual loading of the model using the model_cls's load_model method, passing in the package, conditioning data source, and model name. It then moves the loaded model to the specified device and sets it to evaluation mode. This function is called within a try-except block in the outer function to handle any exceptions that may occur during loading and provide informative error messages if the loading fails.

        Parameters:
            None

        Returns:
            Model: The loaded and device-moved model instance.
        """
        return model_cls.load_model(
            package=package,
            conditioning_data_source=conditioning,
            model_name=model_name,
        ).to(device).eval()

    if kind is None:
        return _load()
    try:
        return _load()
    except Exception as exc:  # noqa: BLE001 - surface real options then exit
        fail_with_available(kind, model_name, exc, package, known)


def build_goes_interpolators(goes_model: Model,
                             goes_lat: np.ndarray,
                             goes_lon: np.ndarray,
                             gfs_lat: np.ndarray,
                             gfs_lon: np.ndarray) -> None:
    """
    This function builds the input and conditioning interpolators for the GOES model. The input grid for the GOES model is defined by the goes_lat and goes_lon arrays, which represent the latitude and longitude of the GOES satellite observations. The conditioning grid for the GOES model is defined by the gfs_lat and gfs_lon arrays, which represent the latitude and longitude of the GFS forecast data used for conditioning. The function calls the build_input_interpolator method of the goes_model to set up interpolation from the GOES grid to the model's internal grid, and then calls build_conditioning_interpolator to set up interpolation from the GFS grid to the model's internal grid. This allows the model to take inputs and conditioning data on their native grids and interpolate them as needed during inference.

    Parameters:
        goes_model (Model): The GOES model instance for which to build the interpolators.
        goes_lat (np.ndarray): The latitude array for the GOES input grid.
        goes_lon (np.ndarray): The longitude array for the GOES input grid.
        gfs_lat (np.ndarray): The latitude array for the GFS conditioning grid.
        gfs_lon (np.ndarray): The longitude array for the GFS conditioning grid.

    Returns:
        None
    """
    goes_model.build_input_interpolator(goes_lat, goes_lon)
    goes_model.build_conditioning_interpolator(gfs_lat, gfs_lon)


def build_mrms_interpolators(mrms_model: Model,
                             mrms_lat: np.ndarray,
                             mrms_lon: np.ndarray,
                             goes_lat: np.ndarray,
                             goes_lon: np.ndarray) -> None:
    """
    This function builds the input and conditioning interpolators for the MRMS model. The input grid for the MRMS model is defined by the mrms_lat and mrms_lon arrays, which represent the latitude and longitude of the MRMS radar observations. The conditioning grid for the MRMS model is defined by the goes_lat and goes_lon arrays, which represent the latitude and longitude of the GOES satellite data used for conditioning. The function calls the build_input_interpolator method of the mrms_model to set up interpolation from the MRMS grid to the model's internal grid, and then calls build_conditioning_interpolator to set up interpolation from the GOES grid to the model's internal grid. This allows the MRMS model to take inputs and conditioning data on their native grids and interpolate them as needed during inference. 

    Parameters:
        mrms_model (Model): The MRMS model instance for which to build the interpolators.
        mrms_lat (np.ndarray): The latitude array for the MRMS input grid.
        mrms_lon (np.ndarray): The longitude array for the MRMS input grid.
        goes_lat (np.ndarray): The latitude array for the GOES conditioning grid.
        goes_lon (np.ndarray): The longitude array for the GOES conditioning grid.

    Returns:
        None
    """
    mrms_model.build_input_interpolator(mrms_lat, mrms_lon)
    mrms_model.build_conditioning_interpolator(goes_lat, goes_lon)


def init_member_state(ic_chunk: torch.Tensor,
                      coords: CoordSystem,
                      device: torch.device) -> tuple[torch.Tensor, CoordSystem]:
    """
    This function initializes the state for a chunk of ensemble members by moving the initial condition (IC) chunk to the specified device (e.g., GPU) and updating the associated coordinate system to include a new "batch" dimension that indexes the ensemble members. The IC chunk is expected to have a shape that includes a leading dimension for the ensemble members, and the function creates a new coordinate array for this "batch" dimension that ranges from 0 to the number of members in the chunk. The updated state tensor and coordinate system are returned as a tuple. 

    Parameters:
        ic_chunk (torch.Tensor): The initial condition chunk to be moved.
        coords (CoordSystem): The coordinate system associated with the IC chunk.
        device (torch.device): The device to which the IC chunk should be moved.

    Returns:
        tuple[torch.Tensor, CoordSystem]: The moved IC chunk and its updated coordinates.
    """
    state = ic_chunk.to(device)
    state_coords = type(coords)(coords)
    state_coords["batch"] = np.arange(ic_chunk.shape[0])
    return state, state_coords


def add_ensemble_dim(ic: torch.Tensor,
                     coords: CoordSystem,
                     n_members: int) -> tuple[torch.Tensor, CoordSystem]:
    """
    This function adds an ensemble dimension to the initial condition (IC) tensor and updates the associated coordinate system accordingly. If the input IC tensor has 5 dimensions (e.g., time, lead, variable, height, width), the function unsqueezes a new dimension at the front (for the ensemble members) and repeats the IC data along this new dimension for the specified number of ensemble members. The coordinate system is updated to include a new "batch" coordinate that indexes the ensemble members, and this coordinate is moved to be the first dimension in the coordinate system. The resulting IC tensor with the added ensemble dimension and its updated coordinates are returned as a tuple. 

    Parameters:
        ic (torch.Tensor): The initial condition tensor to be repeated.
        coords (CoordSystem): The coordinate system associated with the IC tensor.
        n_members (int): The number of ensemble members.

    Returns:
        tuple[torch.Tensor, CoordSystem]: The repeated IC tensor and its updated coordinates.
    """
    if ic.dim() == 5:
        ic = ic.unsqueeze(0).repeat(n_members, 1, 1, 1, 1, 1)
        coords["batch"] = np.arange(n_members)
        coords.move_to_end("batch", last=False)
    return ic.to(torch.float32), coords


class RolloutState(NamedTuple):
    """ The four tensors/coords carried between autoregressive steps. """

    goes_state: torch.Tensor
    goes_coords: CoordSystem
    mrms_state: torch.Tensor
    mrms_coords: CoordSystem


def rollout_step(goes_model: Model,
                 mrms_model: Model,
                 state: RolloutState) -> tuple[RolloutState, np.ndarray, np.ndarray, int]:
    """
    This function performs one step of the autoregressive rollout for both the GOES and MRMS models. It takes the current state of the GOES and MRMS models, which includes their respective tensors and coordinate systems, and passes them through the models to obtain predictions for the next time step. The function then applies the valid masks from each model to the predictions to create masked fields, which are converted to NumPy arrays for output. The lead time in minutes is extracted from the GOES prediction coordinates. Finally, the function computes the next input states for both models using their respective next_input methods, which take into account the current predictions and states. The updated state for the next step, along with the GOES and MRMS fields and lead time, are returned as a tuple.

    Parameters:
        goes_model (Model): The GOES model instance used for prediction.
        mrms_model (Model): The MRMS model instance used for prediction.
        state (RolloutState): The current state of the rollout, containing tensors and coordinates for both models.

    Returns:
        tuple[RolloutState, np.ndarray, np.ndarray, int]: The updated state for the next step, the GOES field, the MRMS field, and the lead time in minutes.
    """
    goes_pred, goes_pred_coords = goes_model(state.goes_state, state.goes_coords)

    mrms_pred, mrms_pred_coords = mrms_model.call_with_conditioning(
        state.mrms_state, state.mrms_coords,
        conditioning=state.goes_state, conditioning_coords=state.goes_coords,
    )

    goes_masked = torch.where(goes_model.valid_mask, goes_pred, torch.nan)
    mrms_masked = torch.where(mrms_model.valid_mask, mrms_pred, torch.nan)

    goes_field = goes_masked[:, 0, 0].detach().cpu().numpy()
    mrms_field = mrms_masked[:, 0, 0].detach().cpu().numpy()

    lead_min = int(goes_pred_coords["lead_time"][0] / np.timedelta64(1, "m"))

    next_goes, next_goes_coords = goes_model.next_input(
        goes_pred, goes_pred_coords, state.goes_state, state.goes_coords)
    
    next_mrms, next_mrms_coords = mrms_model.next_input(
        mrms_pred, mrms_pred_coords, state.mrms_state, state.mrms_coords)
        
    return (RolloutState(next_goes, next_goes_coords, next_mrms, next_mrms_coords),
            goes_field, mrms_field, lead_min)


class MapLayer(NamedTuple):
    """ One pcolormesh + colorbar overlaid on a quick-look map panel. """

    field: np.ndarray
    cmap: str
    label: str
    vmin: float | None = None
    vmax: float | None = None
    mask_nonpos: bool = False
    pad: float = 0.05


def make_map_panel(path: str,
                   title: str,
                   layers: list[MapLayer],
                   *,
                   proj: Any,
                   lon: np.ndarray,
                   lat: np.ndarray) -> None:
    """
    This function creates a quick-look map panel with the specified title and layers, and saves it to the given file path. It uses Cartopy for map projections and features, and Matplotlib for plotting. The function sets up a figure and axes with the specified projection, adds state boundaries and coastlines, and then iterates over the provided layers to plot each field as a pcolormesh on the map. Each layer can have its own colormap, label, value range (vmin/vmax), and options for masking non-positive values. A colorbar is added for each layer with the specified label. Finally, the title is set, the layout is tightened, and the figure is saved to the specified path with a resolution of 200 DPI. 

    Parameters:
        path (str): The file path to save the panel.
        title (str): The title of the panel.
        layers (list[MapLayer]): A list of MapLayer instances to render.
        proj (Any): The cartopy projection to use.
        lon (np.ndarray): The longitude coordinates.
        lat (np.ndarray): The latitude coordinates.

    Returns:
        None
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection=proj)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.6)
    ax.coastlines(color="black", linewidth=0.8)

    for layer in layers:
        masked_field = (np.where(layer.field <= 0, np.nan, layer.field)
                        if layer.mask_nonpos else layer.field)
        
        clim_kwargs: dict[str, float] = {}

        if layer.vmin is not None:
            clim_kwargs["vmin"] = layer.vmin

        if layer.vmax is not None:
            clim_kwargs["vmax"] = layer.vmax

        mesh = ax.pcolormesh(lon, lat, masked_field,
                             transform=ccrs.PlateCarree(),
                             cmap=layer.cmap, shading="auto", **clim_kwargs)
        
        plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=layer.pad,
                     shrink=0.5, label=layer.label)
        
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log.info(f"    wrote {path}")


def _rollout_worker(device_id: int,
                    args: argparse.Namespace,
                    goes_lat: np.ndarray,
                    goes_lon: np.ndarray,
                    mrms_lat: np.ndarray,
                    mrms_lon: np.ndarray,
                    goes_ic_chunk: torch.Tensor,
                    goes_coords: CoordSystem,
                    mrms_ic_chunk: torch.Tensor,
                    mrms_coords: CoordSystem,
                    out_path: str,) -> None:
    """
    This function is the worker process that performs the autoregressive rollout for a chunk of ensemble members. It sets up logging, loads the GOES and MRMS models onto the specified CUDA device, builds the necessary interpolators for the input and conditioning grids, and then iteratively rolls out the forecast for the specified number of steps. At each step, it calls the rollout_step function to get the next state and the predicted fields, which are stored in lists. The function also logs progress, including timing and memory usage, at each step. Finally, it saves the collected GOES and MRMS fields along with the lead times to a .npz file at the specified output path. 

    Parameters:
        device_id (int): The ID of the CUDA device to use.
        args (argparse.Namespace): The command-line arguments.
        goes_lat (np.ndarray): The latitude coordinates for GOES data.
        goes_lon (np.ndarray): The longitude coordinates for GOES data.
        mrms_lat (np.ndarray): The latitude coordinates for MRMS data.
        mrms_lon (np.ndarray): The longitude coordinates for MRMS data.
        goes_ic_chunk (torch.Tensor): The initial conditions for GOES data.
        goes_coords (CoordSystem): The coordinate system for GOES data.
        mrms_ic_chunk (torch.Tensor): The initial conditions for MRMS data.
        mrms_coords (CoordSystem): The coordinate system for MRMS data.
        out_path (str): The file path to save the results.

    Returns:
        None
    """
    setup_logging() 

    from earth2studio.data import GFS_FX, GOES
    from earth2studio.models.px.stormscope import (
        StormScopeBase, StormScopeGOES, StormScopeMRMS,
    )

    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    log.info(f"  [GPU{device_id}] starting — {torch.cuda.get_device_name(device_id)} — "
             f"{goes_ic_chunk.shape[0]} member(s)")

    package = StormScopeBase.load_default_package()

    goes_model = load_stormscope_model(
        StormScopeGOES, package=package, conditioning=GFS_FX(),
        model_name=args.goes_model, device=device)
    
    mrms_model = load_stormscope_model(
        StormScopeMRMS, package=package, conditioning=GOES(),
        model_name=args.mrms_model, device=device)
    
    build_goes_interpolators(goes_model, goes_lat, goes_lon,
                             GFS_FX.GFS_LAT, GFS_FX.GFS_LON)
    
    build_mrms_interpolators(mrms_model, mrms_lat, mrms_lon, goes_lat, goes_lon)

    mem_gb = torch.cuda.memory_allocated(device_id) / 1e9
    log.info(f"  [GPU{device_id}] models loaded — {mem_gb:.1f} GB — starting rollout")

    goes_state, goes_state_coords = init_member_state(
        goes_ic_chunk, goes_coords, device)
        
    mrms_state, mrms_state_coords = init_member_state(
        mrms_ic_chunk, mrms_coords, device)
    
    state = RolloutState(goes_state, goes_state_coords,
                         mrms_state, mrms_state_coords)

    goes_frames: list[np.ndarray] = []
    mrms_frames: list[np.ndarray] = []
    lead_minutes: list[int] = []

    loop_start = time.perf_counter()
    prev_step_end = loop_start

    for step in range(args.n_steps):
        state, goes_field, mrms_field, lead_min = rollout_step(
            goes_model, mrms_model, state)
        
        goes_frames.append(goes_field)
        mrms_frames.append(mrms_field)
        lead_minutes.append(lead_min)

        torch.cuda.empty_cache()

        now = time.perf_counter()
        step_secs = now - prev_step_end
        prev_step_end = now
        steps_done = step + 1
        avg_secs = (now - loop_start) / steps_done

        remaining_secs = avg_secs * (args.n_steps - steps_done)
        rem_hours = int(remaining_secs // 3600)
        rem_mins = int((remaining_secs % 3600) // 60)

        eta_delta = (f"+{rem_hours}h{rem_mins:02d}m" if rem_hours
                     else f"+{rem_mins}m")
        
        eta_clock = (datetime.now()
                     + timedelta(seconds=remaining_secs)).strftime("%H:%M:%S")
        
        gpu_gb = torch.cuda.memory_allocated(device_id) / 1e9

        log.info(
            f"  [GPU{device_id}] step {steps_done:>3}/{args.n_steps}  "
            f"lead +{lead_min:>3} min  step {step_secs:.1f}s  avg {avg_secs:.1f}s  "
            f"ETA {eta_clock} ({eta_delta})  mem {gpu_gb:.1f} GB")

    np.savez(
        out_path,
        goes=np.stack(goes_frames, axis=1),
        mrms=np.stack(mrms_frames, axis=1),
        lead=np.array(lead_minutes),
    )
    
    log.info(f"  [GPU{device_id}] done — saved {out_path}")


def fail_with_available(kind: str,
                        name: str,
                        err: Exception,
                        package: Package,
                        known: list[str]) -> NoReturn:
    """
    This function is called when loading a StormScope model fails. It logs an error message indicating that the specified model could not be loaded, along with the exception message. It then logs a list of documented checkpoint names for the specified kind of model (e.g., GOES or MRMS) to help the user identify valid options. Additionally, it attempts to list the contents of the downloaded package directory to provide further context on what models are actually available. Finally, it exits the program with a non-zero status code to indicate failure. 

    Parameters:
        kind (str): The type of model (e.g., "GOES" or "MRMS").
        name (str): The name of the model that failed to load.
        err (Exception): The exception that was raised.
        package (Package): The package containing the model.
        known (list[str]): A list of known valid model names.

    Returns:
        NoReturn: This function exits the program.
    """
    log.error(f"ERROR: could not load {kind} model_name={name!r}: {err}")
    log.error(f"Documented {kind} checkpoint names:")

    for ckpt_name in known:
        log.error(f"  - {ckpt_name}")

    for attr in ("root", "cache", "path"):
        root = getattr(package, attr, None)
        if isinstance(root, str) and os.path.isdir(root):
            log.error(f"Package contents ({attr}={root}):")
            for entry in sorted(os.listdir(root)):
                log.error(f"  {entry}")
            break
    sys.exit(2)


def setup_device(args: argparse.Namespace) -> torch.device:
    """
    This function checks for the availability of CUDA devices and sets up the random seed for reproducibility. If no CUDA device is available, it logs an error message and exits the program, as StormScope requires an NVIDIA GPU to run. If a CUDA device is available, it sets the random seed for PyTorch using the value provided in the command-line arguments and returns a torch.device object representing the CUDA device to be used for model inference. 

    Parameters:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        torch.device: The CUDA device to use.
    """
    if not torch.cuda.is_available():
        log.error("ERROR: no CUDA device. StormScope requires an NVIDIA GPU "
                  "(run on a Casper GPU node, not a login node).")
        sys.exit(1)
    torch.manual_seed(args.seed)
    return torch.device("cuda")


def load_models(package: Package,
                args: argparse.Namespace,
                device: torch.device,
                *,
                goes_cls: type,
                mrms_cls: type,
                gfs_fx: type,
                goes_src: type) -> tuple[Model, Model]:
    """
    This function loads the GOES and MRMS models specified in the command-line arguments using the load_stormscope_model helper function. It logs the progress of loading the models, including the names of the models being loaded. The GOES model is conditioned on GFS forecast data, while the MRMS model is conditioned on GOES satellite data. The loaded models are returned as a tuple. If loading either model fails, the load_stormscope_model function will handle logging the error and exiting the program. 

    Parameters:
        package (Package): The package containing the models.
        args (argparse.Namespace): The command-line arguments.
        device (torch.device): The device to use for model inference.
        goes_cls (type): The GOES model class.
        mrms_cls (type): The MRMS model class.
        gfs_fx (type): The GFS forcing class.
        goes_src (type): The GOES source class.

    Returns:
        tuple[Model, Model]: The loaded GOES and MRMS models.
    """
    log.info(f"[2/6] Loading models  GOES={args.goes_model}  MRMS={args.mrms_model}")

    goes_model = load_stormscope_model(
        goes_cls, package=package, conditioning=gfs_fx(),
        model_name=args.goes_model, device=device,
        kind="GOES", known=GOES_KNOWN)

    mrms_model = load_stormscope_model(
        mrms_cls, package=package, conditioning=goes_src(),
        model_name=args.mrms_model, device=device,
        kind="MRMS", known=MRMS_KNOWN)

    return goes_model, mrms_model


class Inputs(NamedTuple):
    """ Initial conditions + grids handed to the rollout workers. """

    goes_lat: np.ndarray
    goes_lon: np.ndarray
    mrms_lat: np.ndarray
    mrms_lon: np.ndarray
    goes_ic: torch.Tensor  
    goes_coords: CoordSystem
    mrms_ic: torch.Tensor   
    mrms_coords: CoordSystem


def prepare_inputs(args: argparse.Namespace,
                   goes_model: Model,
                   mrms_model: Model,
                   device: torch.device,
                   init_times: list[np.datetime64],
                   variables: list[str],
                   *,
                   gfs_fx: type,
                   goes_cls: type,
                   mrms_cls: type,
                   fetch_data: Callable) -> Inputs:
    """
    This function prepares the initial conditions and associated grids for the GOES and MRMS models to be used in the rollout. It first builds the necessary interpolators for both models based on their respective input and conditioning grids. Then, it fetches the initial condition data for both GOES and MRMS using the provided fetch_data function, which may involve downloading data from NOAA if not already cached. The fetched data is then processed to add an ensemble dimension by repeating the initial conditions across the specified number of ensemble members. Finally, the prepared inputs, including the latitude and longitude grids and the initial condition tensors with their coordinates, are returned as an Inputs named tuple. 

    Parameters:
        args (argparse.Namespace): The command-line arguments.
        goes_model (Model): The GOES model.
        mrms_model (Model): The MRMS model.
        device (torch.device): The device to use for model inference.
        init_times (list[np.datetime64]): The initial times for the forecast.
        variables (list[str]): The variables to fetch.
        gfs_fx (type): The GFS forcing class.
        goes_cls (type): The GOES model class.
        mrms_cls (type): The MRMS model class.
        fetch_data (Callable): The function to fetch data.

    Returns:
        Inputs: The prepared inputs for the rollout.
    """
    log.info("[3/6] Building grid interpolators")

    goes_source = goes_cls(satellite=args.goes_satellite,
                           scan_mode=args.scan_mode)
    
    goes_lat, goes_lon = goes_cls.grid(satellite=args.goes_satellite,
                                       scan_mode=args.scan_mode)
    
    mrms_source = mrms_cls()

    build_goes_interpolators(goes_model, goes_lat, goes_lon,
                             gfs_fx.GFS_LAT, gfs_fx.GFS_LON)

    goes_in_coords = goes_model.input_coords()
    mrms_in_coords = mrms_model.input_coords()

    log.info("[4/6] Fetching initial conditions (GOES, MRMS) — may download from NOAA")

    with phase("fetch GOES IC"):
        goes_ic, goes_coords = fetch_data(
            goes_source, time=init_times, variable=np.array(variables),
            lead_time=goes_in_coords["lead_time"], device=device,
        )

    with phase("fetch MRMS IC"):
        mrms_ic, mrms_coords = fetch_data(
            mrms_source, time=init_times, variable=np.array(["refc"]),
            lead_time=mrms_in_coords["lead_time"], device=device,
        )

    build_mrms_interpolators(mrms_model, mrms_coords["lat"], mrms_coords["lon"],
                             goes_lat, goes_lon)

    n_members = args.n_ensemble
    goes_ic, goes_coords = add_ensemble_dim(goes_ic, goes_coords, n_members)
    mrms_ic, mrms_coords = add_ensemble_dim(mrms_ic, mrms_coords, n_members)

    return Inputs(
        goes_lat, goes_lon,
        mrms_coords["lat"], mrms_coords["lon"],
        goes_ic.cpu(), goes_coords, mrms_ic.cpu(), mrms_coords,
    )


def run_rollout(args: argparse.Namespace,
                inputs: Inputs) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    This function runs the autoregressive rollout for the specified number of steps, using multiprocessing to parallelize across available GPUs. It divides the ensemble members into chunks that are processed by separate worker processes, each of which performs the rollout for its assigned chunk and saves the results to a temporary .npz file. After all workers have completed, the parent process collects the results from the temporary files, reassembles them in the original member order, and returns the GOES and MRMS arrays along with the lead times. The function also logs progress and handles any errors that may occur in the worker processes. 

    Parameters:
        args (argparse.Namespace): The command-line arguments.
        inputs (Inputs): The prepared inputs for the rollout.

    Returns:
        tuple[np.ndarray, np.ndarray, list[int]]: The GOES and MRMS arrays and the lead times.
    """
    n_members = args.n_ensemble
    n_gpus = min(torch.cuda.device_count(), n_members)

    log.info(f"[5/6] Rolling out {args.n_steps} steps "
             f"(~{args.n_steps * 10} min lead)  "
             f"[{n_members} members across {n_gpus} GPU(s)]")

    chunk_sizes = [(n_members // n_gpus) + (1 if i < n_members % n_gpus else 0)
                   for i in range(n_gpus)]
    
    chunk_starts = [sum(chunk_sizes[:i]) for i in range(n_gpus)]

    tmp_dir = tempfile.mkdtemp(dir=args.output_dir)
    out_paths = [os.path.join(tmp_dir, f"chunk_{i}.npz") for i in range(n_gpus)]

    mp_ctx = mp.get_context("spawn")

    def build_worker_args(i: int) -> tuple:
        """
        This helper function builds the arguments for the i-th worker process, including slicing the initial conditions for the assigned chunk of ensemble members. It returns a tuple of arguments that will be passed to the _rollout_worker function, which includes the device ID, command-line arguments, latitude and longitude grids for both GOES and MRMS, the sliced initial condition tensors and their coordinates for the assigned chunk, and the output path for saving results.

        Parameters:
            i (int): The index of the worker process (and corresponding GPU).

        Returns:
            tuple: The arguments for the worker process.
        """
        member_slice = slice(chunk_starts[i], chunk_starts[i] + chunk_sizes[i])

        return (
            i, args, inputs.goes_lat, inputs.goes_lon,
            inputs.mrms_lat, inputs.mrms_lon,
            inputs.goes_ic[member_slice], inputs.goes_coords,
            inputs.mrms_ic[member_slice], inputs.mrms_coords,
            out_paths[i],
        )

    if n_gpus == 1:
        _rollout_worker(*build_worker_args(0))
    else:
        processes = [mp_ctx.Process(target=_rollout_worker,
                                    args=build_worker_args(i))
                     for i in range(n_gpus)]
        
        for proc in processes:
            proc.start()
            
        for proc in processes:
            proc.join()

        for i, proc in enumerate(processes):
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"GPU{i} worker exited with code {proc.exitcode}")

    chunk_files = [np.load(out_paths[i]) for i in range(n_gpus)]
    goes_arr = np.concatenate([c["goes"] for c in chunk_files], axis=0) 
    mrms_arr = np.concatenate([c["mrms"] for c in chunk_files], axis=0) 
    lead_min = list(chunk_files[0]["lead"])
    shutil.rmtree(tmp_dir)
    return goes_arr, mrms_arr, lead_min


def compute_precip(args: argparse.Namespace,
                   mrms_arr: np.ndarray,
                   lead_min: list[int]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    This function computes the precipitation rate and accumulation from the MRMS reflectivity (refc) field using a Z-R relationship. If the --no-precip flag is set, it returns None for both rate and accumulation. Otherwise, it calculates the precipitation rate by applying the refc_to_rain_rate function to the MRMS reflectivity field, using the specified Z-R parameters and caps/floors for reflectivity. The precipitation accumulation is then computed as the cumulative sum of the precipitation rate multiplied by the time step in hours. The resulting precipitation rate and accumulation arrays are returned as a tuple.

    Parameters:
        args (argparse.Namespace): The command-line arguments.
        mrms_arr (np.ndarray): The MRMS reflectivity array with shape (ensemble, step, 1, H, W).
        lead_min (list[int]): The list of lead times in minutes for each step.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: The precipitation rate and accumulation arrays, or None if precipitation is not computed. 
    """
    if args.no_precip:
        return None, None
    
    step_minutes = np.diff(np.asarray(lead_min), prepend=0)

    precip_rate = refc_to_rain_rate(mrms_arr[:, :, 0], args.zr_a, args.zr_b,
                                    args.refc_cap_dbz, args.refc_floor_dbz)
    
    precip_accum = np.cumsum(
        precip_rate * (step_minutes[None, :, None, None] / 60.0), axis=1)
    
    return precip_rate, precip_accum


class Forecast(NamedTuple):
    """ Everything the Zarr/NetCDF + plot writers need. """

    goes_arr: np.ndarray
    mrms_arr: np.ndarray
    lead_min: list[int]
    valid_time: np.ndarray
    init_time: np.datetime64
    variables: list[str]
    out_lat: np.ndarray
    out_lon: np.ndarray
    precip_rate: np.ndarray | None
    precip_accum: np.ndarray | None


def write_outputs(args: argparse.Namespace,
                  time_tag: str,
                  forecast: Forecast) -> str:
    """
    This function writes the forecast results to Zarr and NetCDF files, and also generates quick-look PNG plots. It constructs an xarray Dataset from the forecast data, including the GOES and MRMS fields, coordinates for ensemble members, lead times, valid times, and spatial coordinates. The dataset is then chunked appropriately for Zarr storage and written to a .zarr file, as well as saved in NetCDF format. The function logs the output paths and returns the stem of the output files (without extension) for use in naming the plot files.

    Parameters:
        args (argparse.Namespace): The command-line arguments.
        time_tag (str): A string tag representing the initialization time for naming output files.
        forecast (Forecast): The forecast data containing the GOES and MRMS arrays, lead times, valid times, initialization time, variables, spatial coordinates, and precipitation fields.

    Returns:
        str: The stem of the output files (without extension).
    """
    log.info("[6/6] Writing Zarr + NetCDF + plots")
    import xarray as xr

    data_vars: dict[str, tuple] = {
        "goes": (("ensemble", "step", "goes_band", "y", "x"),
                 forecast.goes_arr.astype("float32")),
        "mrms_refc": (("ensemble", "step", "y", "x"),
                      forecast.mrms_arr[:, :, 0].astype("float32")),
    }

    attrs: dict[str, Any] = {
        "title": "NVIDIA StormScope ensemble nowcast (GOES + MRMS)",
        "model_goes": args.goes_model,
        "model_mrms": args.mrms_model,
        "init_time": str(forecast.init_time),
        "source": "earth2studio nvidia/stormscope-goes-mrms",
    }

    if not args.no_precip:
        data_vars["precip_rate"] = (("ensemble", "step", "y", "x"),
                                    forecast.precip_rate.astype("float32"))
        data_vars["precip_accum"] = (("ensemble", "step", "y", "x"),
                                     forecast.precip_accum.astype("float32"))
        attrs["precip_method"] = (
            "derived diagnostic (Z-R), NOT a model prognostic")
        attrs["precip_zr"] = (
            f"Z = {args.zr_a} * R^{args.zr_b} (MRMS radar-only)")
        attrs["precip_refc_cap_dbz"] = args.refc_cap_dbz

    dataset = xr.Dataset(
        data_vars,
        coords={
            "ensemble": np.arange(args.n_ensemble),
            "step": np.arange(args.n_steps),
            "goes_band": np.array(forecast.variables),
            "lead_minutes": ("step", np.array(forecast.lead_min)),
            "valid_time": ("step", forecast.valid_time),
            "lat": (("y", "x"), forecast.out_lat.astype("float32")),
            "lon": (("y", "x"), forecast.out_lon.astype("float32")),
        },
        attrs=attrs,
    )

    out_stem = os.path.join(args.output_dir, f"stormscope_{time_tag}")
    zarr_chunks = {"ensemble": 1, "step": 1, "y": -1, "x": -1}
    dataset.chunk(zarr_chunks).to_zarr(f"{out_stem}.zarr", mode="w")
    dataset.to_netcdf(f"{out_stem}.nc")
    log.info(f"    wrote {out_stem}.zarr  and  {out_stem}.nc")
    return out_stem


def write_plots(args: argparse.Namespace,
                out_stem: str,
                forecast: Forecast) -> None:
    """
    This function generates quick-look PNG plots for the GOES and MRMS fields, as well as the derived precipitation fields if applicable. It defines a helper function goes_mrms_panel to create a panel comparing the GOES and MRMS fields for a given key (e.g., ensemble mean or member 0). It then constructs a list of panels to generate, including the ensemble mean and member 0 comparisons. If precipitation is being computed, it also adds panels for the precipitation accumulation. Finally, it iterates over the defined panels and calls the make_map_panel function to create and save each plot with the appropriate title and layers. 

    Parameters:
        args (argparse.Namespace): The command-line arguments.
        out_stem (str): The stem of the output files (without extension).
        forecast (Forecast): The forecast data containing the GOES and MRMS arrays, lead times, valid times, initialization time, variables, spatial coordinates, and precipitation fields.

    Returns:
        None
    """
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")

    proj = ccrs.LambertConformal(
        central_longitude=262.5, central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )

    if args.plot_channel not in forecast.variables:
        log.warning(f"    WARNING: --plot-channel {args.plot_channel} not in model "
                    f"variables {forecast.variables}; using {forecast.variables[0]}")
        args.plot_channel = forecast.variables[0]

    band_idx = forecast.variables.index(args.plot_channel)
    goes_label = f"GOES {args.plot_channel} [K]"
    last_lead_min = forecast.lead_min[-1]

    def goes_mrms_panel(key: str,
                        label: str,
                        goes_field: np.ndarray,
                        refc_field: np.ndarray) -> tuple[str, str, list[MapLayer]]:
        """
        This helper function creates a panel comparing the GOES and MRMS fields for a given key (e.g., ensemble mean or member 0). It constructs a title for the panel that includes the label, initialization time, and lead time. It then returns a tuple containing the output file path for the panel, the title, and a list of MapLayer objects representing the GOES field and the MRMS reflectivity field. The GOES field is plotted with a grayscale colormap, while the MRMS reflectivity is plotted with an inferno colormap and includes a colorbar label indicating that it represents reflectivity in dBZ. The MRMS layer also has specified value limits and options for masking non-positive values. 

        Parameters:
            key (str): The key identifying the panel (e.g., "ensmean" or "member00").
            label (str): The label for the panel (e.g., "ensemble mean" or "member 0").
            goes_field (np.ndarray): The GOES field data.
            refc_field (np.ndarray): The MRMS reflectivity field data.

        Returns:
            tuple[str, str, list[MapLayer]]: A tuple containing the output file path, title, and list of MapLayer objects.
        """
        title = (f"StormScope {label}  init {forecast.init_time} UTC  "
                 f"lead +{last_lead_min} min")
        return (f"{out_stem}_{key}.png", title, [
            MapLayer(goes_field, "gray_r", goes_label),
            MapLayer(refc_field, "inferno", "MRMS reflectivity [dBZ]",
                     vmin=0.0, vmax=55.0, mask_nonpos=True, pad=0.10),
        ])

    panels: list[tuple[str, str, list[MapLayer]]] = [
        goes_mrms_panel("ensmean", "ensemble mean",
                        forecast.goes_arr[:, -1, band_idx].mean(0),
                        np.nanmean(forecast.mrms_arr[:, -1, 0], axis=0)),
        goes_mrms_panel("member00", "member 0",
                        forecast.goes_arr[0, -1, band_idx],
                        forecast.mrms_arr[0, -1, 0]),
    ]

    if not args.no_precip:
        def precip_panel(key: str,
                         label: str,
                         accum_field: np.ndarray) -> tuple[str, str, list[MapLayer]]:
            title = (f"StormScope {label}  init {forecast.init_time} UTC  "
                     f"+{last_lead_min} min storm total\n"
                     f"Z-R diagnostic (Z={args.zr_a}*R^{args.zr_b}), "
                     f"NOT a model prognostic")
            return (f"{out_stem}_{key}_precip.png", title, [
                MapLayer(accum_field, "viridis",
                         "derived precip accumulation [mm]",
                         mask_nonpos=True),
            ])

        panels += [
            precip_panel("ensmean", "ensemble mean",
                         np.nanmean(forecast.precip_accum[:, -1], axis=0)),
            precip_panel("member00", "member 0",
                         forecast.precip_accum[0, -1]),
        ]

    for path, title, layers in panels:
        make_map_panel(path, title, layers, proj=proj,
                       lon=forecast.out_lon, lat=forecast.out_lat)


def main() -> None:
    """
    This is the main function that orchestrates the entire workflow of the StormScope nowcasting pipeline. It starts by parsing command-line arguments and setting up logging. It then creates the output directory if it doesn't exist and sets up the device for model inference. The function initializes the initial times for the forecast based on the provided date argument and generates a time tag for naming outputs. It imports the necessary data sources and model classes, loads the StormScope models, prepares the inputs by fetching initial condition data, and then runs the autoregressive rollout using multiprocessing across available GPUs. After the rollout is complete, it computes precipitation fields if applicable, constructs a Forecast named tuple with all relevant data, writes the outputs to Zarr and NetCDF files, and generates quick-look plots. Finally, it logs the total execution time and the time spent in each phase of the pipeline.

    Parameters:
        None

    Returns:
        None
    """
    args = parse_args()
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    device = setup_device(args)

    init_times = [np.datetime64(datetime.fromisoformat(args.date))]
    time_tag = np.datetime_as_string(init_times[0], unit="m").replace(
        ":", "").replace("-", "")

    from earth2studio.data import GFS_FX, GOES, MRMS, fetch_data

    from earth2studio.models.px.stormscope import (
        StormScopeBase, StormScopeGOES, StormScopeMRMS,
    )

    log.info(f"[0/6] Data cache: {os.environ['EARTH2STUDIO_CACHE']}")

    log.info(f"[1/6] Loading StormScope package (date={args.date}, "
             f"ensemble={args.n_ensemble}, steps={args.n_steps})")
    
    with phase("package load"):
        package = StormScopeBase.load_default_package()

    with phase("model load"):
        goes_model, mrms_model = load_models(
            package, args, device,
            goes_cls=StormScopeGOES, mrms_cls=StormScopeMRMS,
            gfs_fx=GFS_FX, goes_src=GOES)

    out_lat = goes_model.latitudes.detach().cpu().numpy()
    out_lon = goes_model.longitudes.detach().cpu().numpy()
    variables = list(np.array(goes_model.input_coords()["variable"]))

    with phase("prepare inputs"):
        inputs = prepare_inputs(
            args, goes_model, mrms_model, device, init_times, variables,
            gfs_fx=GFS_FX, goes_cls=GOES, mrms_cls=MRMS, fetch_data=fetch_data)

    del goes_model, mrms_model
    torch.cuda.empty_cache()

    with phase("rollout"):
        goes_arr, mrms_arr, lead_min = run_rollout(args, inputs)

    init_time = init_times[0]

    valid_time = np.array(
        [init_time + np.timedelta64(m, "m") for m in lead_min])
    
    precip_rate, precip_accum = compute_precip(args, mrms_arr, lead_min)

    forecast = Forecast(goes_arr, mrms_arr, lead_min, valid_time, init_time,
                        variables, out_lat, out_lon, precip_rate, precip_accum)
    
    with phase("write outputs"):
        out_stem = write_outputs(args, time_tag, forecast)

    with phase("write plots"):
        write_plots(args, out_stem, forecast)

    log.info(f"Done — total {_fmt_hms(time.time() - START)}")

    for label, duration in PHASE_TIMES:
        log.info(f"  {label:<16} {duration:8.1f}s")


if __name__ == "__main__":
    main()

