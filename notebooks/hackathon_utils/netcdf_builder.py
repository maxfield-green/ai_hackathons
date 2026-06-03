"""Build CorrDiff-compatible grouped NetCDF from GFS GRIB files."""

import os
import tarfile
import urllib.request

import numpy as np
import xarray as xr

from .gfs import (
    INPUT_VARS,
    UPSAMPLE_FACTOR,
    compute_domain,
    grib_to_channels,
)

INVARIANTS_URL = (
    "https://raw.githubusercontent.com/maxfield-green/ai_hackathons/"
    "main/invariants.tar.gz"
)
OUTPUT_VARS = ["10u", "10v"]


def download_invariants(corrdiff_dir):
    """Download topo.nc and land_fraction.nc if not already present."""
    dest = os.path.join(corrdiff_dir, "data")
    archive = os.path.join(dest, "invariants.tar.gz")
    os.makedirs(dest, exist_ok=True)

    topo_path = os.path.join(dest, "topo.nc")
    lsm_path = os.path.join(dest, "land_fraction.nc")

    if os.path.exists(topo_path) and os.path.exists(lsm_path):
        print("Invariant files already present — skipping download.")
        return topo_path, lsm_path

    print("Downloading invariants archive from GitHub...")
    urllib.request.urlretrieve(INVARIANTS_URL, archive)
    print(f"Extracting to {dest}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)
    os.remove(archive)
    print(f"  topo.nc:          {os.path.getsize(topo_path) / 1e6:.1f} MB")
    print(f"  land_fraction.nc: {os.path.getsize(lsm_path) / 1e6:.1f} MB")
    return topo_path, lsm_path


def build_regional_netcdf(
    grib_files,
    forecast_hours,
    region_center,
    hr_grid,
    target_res_km,
    corrdiff_dir,
    output_path,
):
    """Convert downloaded GRIB files into a single grouped NetCDF.

    Builds the ``input``, ``output``, ``invariant``, and root groups expected
    by the CorrDiff dataset loader.  Returns a list of ISO-formatted valid-time
    strings (one per forecast frame).
    """
    center_lat, center_lon = region_center
    region_bbox, _, domain_km = compute_domain(
        center_lat, center_lon, hr_grid, target_res_km
    )
    lr_grid = hr_grid // UPSAMPLE_FACTOR

    # -- Process each GRIB into input channels --------------------------------
    custom_times = []
    input_stacks = {var: [] for var in INPUT_VARS}
    for fhour, grib_path in zip(forecast_hours, grib_files):
        channels, valid_time = grib_to_channels(grib_path, region_bbox, hr_grid)
        custom_times.append(np.datetime64(valid_time, "s"))
        for var in INPUT_VARS:
            input_stacks[var].append(channels[var])
        print(f"  f{fhour:03d}  valid {valid_time}")

    n_times = len(custom_times)

    # -- Build xarray datasets ------------------------------------------------
    input_ds = xr.Dataset(
        {
            var: (
                ("time", "y", "x"),
                np.stack(input_stacks[var], axis=0).astype(np.float32),
            )
            for var in INPUT_VARS
        },
        coords={
            "time": custom_times,
            "y": np.arange(lr_grid),
            "x": np.arange(lr_grid),
        },
    )

    output_ds = xr.Dataset(
        {
            var: (
                ("time", "y", "x"),
                np.zeros((n_times, hr_grid, hr_grid), dtype=np.float32),
            )
            for var in OUTPUT_VARS
        },
        coords={
            "time": custom_times,
            "y": np.arange(hr_grid),
            "x": np.arange(hr_grid),
        },
    )

    # -- Invariant fields (lat, lon, elevation, land-sea mask) ----------------
    lat_hr = np.linspace(region_bbox["toplat"], region_bbox["bottomlat"], hr_grid)
    lon_hr = np.linspace(region_bbox["leftlon"], region_bbox["rightlon"], hr_grid)
    lon_grid, lat_grid = np.meshgrid(lon_hr, lat_hr)

    topo_path, lsm_path = download_invariants(corrdiff_dir)

    sel_pad = 1.0
    lat_sel_lo = float(min(lat_hr)) - sel_pad
    lat_sel_hi = float(max(lat_hr)) + sel_pad
    lon_sel_lo = float(min(lon_hr)) - sel_pad
    lon_sel_hi = float(max(lon_hr)) + sel_pad

    with xr.open_dataset(topo_path) as ds_topo:
        elev_hr = (
            ds_topo["z"]
            .sel(lat=slice(lat_sel_lo, lat_sel_hi), lon=slice(lon_sel_lo, lon_sel_hi))
            .interp(lat=lat_hr, lon=lon_hr, method="linear")
            .values.astype(np.float32)
        )
    np.nan_to_num(elev_hr, copy=False, nan=0.0)
    print(f"Loaded HR elevation  -> regridded to ({hr_grid}, {hr_grid})")

    with xr.open_dataset(lsm_path) as ds_lsm:
        lsm_hr = (
            ds_lsm["land_fraction"]
            .sel(lat=slice(lat_sel_lo, lat_sel_hi), lon=slice(lon_sel_lo, lon_sel_hi))
            .interp(lat=lat_hr, lon=lon_hr, method="linear")
            .values.astype(np.float32)
        )
    np.nan_to_num(lsm_hr, copy=False, nan=0.0)
    print(f"Loaded HR land fraction -> regridded to ({hr_grid}, {hr_grid})")

    invariant_ds = xr.Dataset(
        {
            "latitude": (("y", "x"), lat_grid.astype(np.float32)),
            "longitude": (("y", "x"), lon_grid.astype(np.float32)),
            "elev_mean": (("y", "x"), elev_hr),
            "lsm_mean": (("y", "x"), lsm_hr),
        }
    )

    # -- Write grouped NetCDF -------------------------------------------------
    root_ds = xr.Dataset(
        {"coord": (("time", "ij"), np.zeros((n_times, 2), dtype=np.int64))},
        coords={"time": custom_times},
    )
    root_ds.to_netcdf(output_path, mode="w")
    input_ds.to_netcdf(output_path, mode="a", group="input")
    output_ds.to_netcdf(output_path, mode="a", group="output")
    invariant_ds.to_netcdf(output_path, mode="a", group="invariant")
    print(f"Wrote: {output_path}")

    # -- NaN sanity check -----------------------------------------------------
    for grp_name in ["input", "output", "invariant"]:
        with xr.open_dataset(output_path, group=grp_name) as ds_check:
            for v in ds_check.data_vars:
                n_nan = int(np.isnan(ds_check[v].values).sum())
                if n_nan > 0:
                    print(f"  WARNING: {grp_name}/{v} has {n_nan} NaN values!")
    print("NaN check complete.")

    return [str(t) for t in custom_times]
