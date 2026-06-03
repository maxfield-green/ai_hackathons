"""Visualization helpers for CorrDiff regional transfer experiments."""

import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
from IPython.display import Image, display

from .gfs import UPSAMPLE_FACTOR, compute_domain

MAJOR_CITIES = {
    "Bangkok": (13.7563, 100.5018),
    "Chiang Mai": (18.7883, 98.9853),
    "Phuket": (7.8804, 98.3923),
    "Pattaya": (12.9236, 100.8825),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Singapore": (1.3521, 103.8198),
    "Yangon": (16.8661, 96.1951),
    "Ho Chi Minh City": (10.8231, 106.6297),
    "Hanoi": (21.0285, 105.8542),
    "Phnom Penh": (11.5564, 104.9282),
    "Vientiane": (17.9757, 102.6331),
    "Tokyo": (35.6762, 139.6503),
    "Seoul": (37.5665, 126.9780),
    "Taipei": (25.0330, 121.5654),
    "Manila": (14.5995, 120.9842),
    "Jakarta": (-6.2088, 106.8456),
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Dubai": (25.2048, 55.2708),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Rome": (41.9028, 12.4964),
    "Madrid": (40.4168, -3.7038),
    "Moscow": (55.7558, 37.6173),
    "Istanbul": (41.0082, 28.9784),
    "Cairo": (30.0444, 31.2357),
    "Lagos": (6.5244, 3.3792),
    "Nairobi": (-1.2921, 36.8219),
    "Cape Town": (-33.9249, 18.4241),
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Mexico City": (19.4326, -99.1332),
    "São Paulo": (-23.5505, -46.6333),
    "Buenos Aires": (-34.6037, -58.3816),
    "Lima": (-12.0464, -77.0428),
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Auckland": (-36.8485, 174.7633),
    "Zagreb": (45.8150, 15.9819),
}


def _cities_in_bbox(region_bbox, pad_deg=0.35):
    """Return major cities that fall within (or near) *region_bbox*."""
    lat_min = region_bbox["bottomlat"] - pad_deg
    lat_max = region_bbox["toplat"] + pad_deg
    lon_min = region_bbox["leftlon"] - pad_deg
    lon_max = region_bbox["rightlon"] + pad_deg
    return {
        name: (lat, lon)
        for name, (lat, lon) in MAJOR_CITIES.items()
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    }


def _coarse_lat_lon_grid(region_bbox, n):
    lat_1d = np.linspace(region_bbox["toplat"], region_bbox["bottomlat"], n)
    lon_1d = np.linspace(region_bbox["leftlon"], region_bbox["rightlon"], n)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    return lat_2d.astype(np.float32), lon_2d.astype(np.float32)


def _upsample_native(field, hr_grid, corrdiff_dir):
    """Bilinear upsample matching HRRRMiniDataset._zoom_extrapolate."""
    if corrdiff_dir not in sys.path:
        sys.path.insert(0, corrdiff_dir)
    from datasets.hrrrmini import _zoom_extrapolate

    factor = hr_grid // field.shape[0]
    x = field[np.newaxis, ...].astype(np.float32)
    y = np.empty((1, hr_grid, hr_grid), dtype=np.float32)
    _zoom_extrapolate(x, y, factor)
    return y[0]


def _plot_map_panel(ax, field, lat_2d, lon_2d, title, region_bbox,
                    cmap="RdBu_r", vmin=None, vmax=None, add_cities=False):
    mesh = ax.pcolormesh(
        lon_2d, lat_2d, field,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    lon_min, lon_max = float(lon_2d.min()), float(lon_2d.max())
    lat_min, lat_max = float(lat_2d.min()), float(lat_2d.max())
    pad_lon = max(0.05, (lon_max - lon_min) * 0.06)
    pad_lat = max(0.05, (lat_max - lat_min) * 0.06)
    ax.set_extent(
        [lon_min - pad_lon, lon_max + pad_lon,
         lat_min - pad_lat, lat_max + pad_lat],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.45, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title)

    if add_cities:
        for name, (city_lat, city_lon) in _cities_in_bbox(region_bbox).items():
            ax.plot(
                city_lon, city_lat,
                marker="o", color="black", markersize=4,
                transform=ccrs.PlateCarree(), zorder=5,
            )
            ax.text(
                city_lon + 0.03, city_lat, name,
                transform=ccrs.PlateCarree(), fontsize=8,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, lw=0),
                zorder=6,
            )
    return mesh


def build_forecast_animation(
    custom_file, output_nc, hr_grid, target_res_km,
    region_center, data_dir, corrdiff_dir,
):
    """Build and display an animated GIF comparing GFS input vs predictions."""
    center_lat, center_lon = region_center
    region_bbox, _, _ = compute_domain(center_lat, center_lon, hr_grid, target_res_km)
    lr_grid = hr_grid // UPSAMPLE_FACTOR

    if not os.path.exists(output_nc):
        raise RuntimeError(
            "Regional GFS inference output not found. Run inference first."
        )
    if not os.path.exists(custom_file):
        raise RuntimeError(
            "Custom GFS input file not found. Run the data-prep step first."
        )

    plot_vars = ["10u", "10v"]
    input_var_map = {"10u": "u10m", "10v": "v10m"}

    with xr.open_dataset(custom_file, group="input") as ds_native:
        native_times = np.array(ds_native["time"].values, dtype="datetime64[s]")
        native_inputs = {
            var: ds_native[input_var_map[var]].values.astype(np.float32)
            for var in plot_vars
        }

    with xr.open_dataset(custom_file, group="invariant") as ds_inv:
        lat_hr = ds_inv["latitude"].values
        lon_hr = ds_inv["longitude"].values

    lat_lr, lon_lr = _coarse_lat_lon_grid(region_bbox, lr_grid)

    ds_gfs = nc.Dataset(output_nc, "r")
    pred_group = ds_gfs.groups["prediction"]
    n_pred_time_dim = pred_group[plot_vars[0]].shape[1]
    n_frames = int(min(len(native_times), n_pred_time_dim))
    print(f"Animating {n_frames} forecast frame(s)")

    # Precompute fields and fixed per-variable color scales
    frames_data = {}
    vlims = {}
    for var in plot_vars:
        per_frame = []
        lo, hi = np.inf, -np.inf
        for t in range(n_frames):
            native = native_inputs[var][t]
            model_input = _upsample_native(native, hr_grid, corrdiff_dir)
            pred = np.array(pred_group[var][0, t])
            per_frame.append((native, model_input, pred))
            lo = min(lo, float(np.nanmin(native)),
                     float(np.nanmin(model_input)), float(np.nanmin(pred)))
            hi = max(hi, float(np.nanmax(native)),
                     float(np.nanmax(model_input)), float(np.nanmax(pred)))
        frames_data[var] = per_frame
        vlims[var] = (lo, hi)
    ds_gfs.close()

    # Build figure
    fig, axes = plt.subplots(
        len(plot_vars), 3,
        figsize=(18, 5.5 * len(plot_vars)),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    if len(plot_vars) == 1:
        axes = np.array([axes])

    panel_meshes = []
    for i, var in enumerate(plot_vars):
        vmin, vmax = vlims[var]
        native0, model_input0, pred0 = frames_data[var][0]
        panels = [
            (native0, lat_lr, lon_lr,
             f"Native GFS Input ({lr_grid}x{lr_grid}, "
             f"~{target_res_km * UPSAMPLE_FACTOR:.0f} km): {input_var_map[var]}"),
            (model_input0, lat_hr, lon_hr,
             f"Model Input ({hr_grid}x{hr_grid} upsampled): {input_var_map[var]}"),
            (pred0, lat_hr, lon_hr,
             f"CorrDiff Prediction ({hr_grid}x{hr_grid}, "
             f"~{target_res_km:.0f} km): {var}"),
        ]
        row_meshes = []
        for j, (field, lat_2d, lon_2d, title) in enumerate(panels):
            mesh = _plot_map_panel(
                axes[i, j], field, lat_2d, lon_2d, title, region_bbox,
                cmap="RdBu_r", vmin=vmin, vmax=vmax, add_cities=True,
            )
            plt.colorbar(mesh, ax=axes[i, j], shrink=0.85, pad=0.02)
            row_meshes.append(mesh)
        panel_meshes.append(row_meshes)

    suptitle = fig.suptitle("", fontsize=15)

    def _update(t):
        for i, var in enumerate(plot_vars):
            native, model_input, pred = frames_data[var][t]
            panel_meshes[i][0].set_array(native.ravel())
            panel_meshes[i][1].set_array(model_input.ravel())
            panel_meshes[i][2].set_array(pred.ravel())
        suptitle.set_text(
            f"Regional Downscaling — forecast frame {t + 1}/{n_frames} "
            f"(valid {native_times[t]})\n"
            "Native Input -> Model Input -> Prediction"
        )
        return [m for row in panel_meshes for m in row] + [suptitle]

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=700, blit=False
    )
    gif_path = os.path.join(data_dir, "gfs_region_forecast.gif")
    anim.save(gif_path, writer=animation.PillowWriter(fps=2))
    plt.close(fig)

    print(f"Saved forecast animation: {gif_path}")
    display(Image(filename=gif_path))


def plot_input_channels(custom_file):
    """Plot all 26 low-res input channels and 4 invariant fields."""
    input_ds = xr.open_dataset(custom_file, group="input")
    invar_ds = xr.open_dataset(custom_file, group="invariant")

    input_groups = {
        "Surface": ["u10m", "v10m", "t2m", "tcwv", "sp", "msl"],
        "U-wind (hPa)": ["u1000", "u850", "u500", "u250"],
        "V-wind (hPa)": ["v1000", "v850", "v500", "v250"],
        "Geopotential (hPa)": ["z1000", "z850", "z500", "z250"],
        "Temperature (hPa)": ["t1000", "t850", "t500", "t250"],
        "Specific humidity": ["q1000", "q850", "q500", "q250"],
    }
    flat_vars = [v for grp in input_groups.values() for v in grp]

    ncols = 6
    nrows = int(np.ceil(len(flat_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8))
    axes_flat = axes.flatten()

    for idx, var in enumerate(flat_vars):
        ax = axes_flat[idx]
        data = input_ds[var].isel(time=0).values
        im = ax.imshow(data, origin="upper", cmap="viridis")
        ax.set_title(var, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(flat_vars), len(axes_flat)):
        axes_flat[idx].axis("off")

    lr_grid = input_ds[flat_vars[0]].isel(time=0).shape[0]
    fig.suptitle(
        f"Model Inputs — Regional GFS ({lr_grid}x{lr_grid})",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    # Invariant fields
    invar_names = ["latitude", "longitude", "elev_mean", "lsm_mean"]
    cmaps = ["coolwarm", "coolwarm", "terrain", "BrBG"]
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, name, cmap in zip(axes2, invar_names, cmaps):
        data = invar_ds[name].values
        im = ax.imshow(data, origin="upper", cmap=cmap)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    hr_grid = invar_ds["latitude"].shape[0]
    fig2.suptitle(
        f"Invariant Fields — Regional GFS ({hr_grid}x{hr_grid})",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.show()

    input_ds.close()
    invar_ds.close()
