"""GFS data download and GRIB-to-channel processing utilities."""

import os
import urllib.parse
from datetime import datetime, timedelta, timezone

import cfgrib
import numpy as np
import requests
import xarray as xr


INPUT_VARS = [
    "u10m", "v10m", "t2m", "tcwv", "sp", "msl",
    "u1000", "u850", "u500", "u250",
    "v1000", "v850", "v500", "v250",
    "z1000", "z850", "z500", "z250",
    "t1000", "t850", "t500", "t250",
    "q1000", "q850", "q500", "q250",
]

SURFACE_FIELDS = {
    "u10m": ["u10", "u10m"],
    "v10m": ["v10", "v10m"],
    "t2m": ["t2m", "2t"],
    "tcwv": ["pwat", "tcwv"],
    "sp": ["sp", "pres"],
    "msl": ["msl", "prmsl"],
}

PRESSURE_LEVELS = [1000, 850, 500, 250]

UPSAMPLE_FACTOR = 8


def compute_domain(center_lat, center_lon, hr_grid, target_res_km, download_pad=0.5):
    """Derive bounding boxes from a center point and grid parameters.

    Returns ``(region_bbox, download_bbox, domain_km)`` where each bbox is a
    dict with keys ``leftlon, rightlon, toplat, bottomlat``.
    """
    domain_km = target_res_km * hr_grid
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))
    lat_span = domain_km / km_per_deg_lat
    lon_span = domain_km / km_per_deg_lon

    region_bbox = {
        "leftlon": center_lon - lon_span / 2.0,
        "rightlon": center_lon + lon_span / 2.0,
        "toplat": center_lat + lat_span / 2.0,
        "bottomlat": center_lat - lat_span / 2.0,
    }
    download_bbox = {
        "leftlon": region_bbox["leftlon"] - download_pad,
        "rightlon": region_bbox["rightlon"] + download_pad,
        "toplat": region_bbox["toplat"] + download_pad,
        "bottomlat": region_bbox["bottomlat"] - download_pad,
    }
    return region_bbox, download_bbox, domain_km


# -- NOMADS download helpers -------------------------------------------------

def _build_nomads_filter_url(date_yyyymmdd, cycle_hh, endpoint, file_name,
                             download_bbox):
    base = f"https://nomads.ncep.noaa.gov/cgi-bin/{endpoint}"
    params = {
        "file": file_name,
        "lev_10_m_above_ground": "on",
        "lev_2_m_above_ground": "on",
        "lev_surface": "on",
        "lev_mean_sea_level": "on",
        "lev_entire_atmosphere_(considered_as_a_single_layer)": "on",
        "lev_1000_mb": "on",
        "lev_850_mb": "on",
        "lev_500_mb": "on",
        "lev_250_mb": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "var_TMP": "on",
        "var_SPFH": "on",
        "var_HGT": "on",
        "var_PRES": "on",
        "var_PRMSL": "on",
        "var_PWAT": "on",
        "var_LAND": "on",
        "subregion": "",
        "leftlon": str(download_bbox["leftlon"]),
        "rightlon": str(download_bbox["rightlon"]),
        "toplat": str(download_bbox["toplat"]),
        "bottomlat": str(download_bbox["bottomlat"]),
        "dir": f"/gfs.{date_yyyymmdd}/{cycle_hh}/atmos",
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def _download_grib(url, out_path, min_size_mb=0.1):
    tmp = f"{out_path}.tmp"
    try:
        r = requests.get(
            url, stream=True, timeout=300, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return False
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        size_mb = os.path.getsize(tmp) / 1e6
        with open(tmp, "rb") as f:
            magic = f.read(4)
        if magic != b"GRIB":
            print("  Not a valid GRIB file")
            os.remove(tmp)
            return False
        if size_mb < min_size_mb:
            print(f"  File too small ({size_mb:.3f} MB < {min_size_mb} MB)")
            os.remove(tmp)
            return False
        os.replace(tmp, out_path)
        print(f"  OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def _download_fhour(date_str, cycle, fhour, out_path, download_bbox):
    """Download a single GFS forecast lead hour."""
    fname = f"gfs.t{cycle}z.pgrb2.0p25.f{fhour:03d}"
    for endpoint in ["filter_gfs_0p25_1hr.pl", "filter_gfs_0p25.pl"]:
        url = _build_nomads_filter_url(
            date_str, cycle, endpoint, fname, download_bbox
        )
        if _download_grib(url, out_path, min_size_mb=0.001):
            return True
    direct = (
        f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        f"gfs.{date_str}/{cycle}/atmos/{fname}"
    )
    if _download_grib(direct, out_path, min_size_mb=5.0):
        return True
    return False


def download_gfs_series(region_center, n_hours, step_hours, data_dir,
                        hr_grid=512, target_res_km=3.0, max_days_back=3):
    """Download a multi-hour GFS forecast series for *region_center*.

    Returns ``(date_str, cycle, [grib_path, ...], forecast_hours)``.
    """
    center_lat, center_lon = region_center
    _, download_bbox, _ = compute_domain(
        center_lat, center_lon, hr_grid, target_res_km
    )
    forecast_hours = [i * step_hours for i in range(n_hours)]
    out_dir = os.path.join(data_dir, "gfs_forecast_series")
    os.makedirs(out_dir, exist_ok=True)

    now = datetime.now(timezone.utc)
    for day_offset in range(max_days_back + 1):
        date_str = (now - timedelta(days=day_offset)).strftime("%Y%m%d")
        for cycle in ["18", "12", "06", "00"]:
            paths = []
            ok = True
            for fhour in forecast_hours:
                out_path = os.path.join(
                    out_dir, f"gfs_{date_str}_t{cycle}z_f{fhour:03d}.grib2"
                )
                if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                    with open(out_path, "rb") as fh:
                        cached_ok = fh.read(4) == b"GRIB"
                    if cached_ok:
                        print(f"{date_str} {cycle}z  f{fhour:03d}  (cached)")
                        paths.append(out_path)
                        continue
                print(f"{date_str} {cycle}z  f{fhour:03d}")
                if _download_fhour(date_str, cycle, fhour, out_path, download_bbox):
                    paths.append(out_path)
                else:
                    ok = False
                    break
            if ok:
                print(f"Got full {len(paths)}-frame series from {date_str} {cycle}z")
                return date_str, cycle, paths, forecast_hours
    raise RuntimeError(
        "Could not download a complete GFS forecast series. Try rerunning later "
        "or reduce n_hours / step_hours."
    )


# -- GRIB field extraction ---------------------------------------------------

def _get_first_var(datasets, names, required_dim=None, exclude_dim=None):
    for name in names:
        for ds in datasets:
            if name in ds.data_vars:
                if required_dim is not None and required_dim not in ds.dims:
                    continue
                if exclude_dim is not None and exclude_dim in ds.dims:
                    continue
                return ds[name]
    raise KeyError(f"Could not find {names}")


def _interpolate_to_grid(da, lat_target, lon_target):
    d = da
    for dim in [
        "time", "step", "heightAboveGround", "surface",
        "meanSea", "entireAtmosphere", "valid_time",
    ]:
        if dim in d.dims and d.sizes[dim] == 1:
            d = d.isel({dim: 0})

    lat_name = "latitude" if "latitude" in d.coords else "lat"
    lon_name = "longitude" if "longitude" in d.coords else "lon"

    lon_vals = (((d[lon_name] + 180) % 360) - 180).values
    d = d.assign_coords({lon_name: lon_vals}).sortby(lon_name)
    d = d.sortby(lat_name, ascending=False)

    patch = d.interp(
        {lat_name: lat_target, lon_name: lon_target},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return np.asarray(patch.values, dtype=np.float32)


def _extract_valid_time(groups):
    for ds in groups:
        for coord in ("valid_time", "time"):
            if coord in ds.coords:
                t = ds.coords[coord].values
                return np.datetime64(
                    t.flat[0] if isinstance(t, np.ndarray) else t, "s"
                )
    return np.datetime64("now", "s")


def grib_to_channels(grib_path, region_bbox, hr_grid):
    """Extract and interpolate one GRIB file onto the low-res input grid.

    Returns ``(channel_dict, valid_time)`` where *channel_dict* maps every
    name in ``INPUT_VARS`` to a ``(lr_grid, lr_grid)`` float32 array.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")

    lr_grid = hr_grid // UPSAMPLE_FACTOR
    lat_target = np.linspace(
        region_bbox["toplat"], region_bbox["bottomlat"], lr_grid
    )
    lon_target = np.linspace(
        region_bbox["leftlon"], region_bbox["rightlon"], lr_grid
    )

    groups = cfgrib.open_datasets(grib_path, backend_kwargs={"indexpath": ""})

    pl_fields = {
        "u": _get_first_var(groups, ["u"], required_dim="isobaricInhPa"),
        "v": _get_first_var(groups, ["v"], required_dim="isobaricInhPa"),
        "t": _get_first_var(groups, ["t"], required_dim="isobaricInhPa"),
        "q": _get_first_var(groups, ["q"], required_dim="isobaricInhPa"),
    }
    gh_pl = _get_first_var(groups, ["gh", "z"], required_dim="isobaricInhPa")

    channels = {}
    for channel, aliases in SURFACE_FIELDS.items():
        channels[channel] = _interpolate_to_grid(
            _get_first_var(groups, aliases), lat_target, lon_target
        )
    for lev in PRESSURE_LEVELS:
        for prefix, da in pl_fields.items():
            channels[f"{prefix}{lev}"] = _interpolate_to_grid(
                da.sel(isobaricInhPa=lev), lat_target, lon_target
            )
        channels[f"z{lev}"] = _interpolate_to_grid(
            gh_pl.sel(isobaricInhPa=lev) * 9.80665, lat_target, lon_target
        )

    return channels, _extract_valid_time(groups)
