"""Hackathon utility package — keeps the notebook clean."""

from .gfs import (
    INPUT_VARS,
    compute_domain,
    download_gfs_series,
    grib_to_channels,
)
from .inference import (
    download_pretrained_checkpoint,
    patch_song_unet,
    run_regional_inference,
    run_with_conda_env,
)
from .netcdf_builder import (
    build_regional_netcdf,
    download_invariants,
)
from .visualization import (
    build_forecast_animation,
    plot_input_channels,
)

__all__ = [
    "INPUT_VARS",
    "build_forecast_animation",
    "build_regional_netcdf",
    "compute_domain",
    "download_gfs_series",
    "download_invariants",
    "download_pretrained_checkpoint",
    "grib_to_channels",
    "patch_song_unet",
    "plot_input_channels",
    "run_regional_inference",
    "run_with_conda_env",
]
