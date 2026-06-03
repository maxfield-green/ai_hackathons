"""Inference helpers: conda wrappers, song_unet patching, checkpoint download."""

import os
import pathlib
import shutil
import subprocess
import urllib.request


def patch_song_unet():
    """Patch physicsnemo's song_unet.py so pos_embd is interpolated to match
    arbitrary input resolutions (needed for non-training-size grids)."""
    import physicsnemo

    base = pathlib.Path(physicsnemo.__file__).parent
    path = base / "models" / "diffusion_unets" / "song_unet.py"

    old = "                selected_embd.append(pos_embd[None].expand((x.shape[0], -1, -1, -1)))"
    new = """
                _pe = pos_embd[None].expand((x.shape[0], -1, -1, -1))
                if _pe.shape[-2:] != x.shape[-2:]:
                    _pe = torch.nn.functional.interpolate(
                        _pe, size=x.shape[-2:], mode="bilinear", align_corners=False
                    )
                selected_embd.append(_pe)"""

    if not path.exists():
        print(f"WARNING: song_unet.py not found at {path}. Skipping patch.")
        return

    src = path.read_text()
    if old in src:
        path.write_text(src.replace(old, new))
        print("Patched song_unet.py: pos_embd will be interpolated to match input resolution.")
    elif (
        "interpolate"
        in src.split("selected_embd.append(")[0].split("if global_index is None:")[-1]
    ):
        print("song_unet.py already patched.")
    else:
        print("WARNING: Could not find expected snippet to patch. Check song_unet.py manually.")


def download_pretrained_checkpoint(corrdiff_dir):
    """Download the 2M-sample pretrained regression checkpoint.

    Returns the local path to the checkpoint file.
    """
    name = "CorrDiffRegressionUNet.0.2000000.mdlus"
    url = f"https://github.com/maxfield-green/ai_hackathons/raw/main/ckpts/{name}"

    ckpt_dir = os.path.join(corrdiff_dir, "checkpoints_regression")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, name)

    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1_000_000:
        print(f"Downloading pretrained checkpoint from {url} ...")
        urllib.request.urlretrieve(url, ckpt_path)

    size_mb = os.path.getsize(ckpt_path) / 1e6
    with open(ckpt_path, "rb") as f:
        magic = f.read(2)
    if magic != b"PK" or size_mb < 1.0:
        raise RuntimeError(
            f"Downloaded checkpoint looks invalid ({size_mb:.2f} MB, magic={magic!r}). "
            "Check the URL or network access."
        )

    print(f"Using pretrained checkpoint: {ckpt_path} ({size_mb:.1f} MB)")
    return ckpt_path


def _conda_executable():
    conda_exe = shutil.which("conda")
    if conda_exe:
        return conda_exe
    for candidate in ("/anaconda/bin/conda", "/opt/conda/bin/conda"):
        if os.path.isfile(candidate):
            return candidate
    return None


def run_with_conda_env(cmd, env_name="hackathon", cwd=None):
    """Run *cmd* inside a conda environment (or plain bash if conda is absent)."""
    conda_exe = _conda_executable()
    if conda_exe is None:
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash", cwd=cwd,
            capture_output=True, text=True,
        )
    else:
        conda_base = subprocess.check_output(
            f"{conda_exe} info --base", shell=True, text=True
        ).strip()
        full_cmd = (
            f"source {conda_base}/etc/profile.d/conda.sh && "
            f"conda activate {env_name} && "
            f"{cmd}"
        )
        result = subprocess.run(
            full_cmd, shell=True, executable="/bin/bash", cwd=cwd,
            capture_output=True, text=True,
        )
    if result.stderr:
        print("Subprocess STDERR:")
        print(result.stderr)
    if result.stdout:
        print("Subprocess STDOUT:")
        print(result.stdout)
    return result.returncode


def run_regional_inference(
    checkpoint_path, custom_file, output_nc, custom_times_str, corrdiff_dir,
):
    """Run ``generate.py`` for the regional transfer experiment.

    Returns the path to the output NetCDF on success.
    """
    if checkpoint_path is None:
        raise RuntimeError("No checkpoint found. Run training first.")
    if not os.path.exists(custom_file):
        raise RuntimeError(
            f"Custom regional GFS file not found: {custom_file}\n"
            "Run the data-prep step first."
        )
    if not custom_times_str:
        raise RuntimeError("No valid times provided. Run the data-prep step first.")

    if os.path.exists(output_nc) and os.path.getsize(output_nc) == 0:
        os.remove(output_nc)

    generate_cmd = (
        "python generate.py "
        "--config-name=config_generate_hrrr_mini.yaml "
        f"++generation.io.reg_ckpt_filename={checkpoint_path} "
        f"++dataset.data_path={custom_file} "
        f"++generation.io.output_filename={output_nc} "
        "++generation.inference_mode=regression "
        "++generation.num_ensembles=1 "
        "++generation.seed_batch_size=1 "
        f"++generation.times=[{','.join(custom_times_str)}] "
        "++wandb.mode=disabled"
    )

    print(
        f"Running regional inference for {len(custom_times_str)} forecast frame(s): "
        f"{custom_times_str[0]} ... {custom_times_str[-1]}"
    )
    print(f"Input file:  {custom_file}")
    print(f"Output file: {output_nc}")

    exit_code = run_with_conda_env(generate_cmd, cwd=corrdiff_dir)
    if exit_code != 0:
        raise RuntimeError(f"Regional GFS inference failed with exit code {exit_code}")

    if not os.path.exists(output_nc) or os.path.getsize(output_nc) == 0:
        raise RuntimeError(
            f"Inference did not produce a valid output file: {output_nc}"
        )

    print(f"Regional GFS inference output: {output_nc}")
    return output_nc
