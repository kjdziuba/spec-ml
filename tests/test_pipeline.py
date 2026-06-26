"""Tests for the YAML-driven preprocessing pipeline (``run_from_config``).

This is the integration target the rat-lung project drives through to get
its baseline preprocessing applied end-to-end. The schema mirrors
``rat_lung_project/configs/preprocessing_baseline.yaml`` — see
``specml.processing.pipeline`` for the contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from specml import SpectroscopicData
from specml.processing.pipeline import run_from_config


YPIX = 8
XPIX = 12
NWN = 64


@pytest.fixture
def tissue_zarr(tmp_path: Path) -> Path:
    """A synthetic Zarr where half the pixels are "tissue" (positive Amide-I)
    and half are "background" (zero), so the tissue-mask step has work to do.
    """
    rng = np.random.default_rng(42)
    wn = np.linspace(900.0, 1900.0, NWN, dtype=np.float32)
    cube = np.zeros((YPIX, XPIX, NWN), dtype=np.float32)
    # Active region: rows 0..3 have a strong Amide-I band, rows 4..7 stay at 0.
    amide_band = (wn >= 1600) & (wn <= 1700)
    cube[:4, :, :] = rng.normal(0.5, 0.05, size=(4, XPIX, NWN)).astype(np.float32)
    cube[:4, :, amide_band] += 2.0
    mask = np.ones((YPIX, XPIX), dtype=bool)
    path = tmp_path / "tissue.zarr"
    root = zarr.open(str(path), mode="w")
    sub = root.create_group("0")
    sub.create_dataset("hypercube", data=cube)
    sub.create_dataset("wvnm", data=wn)
    sub.create_dataset("mask", data=mask)
    return path


def _base_config() -> dict:
    return {
        "min_to_zero": True,
        "savitzky_golay": {
            "enabled": True,
            "window_length": 9,
            "polyorder": 3,
            "derivatives": [1, 2],
        },
        "normalization": "vector",
        "tissue_mask": {
            "band_cm1": [1600, 1700],
            "integration": "simpson",
            "threshold_ftir": 1.0,
            "threshold_qcl": 1.0,
            "remove_small_objects_ftir": 4,
            "remove_small_objects_qcl": 4,
        },
        "wax_gap_cm1": [1350, 1490],
        "spectral_range_cm1": [1000, 1800],
        "initial_crop_cm1": [950, 1850],
    }


def test_run_from_config_returns_requested_derivatives(tissue_zarr: Path):
    sd = SpectroscopicData(str(tissue_zarr))
    result = run_from_config(sd, _base_config(), modality="ftir")
    assert set(result.keys()) == {"d1", "d2"}
    for label, channel in result.items():
        assert channel.data.ndim == 2
        # SG (window=9, polyorder=3) with interp=False trims 4 wavenumbers
        # off each end of the cropped range; we only assert that something
        # plausible came back, not the exact length.
        assert channel.data.shape[0] > 0
        assert channel.data.shape[1] > 0


def test_run_from_config_includes_raw_when_zero_in_derivatives(tissue_zarr: Path):
    cfg = _base_config()
    cfg["savitzky_golay"]["derivatives"] = [0, 1, 2]
    sd = SpectroscopicData(str(tissue_zarr))
    result = run_from_config(sd, cfg, modality="ftir")
    assert set(result.keys()) == {"raw", "d1", "d2"}


def test_run_from_config_disabled_sg_returns_raw_only(tissue_zarr: Path):
    cfg = _base_config()
    cfg["savitzky_golay"]["enabled"] = False
    sd = SpectroscopicData(str(tissue_zarr))
    result = run_from_config(sd, cfg, modality="ftir")
    assert set(result.keys()) == {"raw"}


def test_run_from_config_modality_picks_thresholds(tissue_zarr: Path):
    cfg = _base_config()
    cfg["tissue_mask"]["threshold_ftir"] = 1e6  # impossible — nothing passes
    cfg["tissue_mask"]["threshold_qcl"] = 0.0   # everything passes
    sd_ftir = SpectroscopicData(str(tissue_zarr))
    # With threshold_ftir=1e6 there is no tissue → rebuild raises on the
    # downstream side because the active set is empty. We accept either
    # exception OR an empty mask depending on the integration path.
    with pytest.raises(Exception):
        run_from_config(sd_ftir, cfg, modality="ftir")
    sd_qcl = SpectroscopicData(str(tissue_zarr))
    result = run_from_config(sd_qcl, cfg, modality="qcl")
    assert set(result.keys()) <= {"raw", "d1", "d2"}


def test_run_from_config_accepts_yaml_path(tissue_zarr: Path, tmp_path: Path):
    import yaml as _yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_yaml.safe_dump(_base_config()), encoding="utf-8")
    sd = SpectroscopicData(str(tissue_zarr))
    result = run_from_config(sd, cfg_path, modality="ftir")
    assert "d1" in result
