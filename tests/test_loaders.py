"""Tests for the Zarr and HDF5 loaders.

Covers both the *pyramid* layout that the rat-lung QCL pipeline writes
(``<store>/0/{hypercube,wvnm,mask}``) and the legacy *flat* layout
(``<store>/{qcl_data,wavenumbers,mask}``) — that distinction was the
root cause of the rat-lung-blocking ZarrReader bug logged in the specml
audit (§ 2 #1).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from specml.data.loaders import DataLoader, ZarrReader


YPIX = 6
XPIX = 9
NWN = 24


def _make_synthetic_cube(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal((YPIX, XPIX, NWN)).astype(np.float32)
    wn = np.linspace(1000.0, 1800.0, NWN, dtype=np.float32)
    mask = np.ones((YPIX, XPIX), dtype=bool)
    mask[0, 0] = False  # one masked pixel to verify mask propagation
    return cube, wn, mask


def _write_pyramid_zarr(path: Path) -> Path:
    cube, wn, mask = _make_synthetic_cube()
    root = zarr.open(str(path), mode="w")
    sub = root.create_group("0")
    sub.create_dataset("hypercube", data=cube)
    sub.create_dataset("wvnm", data=wn)
    sub.create_dataset("mask", data=mask)
    return path


def _write_flat_zarr(path: Path) -> Path:
    cube, wn, mask = _make_synthetic_cube()
    root = zarr.open(str(path), mode="w")
    root.create_dataset("qcl_data", data=cube)
    root.create_dataset("wavenumbers", data=wn)
    root.create_dataset("mask", data=mask)
    return path


# ---- Zarr ----------------------------------------------------------------


def test_zarr_pyramid_layout_reads(tmp_path: Path):
    path = _write_pyramid_zarr(tmp_path / "pyr.zarr")
    result = ZarrReader().read(str(path))
    assert result["data"].shape == (YPIX, XPIX, NWN)
    assert result["wavenumbers"].shape == (NWN,)
    assert result["mask"].shape == (YPIX, XPIX)
    assert result["mask"].dtype == bool
    assert result["mask"][0, 0] == False  # noqa: E712
    assert result["metadata"]["xpixels"] == XPIX
    assert result["metadata"]["ypixels"] == YPIX


def test_zarr_flat_layout_reads(tmp_path: Path):
    path = _write_flat_zarr(tmp_path / "flat.zarr")
    result = ZarrReader().read(str(path))
    assert result["data"].shape == (YPIX, XPIX, NWN)
    assert result["wavenumbers"].shape == (NWN,)
    assert result["mask"].dtype == bool


def test_zarr_layout_dispatches_through_load_data(tmp_path: Path):
    path = _write_pyramid_zarr(tmp_path / "via_dataloader.zarr")
    assert DataLoader.determine_file_type(str(path)) == "zarr"
    result = DataLoader.load_data(str(path))
    # load_data wraps metadata under {'info': ...} for the SpectroscopicData
    # constructor — see _normalise_metadata.
    assert "data" in result and result["data"].shape == (YPIX, XPIX, NWN)
    assert result["metadata"]["xpixels"] == XPIX


def test_zarr_rejects_missing_hypercube(tmp_path: Path):
    bad = tmp_path / "bad.zarr"
    root = zarr.open(str(bad), mode="w")
    root.create_dataset("wavenumbers", data=np.arange(NWN, dtype=np.float32))
    with pytest.raises(KeyError, match="hypercube"):
        ZarrReader().read(str(bad))


def test_zarr_rejects_shape_mismatch(tmp_path: Path):
    bad = tmp_path / "shape_mismatch.zarr"
    root = zarr.open(str(bad), mode="w")
    sub = root.create_group("0")
    sub.create_dataset("hypercube", data=np.zeros((YPIX, XPIX, NWN), dtype=np.float32))
    sub.create_dataset("wvnm", data=np.zeros(NWN + 1, dtype=np.float32))
    with pytest.raises(ValueError, match="wavenumber"):
        ZarrReader().read(str(bad))


# ---- HDF5 ----------------------------------------------------------------


def _write_synthetic_hdf5(path: Path) -> Path:
    cube, wn, mask = _make_synthetic_cube(seed=1)
    intensities = cube.reshape(-1, NWN)  # what load_hdf5 expects
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("intensities", data=intensities)
        grp.create_dataset("xvalues", data=wn[np.newaxis, :])
        grp.create_dataset("xpixels", data=np.array([XPIX]))
        grp.create_dataset("ypixels", data=np.array([YPIX]))
        grp.attrs["source"] = "synthetic"
    return path


def test_hdf5_round_trip(tmp_path: Path):
    path = _write_synthetic_hdf5(tmp_path / "synthetic.h5")
    result = DataLoader.load_hdf5(str(path))
    assert result["data"].shape == (YPIX * XPIX, NWN)
    assert result["wavenumbers"].shape == (NWN,)
    assert result["metadata"]["xpixels"] == XPIX
    assert result["metadata"]["ypixels"] == YPIX
    # The load_hdf5 path leaves mask as None — caller falls back to all-True.
    assert result["mask"] is None
