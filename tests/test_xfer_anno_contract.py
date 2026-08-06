"""Regression test for the four-symbol surface that xfer_anno depends on.

This file is the contract gate documented in
``rat_lung_project/reports/audit_specml.md`` § 3:

    1. ``import specml; specml.__version__`` is a str.
    2. ``from specml import SpectroscopicData`` is callable.
    3. ``SpectroscopicData(path)`` loads a real file.
    4. ``sd.to_image()`` returns an ``(ypixels, xpixels, n_wavenumbers)`` ndarray.
    5. ``sd.wavenumbers`` is a monotonic 1-D float ndarray.

If any of these break, xfer_anno's
``xfer_anno.io.spectro.load_spectroscopic_image`` breaks too — and that's
the load-bearing data path for the rat lung annotation transfer pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

import specml
from specml import SpectroscopicData


YPIXELS = 8
XPIXELS = 12
N_WN = 32


def _make_pyramid_zarr(path: Path) -> Path:
    """Build a tiny rat-lung-style pyramid Zarr at ``path``.

    Uses ``group[name] = array`` assignment, which works across zarr v2 and v3
    (v3 dropped ``Group.create_dataset``).
    """
    root = zarr.open(str(path), mode="w")
    sub = root.create_group("0")
    rng = np.random.default_rng(0)
    sub["hypercube"] = rng.standard_normal((YPIXELS, XPIXELS, N_WN)).astype(np.float32)
    sub["wvnm"] = np.linspace(1000.0, 1800.0, N_WN, dtype=np.float32)
    sub["mask"] = np.ones((YPIXELS, XPIXELS), dtype=bool)
    return path


@pytest.fixture
def pyramid_zarr(tmp_path: Path) -> Path:
    return _make_pyramid_zarr(tmp_path / "synthetic.zarr")


def test_version_exposed():
    assert isinstance(specml.__version__, str)
    assert specml.__version__  # non-empty


def test_spectroscopic_data_is_exported():
    assert callable(SpectroscopicData)
    # Stable import path that downstream code (xfer_anno.io.spectro) uses.
    from specml import SpectroscopicData as _SD

    assert _SD is SpectroscopicData


def test_load_pyramid_zarr_returns_cube(pyramid_zarr: Path):
    sd = SpectroscopicData(str(pyramid_zarr))
    cube = sd.to_image()
    assert cube.ndim == 3
    assert cube.shape == (YPIXELS, XPIXELS, N_WN)


def test_wavenumbers_are_monotonic_1d(pyramid_zarr: Path):
    sd = SpectroscopicData(str(pyramid_zarr))
    wn = sd.wavenumbers
    assert isinstance(wn, np.ndarray)
    assert wn.ndim == 1
    assert wn.size == N_WN
    diffs = np.diff(wn)
    assert (diffs > 0).all() or (diffs < 0).all()


# --------------------------------------------------------------------------- #
# Mask/data consistency after spatial cleanup                                 #
# --------------------------------------------------------------------------- #
#
# Contract: after easy_preprocess, the compressed spectra (self.data, one row
# per active pixel) and the tissue mask (self.mask) must agree —
# ``data.shape[0] == mask.sum()`` — so a downstream consumer can re-expand
# spectra back onto the pixel grid via ``flatnonzero(mask)``. Before the fix,
# ``remove_small_objects`` shrank the mask but left the removed pixels' spectra
# in self.data, desyncing the two (data rows > mask active) and breaking
# rat-lung's train_28d_concordance re-expansion.


def _make_masked_zarr(path: Path, *, Y=20, X=24, n_wn=300) -> Path:
    """Zarr with a big tissue blob + one small isolated object (removable).

    Wavenumbers span a realistic 900-3800 cm^-1 so easy_preprocess's default
    crop/wax/CO2 ranges all fall inside the axis. Uses ``group[name] = array``
    assignment, which works across zarr v2 and v3 (v3 dropped
    ``Group.create_dataset``).
    """
    root = zarr.open(str(path), mode="w")
    sub = root.create_group("0")
    rng = np.random.default_rng(1)
    cube = np.zeros((Y, X, n_wn), dtype=np.float32)  # background = 0 -> area 0
    # Big connected blob (one component, survives remove_small_objects).
    cube[3:17, 3:21, :] = rng.uniform(0.5, 1.5, (14, 18, n_wn)).astype(np.float32)
    # Small isolated 2x2 object, disconnected from the blob (must be removed).
    cube[18:20, 22:24, :] = rng.uniform(0.5, 1.5, (2, 2, n_wn)).astype(np.float32)
    sub["hypercube"] = cube
    sub["wvnm"] = np.linspace(900.0, 3800.0, n_wn, dtype=np.float32)
    sub["mask"] = np.ones((Y, X), dtype=bool)
    return path


def _active(mask) -> int:
    return int(np.asarray(mask).astype(bool).sum())


def test_easy_preprocess_data_and_mask_consistent_without_cleanup(tmp_path: Path):
    sd = SpectroscopicData(str(_make_masked_zarr(tmp_path / "m.zarr")))
    out = sd.easy_preprocess(
        area_range=(1600, 1700), area_threshold=0.5,
        mask_min_size_px=None, mask_closing_radius_px=0, inplace=False,
    )
    # Both tissue regions kept (14*18 + 2*2 = 256), data rows == mask active.
    assert out.data.shape[0] == _active(out.mask)
    assert _active(out.mask) == 256


def test_easy_preprocess_trims_data_when_small_object_removed(tmp_path: Path):
    sd = SpectroscopicData(str(_make_masked_zarr(tmp_path / "m.zarr")))
    out = sd.easy_preprocess(
        area_range=(1600, 1700), area_threshold=0.5,
        mask_min_size_px=10, mask_closing_radius_px=0, inplace=False,
    )
    # The 4-px object is gone (256 -> 252) AND data was trimmed to match.
    assert _active(out.mask) == 252, "small object should be removed from the mask"
    assert out.data.shape[0] == _active(out.mask), (
        f"data rows ({out.data.shape[0]}) must equal mask active "
        f"({_active(out.mask)}) — spectra desynced from the mask"
    )
