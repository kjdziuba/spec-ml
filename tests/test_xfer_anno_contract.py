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
    """Build a tiny rat-lung-style pyramid Zarr at ``path``."""
    root = zarr.open(str(path), mode="w")
    sub = root.create_group("0")
    rng = np.random.default_rng(0)
    sub.create_dataset(
        "hypercube",
        data=rng.standard_normal((YPIXELS, XPIXELS, N_WN)).astype(np.float32),
        chunks=(YPIXELS, XPIXELS, N_WN),
    )
    sub.create_dataset(
        "wvnm",
        data=np.linspace(1000.0, 1800.0, N_WN, dtype=np.float32),
        chunks=(N_WN,),
    )
    sub.create_dataset(
        "mask",
        data=np.ones((YPIXELS, XPIXELS), dtype=bool),
        chunks=(YPIXELS, XPIXELS),
    )
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
