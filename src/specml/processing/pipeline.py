"""YAML-driven preprocessing pipeline runner.

This is the single integration entry-point that downstream projects
(e.g. ``rat_lung_project``) call to apply the baseline preprocessing
described in their ``preprocessing_baseline.yaml`` to a
:class:`specml.data.SpectroscopicData` instance.

Schema mirrors ``rat_lung_project/configs/preprocessing_baseline.yaml``:

.. code-block:: yaml

    min_to_zero: true
    savitzky_golay:
      enabled: true
      window_length: 9
      polyorder: 5
      derivatives: [1, 2]
    normalization: "vector"
    tissue_mask:
      band_cm1: [1600, 1700]
      integration: "simpson"
      threshold_ftir: 2.5
      threshold_qcl: 3.0
      remove_small_objects_ftir: 400
      remove_small_objects_qcl: 1200
    wax_gap_cm1: [1350, 1490]
    spectral_range_cm1: [1000, 1800]

Output is a dict ``{'raw': sd, 'd1': sd, 'd2': sd, ...}`` — one
:class:`SpectroscopicData` per requested derivative order. The caller
stacks the resulting cubes along whichever axis its model expects.
``derivatives: [0]`` (or an empty list / disabled SG block) returns only
``{'raw': sd}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def run_from_config(spec_data, config, *, modality: str = "ftir"):
    """Apply the preprocessing pipeline declared in ``config``.

    Args:
        spec_data: A :class:`specml.data.SpectroscopicData` instance — typically
            freshly loaded from an Agilent mosaic, Zarr, or HDF5 file.
        config: Either a mapping (e.g. the parsed YAML) or a path / string
            pointing at a YAML file with the schema documented above.
        modality: ``'ftir'`` or ``'qcl'``. Selects between
            ``threshold_ftir``/``threshold_qcl`` and the corresponding
            ``remove_small_objects_*`` entries in ``tissue_mask``.

    Returns:
        ``dict[str, SpectroscopicData]`` keyed by derivative label
        (``'raw'``, ``'d1'``, ``'d2'``, ...). The baseline preprocessing
        (crop / wax / mask / normalise) is shared across the derivative
        channels; only the final SG step differs per entry.
    """
    cfg = _load_config(config)
    modality = modality.lower()
    if modality not in ("ftir", "qcl"):
        raise ValueError(f"modality must be 'ftir' or 'qcl', got {modality!r}.")

    sg_cfg = cfg.get("savitzky_golay") or {}
    sg_enabled = bool(sg_cfg.get("enabled", True))
    sg_window = int(sg_cfg.get("window_length", 9))
    sg_polyorder = int(sg_cfg.get("polyorder", 5))
    derivatives = list(sg_cfg.get("derivatives") or [])
    if not sg_enabled or not derivatives:
        derivatives = [0]
    derivatives = sorted({int(d) for d in derivatives})

    tissue_cfg = cfg.get("tissue_mask") or {}
    band_cm1 = tuple(tissue_cfg.get("band_cm1", (1600, 1700)))
    integration = str(tissue_cfg.get("integration", "simpson"))
    threshold = _modality_lookup(tissue_cfg, "threshold", modality, default=2.5)
    mask_min_size = _modality_lookup(
        tissue_cfg, "remove_small_objects", modality, default=None
    )
    mask_closing = int(tissue_cfg.get("closing_radius_px", 0))

    spectral_range = cfg.get("spectral_range_cm1")
    final_range = tuple(spectral_range) if spectral_range else (1000, 1800)

    wax_gap = cfg.get("wax_gap_cm1")
    wax_ranges = (tuple(wax_gap),) if wax_gap else ()

    co2_cfg = cfg.get("co2_range_cm1")
    co2_range = tuple(co2_cfg) if co2_cfg else None

    min_to_zero = bool(cfg.get("min_to_zero", True))
    normalization = str(cfg.get("normalization", "vector")).lower()
    if normalization not in ("", "none", "vector"):
        raise NotImplementedError(
            f"run_from_config: normalization={normalization!r} not yet supported "
            "in the YAML driver. Use 'vector' or omit."
        )

    if modality == "qcl":
        initial_crop = None
    else:
        initial_crop = tuple(cfg.get("initial_crop_cm1", (950, 3550)))

    baseline = spec_data.easy_preprocess(
        wavenumber_range=final_range,
        area_range=band_cm1,
        enable_mask=True,
        area_threshold=float(threshold),
        deriv=0,
        integration=integration,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        wax_ranges=wax_ranges,
        co2_range=co2_range,
        min_to_zero=min_to_zero,
        vector_normalize=(normalization == "vector"),
        mask_min_size_px=mask_min_size,
        mask_closing_radius_px=mask_closing,
        inplace=False,
    )
    if modality == "qcl":
        # easy_preprocess does the wide initial crop; preprocess_qcl skips it.
        # Use the QCL-tuned wrapper instead so we don't accidentally crop
        # below the QCL's actual spectral range.
        baseline = spec_data.preprocess_qcl(
            area_range=band_cm1,
            enable_mask=True,
            area_threshold=float(threshold),
            deriv=0,
            wavenumber_range=final_range,
            integration=integration,
            sg_window=sg_window,
            sg_polyorder=sg_polyorder,
            wax_ranges=wax_ranges,
            co2_range=co2_range,
            min_to_zero=min_to_zero,
            vector_normalize=(normalization == "vector"),
            mask_min_size_px=mask_min_size,
            mask_closing_radius_px=mask_closing,
            inplace=False,
        )

    channels = {}
    for d in derivatives:
        if d == 0:
            channels["raw"] = baseline
        else:
            channels[f"d{d}"] = baseline.apply_derivative(
                window_length=sg_window,
                polyorder=sg_polyorder,
                deriv=d,
                inplace=False,
            )
    return channels


def _load_config(config) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"YAML at {path} did not parse to a mapping.")
    return loaded


def _modality_lookup(section: Mapping[str, Any], key: str, modality: str, *, default):
    """Look up a modality-specific entry, falling back to a generic one.

    Search order: ``{key}_{modality}`` → ``{key}`` → ``default``.
    """
    if f"{key}_{modality}" in section:
        return section[f"{key}_{modality}"]
    if key in section:
        return section[key]
    return default
