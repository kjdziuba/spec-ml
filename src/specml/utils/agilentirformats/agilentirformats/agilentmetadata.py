"""Read acquisition and processing metadata from Agilent FTIR sidecar files.

Agilent's ``.dmt`` (mosaic master), ``.bsp`` and ``.seq`` files carry the
instrument settings and the processing chain that produced the spectra. The
binary layout is undocumented; it was reverse-engineered by hex inspection and
is simple and stable:

**Every string -- key or value -- is stored as a doubled 32-bit little-endian
length prefix followed by that many ASCII bytes:**

    <int32 L> <int32 L> <L bytes>

Strings appear in key/value order. A key whose value was never set is stored
with ``L = 0``, which is why pairing cannot be done by alternating position
alone; :data:`KNOWN_KEYS` disambiguates.

Reading the length prefix rather than scanning for printable runs matters:
Agilent repeats ``Time Stamp`` once per processing step with several instances
empty, and a length byte that happens to fall in printable ASCII range (34 for
a 34-character timestamp, 53 for a 53-character path) otherwise masquerades as
a one-character value.

Why this matters
----------------
Everything needed to reproduce Agilent's Fourier transform is declared here:
``Apodization Type``, ``Phase Correction``, ``Zero Filling Factor``,
``Symmetry``, ``Effective Laser Wavenumber`` and the ``STEPn`` processing
chain. That makes it possible to re-transform raw interferograms (``.drd``)
for an acquisition that was interrupted before Resolutions Pro ran its
transform stage, instead of re-scanning the sample.

It is also the per-scan instrument record needed for batch-effect analysis:
acquisition timestamp, detector, aperture, source power, gain, scan count and
resolution, all per piece.

Example
-------
>>> from specml.utils.agilentirformats.agilentirformats import agilentmetadata
>>> md = agilentmetadata.read_metadata("sample.dmt")
>>> md["Apodization Type"]
'Happ-Genzel'
>>> agilentmetadata.acquisition_summary(md)["laser_wavenumber"]
15797.9209
"""
from __future__ import annotations

import datetime as _dt
import os
import re

__all__ = [
    "KNOWN_KEYS",
    "read_key_values",
    "read_metadata",
    "acquisition_summary",
]

# Keys observed in Agilent Resolutions Pro / "Agilent FTIR" sidecar files.
# Anything not in this table is ignored, which keeps stray ASCII in the binary
# from being mistaken for metadata.
KNOWN_KEYS = frozenset({
    # provenance
    "Time Stamp", "User Stamp", "Software Version", "SpectName", "History",
    "Method", "PropType", "Properties", "Created", "Background Filename",
    "Phase Array Filename", "DefaultScanName",
    # optical / instrument state
    "Aperture", "Selected Aperture", "Source Power", "Beam", "Detector",
    "Filter", "Source", "Speed", "Symmetry", "Sensitivity", "Signal",
    "Effective Laser Wavenumber", "Gain Range Radius", "Gain Factor",
    "In Scan CoAddition", "Resolution", "Scans", "Background Scans",
    "Under Sampling Ratio", "Raman Aperture", "Noise Level Ratio",
    "Visible Pixel Size", "Average Subtract",
    "Match Aperture to Selected Resolution", "Poly Flipper In",
    # Fourier-transform processing chain
    "Apodization Type", "Phase Correction", "Phase Apodization",
    "Zero Filling Factor", "Phase Size", "Included Frequency",
    "Trapezoidal Apodization Break Point", "Store Phase",
    "Use standard processing", "Generate real output",
    "Generate imaginary output", "Generate magnitude output",
    "Region", "To",
    # processing steps
    "STEP0", "STEP1", "STEP2", "STEP3", "STEP4",
    "STEP5", "STEP6", "STEP7", "STEP8", "STEP9",
})

_MAX_STR = 4096  # no Agilent metadata string observed anywhere near this

# "Monday, November 25, 2024 12:22:31"
_TS = re.compile(
    r"^\s*[A-Za-z]+,\s+([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})\s+"
    r"(\d{1,2}):(\d{2}):(\d{2})\s*$"
)
_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _extract_strings(blob):
    """Every length-prefixed ASCII string in the blob, in file order.

    Returns a list of ``(offset, text)``. Candidate positions are found with
    numpy (the files run to ~500 kB, so a byte-by-byte Python scan is
    needlessly slow), then validated individually.
    """
    try:
        import numpy as np
    except ImportError:                                   # pragma: no cover
        np = None

    n = len(blob)
    starts = []
    if np is not None:
        arr = np.frombuffer(blob, dtype=np.uint8)
        # The doubled length prefix is two adjacent uint32s, so both share the
        # same alignment -- but that alignment is not guaranteed to be 0 mod 4,
        # so all four phases must be checked.
        for phase in range(4):
            usable = (n - phase) // 4
            if usable < 2:
                continue
            u = arr[phase:phase + usable * 4].view("<u4")
            hit = np.flatnonzero((u[:-1] == u[1:]) & (u[:-1] <= _MAX_STR))
            starts.append(phase + hit.astype(np.int64) * 4)
        cand = np.unique(np.concatenate(starts)) if starts else []
    else:                                                 # pragma: no cover
        cand = range(n - 8)

    out = []
    consumed_to = -1
    for i in (int(c) for c in cand):
        if i < consumed_to:
            continue
        length = int.from_bytes(blob[i:i + 4], "little")
        if i + 8 + length > n:
            continue
        raw = blob[i + 8:i + 8 + length]
        if length and not all(0x20 <= c < 0x7F for c in raw):
            continue
        out.append((i, raw.decode("ascii")))
        consumed_to = i + 8 + length
    return out


def read_key_values(path, encoding="ascii"):
    """Return every ``(key, value)`` pair found in an Agilent sidecar file.

    Later occurrences of a key are kept alongside earlier ones, so the result
    is a list rather than a dict -- Agilent repeats ``Time Stamp`` once per
    processing step, and most of those instances are empty.
    """
    with open(path, "rb") as fh:
        blob = fh.read()

    strings = _extract_strings(blob)
    pairs = []
    for i, (_, text) in enumerate(strings):
        if text not in KNOWN_KEYS:
            continue
        nxt = strings[i + 1][1] if i + 1 < len(strings) else ""
        # a key immediately followed by another key had no value set
        pairs.append((text, "" if nxt in KNOWN_KEYS else nxt))
    return pairs


def read_metadata(path, keep="last"):
    """Return a ``dict`` of Agilent metadata.

    Parameters
    ----------
    path : str
        A ``.dmt``, ``.bsp`` or ``.seq`` file.
    keep : {"last", "first"}
        Which occurrence to keep when a key repeats. Agilent writes one
        ``Time Stamp`` per processing step; ``"last"`` therefore gives the
        most recent, which is normally what is wanted.
    """
    pairs = read_key_values(path)
    out = {}
    for k, v in pairs:
        if not v:
            out.setdefault(k, v)
            continue
        if keep == "first" and k in out and out[k]:
            continue
        out[k] = v
    out["_source_file"] = os.path.basename(path)
    out["_all_pairs"] = pairs
    return out


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp(pairs):
    """Most recent parseable ``Time Stamp`` value, as a datetime."""
    best = None
    for key, value in pairs:
        if key != "Time Stamp":
            continue
        m = _TS.match(value or "")
        if not m:
            continue
        month, day, year, hh, mm, ss = m.groups()
        if month not in _MONTHS:
            continue
        try:
            when = _dt.datetime(int(year), _MONTHS[month], int(day),
                                int(hh), int(mm), int(ss))
        except ValueError:
            continue
        if best is None or when > best:
            best = when
    return best


def acquisition_summary(metadata):
    """Condense :func:`read_metadata` output into the fields worth reporting.

    Returns a flat dict suitable for a manifest row or a QC figure header.
    Missing entries are ``None`` rather than absent, so the result has a
    stable shape across pieces.
    """
    md = metadata
    pairs = md.get("_all_pairs", [])
    when = _timestamp(pairs)

    laser = _num(md.get("Effective Laser Wavenumber"))
    region = md.get("Region", "") or ""
    lo = hi = None
    nums = re.findall(r"\d+(?:\.\d+)?", region)
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
        lo, hi = min(a, b), max(a, b)

    steps = [f"{k}={md[k]}" for k in
             ("STEP0", "STEP1", "STEP2", "STEP3", "STEP4") if md.get(k)]

    return {
        "acquired": when.isoformat(sep=" ") if when else None,
        "acquired_date": when.date().isoformat() if when else None,
        "software": md.get("Software Version") or None,
        "operator_stamp": md.get("User Stamp") or None,
        # optical state
        "detector": md.get("Detector") or None,
        "source_power": md.get("Source Power") or None,
        "beam": md.get("Beam") or None,
        "filter": md.get("Filter") or None,
        "speed": md.get("Speed") or None,
        "symmetry": md.get("Symmetry") or None,
        "sensitivity": md.get("Sensitivity") or None,
        "gain_range_radius": _num(md.get("Gain Range Radius")),
        "laser_wavenumber": laser,
        # acquisition
        "resolution_cm1": _num(md.get("Resolution")),
        "scans": _num(md.get("Scans")),
        "background_scans": _num(md.get("Background Scans")),
        "undersampling_ratio": _num(md.get("Under Sampling Ratio")),
        "noise_level_ratio": _num(md.get("Noise Level Ratio")),
        # processing chain -- what is needed to reproduce the transform
        "apodization": md.get("Apodization Type") or None,
        "phase_correction": md.get("Phase Correction") or None,
        "phase_apodization": md.get("Phase Apodization") or None,
        "zero_filling": md.get("Zero Filling Factor") or None,
        "region_lo_cm1": lo,
        "region_hi_cm1": hi,
        "background_file": md.get("Background Filename") or None,
        "steps": "; ".join(steps) or None,
        "source_file": md.get("_source_file"),
    }
