"""Tests for the Agilent sidecar metadata reader.

The binary container is synthesised here rather than committed as a fixture,
so the unit tests need no data files. An integration test runs against a real
``.dmt`` when one is available and skips otherwise.
"""
import os
import struct

import pytest

from specml.utils.agilentirformats.agilentirformats import agilentmetadata as M


def _s(text):
    """Encode one string in Agilent's doubled-length-prefix container."""
    raw = text.encode("ascii")
    return struct.pack("<ii", len(raw), len(raw)) + raw


def _blob(*items):
    return b"\x00" * 16 + b"".join(_s(t) for t in items) + b"\x00" * 16


class TestExtractStrings:
    def test_round_trips_simple_pairs(self):
        blob = _blob("Detector", "GND", "Resolution", "8")
        got = [t for _, t in M._extract_strings(blob)]
        assert "Detector" in got and "GND" in got
        assert "Resolution" in got and "8" in got

    def test_single_character_values_survive(self):
        """'Resolution' -> '8' is one byte; a printable-run scanner loses it."""
        pairs = dict(_pairs(_blob("Resolution", "8")))
        assert pairs["Resolution"] == "8"

    def test_length_byte_is_not_mistaken_for_a_value(self):
        """A 34-char value has length 0x22 == '"', which must not be read as
        the value itself."""
        stamp = "Monday, November 25, 2024 12:22:31"
        assert len(stamp) == 34 and chr(34) == '"'
        pairs = dict(_pairs(_blob("Time Stamp", stamp)))
        assert pairs["Time Stamp"] == stamp

    def test_empty_value_is_empty_not_the_next_key(self):
        pairs = dict(_pairs(_blob("Aperture", "", "Source Power", "MEDIUM")))
        assert pairs["Aperture"] == ""
        assert pairs["Source Power"] == "MEDIUM"

    def test_unknown_keys_are_ignored(self):
        pairs = dict(_pairs(_blob("Not A Real Key", "junk", "Detector", "GND")))
        assert "Not A Real Key" not in pairs
        assert pairs["Detector"] == "GND"

    def test_non_ascii_payload_is_skipped(self):
        blob = b"\x00" * 8 + struct.pack("<ii", 4, 4) + b"\x00\x01\x02\x03"
        blob += _s("Detector") + _s("GND")
        pairs = dict(_pairs(blob))
        assert pairs["Detector"] == "GND"


def _pairs(blob, tmp_path=None):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".dmt")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        return M.read_key_values(path)
    finally:
        os.unlink(path)


class TestAcquisitionSummary:
    def test_parses_timestamp(self):
        pairs = [("Time Stamp", "Monday, November 25, 2024 12:22:31")]
        out = M.acquisition_summary({"_all_pairs": pairs})
        assert out["acquired"] == "2024-11-25 12:22:31"
        assert out["acquired_date"] == "2024-11-25"

    def test_picks_the_latest_timestamp(self):
        pairs = [("Time Stamp", ""),
                 ("Time Stamp", "Monday, November 25, 2024 12:22:31"),
                 ("Time Stamp", "Tuesday, November 26, 2024 14:44:30")]
        out = M.acquisition_summary({"_all_pairs": pairs})
        assert out["acquired"] == "2024-11-26 14:44:30"

    def test_parses_region_into_bounds(self):
        out = M.acquisition_summary(
            {"_all_pairs": [], "Region": "3800, 900 Wavenumber"})
        assert out["region_lo_cm1"] == 900.0
        assert out["region_hi_cm1"] == 3800.0

    def test_missing_fields_are_none_not_absent(self):
        out = M.acquisition_summary({"_all_pairs": []})
        for key in ("acquired", "detector", "apodization", "resolution_cm1"):
            assert key in out and out[key] is None

    def test_bad_timestamp_does_not_raise(self):
        out = M.acquisition_summary(
            {"_all_pairs": [("Time Stamp", "Fooday, Smarch 45, 2024 99:99:99")]})
        assert out["acquired"] is None


DMT = ("/Volumes/LaCie/Lung_Data_8cm/8F-8512-01/8F-8512-01_mid2/"
       "8f-8512-01_mid2.dmt")


@pytest.mark.skipif(not os.path.exists(DMT), reason="project drive not mounted")
class TestAgainstRealFile:
    """The values below are what the instrument actually recorded."""

    def test_reads_the_processing_chain(self):
        s = M.acquisition_summary(M.read_metadata(DMT))
        assert s["apodization"] == "Happ-Genzel"
        assert s["symmetry"] == "ASYM"
        assert s["phase_correction"] == "Auto"

    def test_reads_the_optical_constants(self):
        s = M.acquisition_summary(M.read_metadata(DMT))
        assert s["laser_wavenumber"] == pytest.approx(15797.9209)
        assert s["undersampling_ratio"] == 4.0
        assert s["resolution_cm1"] == 8.0
        assert s["scans"] == 64.0

    def test_declared_constants_predict_the_stored_axis(self):
        """laser / UDR is the Nyquist wavenumber; dividing by 1024 gives the
        stored axis step, which independently pins the FFT length at 2048."""
        s = M.acquisition_summary(M.read_metadata(DMT))
        nyquist = s["laser_wavenumber"] / s["undersampling_ratio"]
        assert nyquist / 1024 == pytest.approx(3.8569142818, abs=1e-8)

    def test_reads_the_timestamp(self):
        s = M.acquisition_summary(M.read_metadata(DMT))
        assert s["acquired_date"] == "2024-11-25"
