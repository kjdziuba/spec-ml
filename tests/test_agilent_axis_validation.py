"""Guards against the header that OOM-kills the Agilent readers.

Four pieces in the rat-lung set carry INTERFEROGRAM parameters in the file the
loader reads for the spectral axis: startwavenumber = -132, numberofpoints =
1041, wavenumberstep = 1.266e-4. Read unsigned, -132 becomes 4294967164 and
``np.arange(1, 1041 + 4294967164)`` asks for 34 GB. The process is SIGKILLed
with no traceback, which is undiagnosable from the outside.
"""
import io
import struct

import pytest

from specml.utils.agilentirformats.agilentirformats.agilentmosaic import AgilentMosaic
from specml.utils.agilentirformats.agilentirformats.agilenttile import AgilentTile


READERS = [AgilentMosaic, AgilentTile]


@pytest.mark.parametrize("cls", READERS)
class TestSignedHeaderRead:
    def test_negative_int32_reads_negative(self, cls):
        """-132 must read as -132, not 4294967164."""
        buf = io.BytesIO(struct.pack("<i", -132))
        assert cls()._readwinint32(buf) == -132

    def test_positive_int32_unchanged(self, cls):
        buf = io.BytesIO(struct.pack("<i", 233))
        assert cls()._readwinint32(buf) == 233

    def test_large_positive_still_works(self, cls):
        buf = io.BytesIO(struct.pack("<i", 2_000_000_000))
        assert cls()._readwinint32(buf) == 2_000_000_000


@pytest.mark.parametrize("cls", READERS)
class TestAxisValidation:
    def _make(self, cls, start, npts, step):
        r = cls()
        r.fstub = "probe"
        r.startwavenumber, r.numberofpoints, r.wavenumberstep = start, npts, step
        return r

    def test_accepts_a_real_spectral_axis(self, cls):
        self._make(cls, 233, 754, 3.8569142818)._validate_axis()

    def test_rejects_the_interferogram_header(self, cls):
        r = self._make(cls, -132, 1041, 0.0001265989)
        with pytest.raises(ValueError, match="does not describe a spectral axis"):
            r._validate_axis()

    def test_error_names_the_recovery_path(self, cls):
        r = self._make(cls, -132, 1041, 0.0001265989)
        with pytest.raises(ValueError, match=r"probe_bkgd\.bsp"):
            r._validate_axis()

    def test_rejects_the_unsigned_misread(self, cls):
        """What the old code produced: 2**32 - 132."""
        r = self._make(cls, 4294967164, 1041, 0.0001265989)
        with pytest.raises(ValueError):
            r._validate_axis()

    @pytest.mark.parametrize("start,npts,step", [
        (0, 754, 3.857),            # zero start
        (-1, 754, 3.857),           # negative start
        (233, 0, 3.857),            # no points
        (233, 754, 0.0),            # zero step
        (233, 754, -3.857),         # negative step
        (233, 200_000, 3.857),      # absurd point count
    ])
    def test_rejects_other_nonsense(self, cls, start, npts, step):
        with pytest.raises(ValueError):
            self._make(cls, start, npts, step)._validate_axis()

    def test_failure_is_fast_and_allocates_nothing(self, cls):
        """The whole point: fail before np.arange is reached."""
        import time
        r = self._make(cls, -132, 1041, 0.0001265989)
        t0 = time.time()
        with pytest.raises(ValueError):
            r._validate_axis()
        assert time.time() - t0 < 0.5
