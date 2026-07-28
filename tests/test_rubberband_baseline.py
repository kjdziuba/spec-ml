"""Tests for rubberband_baseline_correction.

The defining property of a rubber-band baseline is that the band lies UNDER
the spectrum everywhere, so the corrected result is non-negative. The previous
implementation violated that on almost every input; these tests pin it down.
"""
import numpy as np
import pytest

from specml.processing.advanced_preprocessing import rubberband_baseline_correction


def _spectrum(n=400, slope=0.4, offset=0.3):
    x = np.linspace(0, 1, n)
    peaks = (1.0 * np.exp(-((x - 0.35) / 0.03) ** 2)
             + 0.6 * np.exp(-((x - 0.70) / 0.02) ** 2))
    return x, offset + slope * x + peaks


class TestRubberbandInvariants:
    def test_output_is_non_negative(self):
        """The band is stretched underneath, so nothing can end up below it."""
        _, y = _spectrum()
        out = rubberband_baseline_correction(y[None, :])[0]
        assert out.min() >= -1e-9, f"min was {out.min()}"

    def test_removes_a_linear_baseline_exactly(self):
        x = np.linspace(0, 1, 300)
        base = 0.25 + 0.8 * x
        out = rubberband_baseline_correction((base)[None, :])[0]
        assert np.allclose(out, 0.0, atol=1e-9)

    def test_preserves_peak_heights_above_baseline(self):
        x, y = _spectrum()
        out = rubberband_baseline_correction(y[None, :])[0]
        # the tall peak sits 1.0 above its local baseline and must survive
        assert out.max() == pytest.approx(1.0, abs=0.02)

    def test_endpoints_touch_zero(self):
        _, y = _spectrum()
        out = rubberband_baseline_correction(y[None, :])[0]
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[-1] == pytest.approx(0.0, abs=1e-9)

    def test_is_monotone_under_added_offset(self):
        """Adding a constant must not change the corrected spectrum."""
        _, y = _spectrum()
        a = rubberband_baseline_correction(y[None, :])[0]
        b = rubberband_baseline_correction((y + 5.0)[None, :])[0]
        assert np.allclose(a, b, atol=1e-9)

    def test_convex_baseline_is_followed_closely(self):
        """A convex (upward-curving) baseline is exactly what a rubber band
        can track, so peak height is recovered accurately."""
        x = np.linspace(0, 1, 400)
        base = 0.2 + 0.6 * (x - 0.5) ** 2
        y = base + 0.8 * np.exp(-((x - 0.65) / 0.02) ** 2)
        out = rubberband_baseline_correction(y[None, :])[0]
        assert out.min() >= -1e-9
        assert out.max() == pytest.approx(0.8, abs=0.02)

    def test_concave_baseline_is_under_estimated_by_design(self):
        """A rubber band is the LOWER CONVEX envelope, so it cannot follow a
        concave-down baseline -- it cuts a straight chord underneath instead.

        This is an inherent limitation of the method, not a defect, but it
        inflates apparent peak heights and is worth knowing about: here the
        baseline peaks at 0.496 while the chord sits at 0.400, so a peak of
        true height 0.8 is reported as ~0.896. Use ALS or ME-EMSC when the
        baseline is genuinely concave.
        """
        x = np.linspace(0, 1, 400)
        base = 0.5 - 0.4 * (x - 0.5) ** 2
        y = base + 0.8 * np.exp(-((x - 0.6) / 0.02) ** 2)
        out = rubberband_baseline_correction(y[None, :])[0]
        assert out.min() >= -1e-9
        chord_gap = base[np.argmin(np.abs(x - 0.6))] - base[0]
        assert out.max() == pytest.approx(0.8 + chord_gap, abs=0.02)


class TestShapesAndEdgeCases:
    def test_accepts_1d_and_returns_1d(self):
        _, y = _spectrum()
        out = rubberband_baseline_correction(y)
        assert out.shape == y.shape

    def test_accepts_2d_and_returns_2d(self):
        _, y = _spectrum()
        batch = np.vstack([y, y * 2, y + 1])
        out = rubberband_baseline_correction(batch)
        assert out.shape == batch.shape
        assert out.min() >= -1e-9

    def test_each_row_is_corrected_independently(self):
        _, y = _spectrum()
        single = rubberband_baseline_correction(y[None, :])[0]
        batch = rubberband_baseline_correction(np.vstack([y, y * 3]))
        assert np.allclose(batch[0], single, atol=1e-9)

    def test_flat_spectrum_becomes_zero(self):
        out = rubberband_baseline_correction(np.full((1, 200), 0.7))[0]
        assert np.allclose(out, 0.0, atol=1e-9)

    def test_too_few_points_passes_through(self):
        y = np.array([[1.0, 2.0]])
        out = rubberband_baseline_correction(y)
        assert np.allclose(out, y)

    def test_nan_channels_do_not_raise(self):
        _, y = _spectrum()
        y = y.copy()
        y[50:60] = np.nan
        out = rubberband_baseline_correction(y[None, :])[0]
        assert np.isfinite(out[np.isfinite(y)]).all()

    def test_num_points_subsampling_still_non_negative(self):
        _, y = _spectrum()
        out = rubberband_baseline_correction(y[None, :], num_points=50)[0]
        # subsampling can miss the exact minimum, so allow a small tolerance
        assert out.min() >= -0.02


class TestRegressionAgainstOldBehaviour:
    """The specific failure that motivated the fix."""

    def test_baseline_never_exceeds_the_spectrum(self):
        _, y = _spectrum()
        out = rubberband_baseline_correction(y[None, :])[0]
        implied_baseline = y - out
        assert np.all(implied_baseline <= y + 1e-9)

    def test_not_mostly_negative(self):
        """The old implementation returned 97.5% negative values here."""
        _, y = _spectrum()
        out = rubberband_baseline_correction(y[None, :])[0]
        assert (out < 0).mean() < 0.01
