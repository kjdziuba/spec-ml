"""
SPECTRAL PREPROCESSING PIPELINE

1. INITIAL DATA QUALITY & CLEANUP
   - Bad pixel/cosmic ray removal using median filtering
   - Range selection (950-3550 cm^-1)
   - CO2 and wax artifact removal
   - Check/remove saturated spectra (intensity > threshold)
   - Apply amide I mask (1600-1700 cm^-1) to remove low-signal pixels

2. NOISE REDUCTION (before any transformative steps)
   - PCA/MNF denoising (optional)
     * Reduces random noise while preserving spectral features
     * Number of components typically 8-15
   - Savitzky-Golay smoothing (optional)
     * Window length: 7-11 points
     * Polynomial order: 2-3
     * Reduces high-frequency noise

3. BASELINE & NORMALIZATION
   - Baseline correction options:
     * Linear (min_to_zero): simplest approach
     * Polynomial: better for curved baselines - implement
     * Asymmetric Least Squares: handles complex baselines
     * Adaptive PLS (Partial Least Squares) with additional peak scaling - implement
   - Normalization methods:
     * Vector normalization: standard for FTIR
     * Min-max: when absolute intensities matter
     * SNV: reduces multiplicative effects

4. FEATURE ENHANCEMENT
   - Derivative spectroscopy (if needed):
     * First derivative: enhances edges
     * Second derivative: enhances peaks
     * Note: Apply after smoothing, increases noise sensitivity

5. QUALITY CONTROL
   - Outlier detection:
     * Mahalanobis distance for multivariate outliers
     * Remove spectra > 3 std from mean
   - Data validation:
     * Check for NaN/Inf values
     * Verify signal-to-noise ratio
     * Confirm normalization success
     * Log all removed spectra
"""